"""
Tests for the IEEE 118-bus system (pandapower case118).

1. Dimensions match Table 1: uncertainties 99 x 2 + 7 = 205, policy
   input 205 + 54 (previous p_g of the 53 non-slack units + previous
   slack output), policy output 54 + 53, HELM output
   2(118 + 54 + 173) = 690; minibatch 128, lr 5e-4, layers 128.
2. Critic vs pandapower on random states and actions (case118 has a 30°
   slack angle and 13 transformers): slack P, unit Q, PQ voltages, line
   currents, cost, reward and every z entry.
3. gradcheck of r and z with respect to action and state.
4. Previous-slack input: the rollout feeds the normalized, clipped slack
   output of step t-1 to the policy; gradients reach every weight; the
   plain forward() refuses (the slack is not an action).
5. Sequential ramp-limited OPF reference on two short profiles.
6. GPU (only if CUDA is available): critic outputs on cuda equal the CPU
   ones, and a training step runs on cuda.

Run:
    python scripts/test_case118.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandapower as pp
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_dhelm_critic import reference  # noqa: E402

from dhelm_opf.algorithms.dhelm_cpo import CPOConfig  # noqa: E402
from dhelm_opf.algorithms.setup import make_agent  # noqa: E402
from dhelm_opf.environment.cases import get_case  # noqa: E402
from dhelm_opf.environment.uncertainty import (  # noqa: E402
    OperatingScenario,
    apply_demand_uncertainty,
    apply_wind_output,
)
from dhelm_opf.helm.solver import HelmSolverConfig  # noqa: E402
from dhelm_opf.models.action_config import RampConfig  # noqa: E402
from dhelm_opf.models.action_decoder import (  # noqa: E402
    ActionDecoder,
    apply_action_to_network,
)
from dhelm_opf.utils.evaluation import check_opf_reference  # noqa: E402
from dhelm_opf.utils.opf_reference import compute_reference  # noqa: E402

F64 = torch.float64
CASE = "case118"


def as_t(x, dev="cpu"):
    return torch.as_tensor(np.asarray(x), dtype=F64, device=dev)


def random_batch(agent, rng, n, action_range):
    d = agent.dhelm
    s = np.concatenate([rng.uniform(0.6, 1.1, (n, 2 * d.n_d)),
                        rng.uniform(0.4, 1.0, (n, d.n_w))], axis=1)
    a = rng.uniform(-action_range, action_range, (n, agent.action_config.action_dim))
    return s, a


def test_dimensions(agent) -> None:

    net, d, pol = agent.net, agent.dhelm, agent.policy
    n_units = len(net.gen) + len(net.sgen) + len(net.ext_grid)

    assert len(net.load) == 99 and len(net.line) == 173 and len(net.trafo) == 13
    assert d.state_dim == 99 * 2 + 7
    assert pol.input_dim == 205 + 54 and pol.prev_slack
    assert (agent.action_config.n_v, agent.action_config.n_g) == (54, 53)
    assert 2 * (len(net.bus) + n_units + len(net.line)) == 690
    assert (agent.cpo.config.batch_size, agent.cpo.config.lr) == (128, 5e-4)
    assert pol.lstm.hidden_size == 128

    print(f"  1. Table 1 dimensions             : OK (state 205, input 205+54, "
          f"output 54+53, HELM 690, batch 128, lr 5e-4, width 128; "
          f"{len(agent.critic.constraint_names)} constraints)")


def solved_net(agent, state, action):
    net = get_case(CASE).build()
    n_d = len(net.load)
    sc = OperatingScenario(state[:n_d], state[n_d:2 * n_d], state[2 * n_d:])
    apply_demand_uncertainty(net, sc)
    with contextlib.redirect_stdout(io.StringIO()):
        apply_wind_output(net, agent.uncertainty.B_w, sc.p_w_scale,
                          agent.uncertainty.p_w_rated_mw)
    cfg = agent.action_config
    apply_action_to_network(net, cfg, ActionDecoder(cfg).decode(action))
    pp.runpp(net, tolerance_mva=1e-10)
    return net


def test_vs_pandapower(agent, rng, batch=8) -> None:

    critic = agent.critic
    s, a = random_batch(agent, rng, batch, action_range=0.2)
    out = critic(as_t(s), as_t(a))
    assert bool(out.helm.converged.all())

    keys = ("p_g_s", "q_g", "v_q", "i_f", "i_t", "f_sum", "r", "z")
    err = dict.fromkeys(keys, 0.0)
    for k in range(batch):
        ref = reference(solved_net(agent, s[k], a[k]), list(agent.action_config.B_vs))
        mine = dict(
            p_g_s=out.p_g_s[k].item(), q_g=out.q_g[k].detach().numpy(),
            v_q=out.v_q[k].detach().numpy(), i_f=out.i_f[k].detach().numpy(),
            i_t=out.i_t[k].detach().numpy(), f_sum=out.f_sum[k].item(),
            r=out.r[k].item(), z=out.z[k].detach().numpy(),
        )
        for key in keys:
            err[key] = max(err[key], float(np.max(np.abs(
                np.asarray(mine[key]) - np.asarray(ref[key])))))

    tol = dict(p_g_s=1e-5, q_g=1e-5, v_q=1e-8, i_f=1e-8, i_t=1e-8,
               f_sum=1e-3, r=1e-8, z=1e-7)
    for key in keys:
        assert err[key] < tol[key], (key, err[key])

    print(f"  2. critic vs pandapower ({batch} samples) : OK (max errors: "
          + ", ".join(f"{k} {v:.0e}" for k, v in err.items()) + ")")


def test_gradcheck(agent, rng) -> None:

    s, a = random_batch(agent, rng, 1, action_range=0.1)
    s_t, a_t = as_t(s).requires_grad_(True), as_t(a).requires_grad_(True)

    def f(s_, a_):
        out = agent.critic(s_, a_, N=4)
        return out.r, out.z.sum(dim=1)

    assert torch.autograd.gradcheck(f, (s_t, a_t), eps=1e-6, atol=1e-5)
    print("  3. gradcheck (reward, z)           : passed")


def test_prev_slack() -> None:

    agent = make_agent(CASE, seed=0, ramp=RampConfig(),
                       cpo_config=CPOConfig(lr=5e-4, batch_size=4, window=3))
    pol, acfg = agent.policy, agent.action_config
    s = torch.as_tensor(agent.sampler.sample(4 * 3).reshape(4, 3, -1), dtype=torch.float32)

    seen = []
    orig = pol.step

    def spy(s_t, prev, hc=None):
        seen.append(prev.detach().clone())
        return orig(s_t, prev, hc)

    pol.step = spy
    ev = agent.cpo.evaluate(s)
    pol.step = orig

    assert seen[0].shape == (4, 54) and torch.all(seen[0] == 0)
    p_s0 = ev.p_g_s.reshape(4, 3)[:, 0].double()
    expect = (2 * (p_s0 - acfg.G_s.P_min) / (acfg.G_s.P_max - acfg.G_s.P_min) - 1).clamp(-1, 1)
    assert torch.allclose(seen[1][:, -1].double(), expect, atol=1e-6)

    agent.cpo.optimizer.zero_grad()
    agent.cpo.loss(ev).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in pol.parameters())

    try:
        pol(s)
        raise AssertionError("forward() should refuse with prev_slack")
    except RuntimeError:
        pass

    print("  4. previous-slack policy input     : OK (fed from the critic, "
          "gradients reach all weights)")


def test_opf_reference() -> None:

    ramp = RampConfig()
    agent = make_agent(CASE, seed=0, ramp=ramp)
    L = 2
    states = agent.sampler.sample(2 * L)
    ref = compute_reference(states, agent.uncertainty, agent.action_config,
                            False, ramp=ramp, episode_length=L, case=CASE)
    assert ref["converged"].all()
    check = check_opf_reference(agent, ref)
    assert check["cost_mismatch_max"] < 0.5, check       # on ~1e5 $/h
    assert check["violation_max"] < 1e-4, check
    print(f"  5. sequential OPF reference       : OK (mean {np.mean(ref['cost']):.0f} $/h; "
          f"OPF@critic cost err {check['cost_mismatch_max']:.1e}, z {check['violation_max']:.1e})")


def test_gpu(rng) -> None:

    if not torch.cuda.is_available():
        print("  6. GPU                             : skipped (no CUDA)")
        return

    cpu = make_agent(CASE, seed=0)
    gpu = make_agent(CASE, seed=0, device="cuda")
    s, a = random_batch(cpu, rng, 16, action_range=0.2)
    o_c = cpu.critic(as_t(s), as_t(a))
    o_g = gpu.critic(as_t(s, "cuda"), as_t(a, "cuda"))
    assert torch.allclose(o_c.r, o_g.r.cpu(), atol=1e-9)
    assert torch.allclose(o_c.z, o_g.z.cpu(), atol=1e-8)

    g = make_agent(CASE, seed=0, ramp=RampConfig(), device="cuda",
                   cpo_config=CPOConfig(lr=5e-4, batch_size=8, window=3))
    m = g.cpo.train_step(g.sampler.sample(8 * 3).reshape(8, 3, -1))
    assert np.isfinite(m["loss"])
    print(f"  6. GPU                             : OK (critic cuda == cpu; "
          f"training step on {torch.cuda.get_device_name(0)})")


def main() -> None:

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("IEEE 118-bus tests")
    agent = make_agent(CASE, seed=0,
                       solver_config=HelmSolverConfig(n_max=60, epsilon=1e-10))
    test_dimensions(agent)
    test_vs_pandapower(agent, rng)
    test_gradcheck(agent, rng)
    test_prev_slack()
    test_opf_reference()
    test_gpu(rng)
    print("All 118-bus tests passed.")


if __name__ == "__main__":
    main()
