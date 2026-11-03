"""
Tests for the IEEE 30-bus system (pandapower case30).

1. Dimensions match Table 1: uncertainties 20 x 2 + 4 = 44, policy
   input 44 + 5, policy output 6 + 5, HELM output 2(30 + 6 + 41) = 154;
   minibatch 64 and lr 1e-3; wind farms per environment/cases.py.
2. Critic vs pandapower: slack P, unit Q per voltage bus, PQ voltages,
   line currents, cost, reward and every z entry match an independent
   computation from pandapower's power flow (same reference as
   test_dhelm_critic.py) on random states and actions.
3. gradcheck of r and z with respect to action and state.
4. Sequential ramp-limited OPF reference: solves, respects ramps, and its
   solutions pass through the critic with matching cost and z ~ 0.
5. A short ramped training run on case30 runs and lowers the loss.

Run:
    python scripts/test_case30.py
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
from test_dhelm_critic import reference  # noqa: E402  (same NumPy reference)

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


def as_t(x):
    return torch.as_tensor(np.asarray(x), dtype=F64)


def test_dimensions(agent) -> None:

    net, d = agent.net, agent.dhelm
    spec = get_case("case30")
    n_units = len(net.gen) + len(net.sgen) + len(net.ext_grid)   # case30 has no sgens

    assert len(net.load) == 20 and len(net.line) == 41 and len(net.trafo) == 0
    assert d.state_dim == 20 * 2 + 4
    assert agent.policy.input_dim == 44 + 5
    assert (agent.action_config.n_v, agent.action_config.n_g) == (6, 5)
    assert 2 * (len(net.bus) + n_units + len(net.line)) == 154
    assert (agent.cpo.config.batch_size, agent.cpo.config.lr) == (64, 1e-3)
    assert agent.uncertainty.B_w == spec.B_w

    print(f"  1. Table 1 dimensions             : OK (state 44, input 49, "
          f"output 6+5, HELM 154, batch 64; wind at buses {spec.B_w})")


def random_batch(agent, rng, n, action_range=1.0):
    d = agent.dhelm
    s = np.concatenate([rng.uniform(0.6, 1.1, (n, 2 * d.n_d)),
                        rng.uniform(0.4, 1.0, (n, d.n_w))], axis=1)
    a = rng.uniform(-action_range, action_range, (n, agent.action_config.action_dim))
    return s, a


def solved_net(agent, state, action):
    net = get_case("case30").build()
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


def test_vs_pandapower(agent, rng, batch=30) -> None:

    critic = agent.critic
    s, a = random_batch(agent, rng, batch, action_range=0.8)
    out = critic(as_t(s), as_t(a))
    assert bool(out.helm.converged.all())

    keys = ("p_g_s", "q_g", "v_q", "i_f", "i_t", "f_sum", "r", "z")
    err = dict.fromkeys(keys, 0.0)
    violated = np.zeros(len(critic.constraint_names), dtype=int)

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
        violated += ref["z"] > 0

    tol = dict(p_g_s=1e-5, q_g=1e-5, v_q=1e-8, i_f=1e-8, i_t=1e-8,
               f_sum=1e-4, r=1e-8, z=1e-7)
    for key in keys:
        assert err[key] < tol[key], (key, err[key])

    print(f"  2. critic vs pandapower ({batch} samples): OK (max errors: "
          + ", ".join(f"{k} {v:.0e}" for k, v in err.items())
          + f"; {int((violated > 0).sum())}/{len(violated)} constraints violated at least once)")


def test_gradcheck(agent, rng) -> None:

    s, a = random_batch(agent, rng, 2, action_range=0.5)
    s_t, a_t = as_t(s).requires_grad_(True), as_t(a).requires_grad_(True)

    def f(s_, a_):
        out = agent.critic(s_, a_, N=6)
        return out.r, out.z

    assert torch.autograd.gradcheck(f, (s_t, a_t), eps=1e-6, atol=1e-5)
    print("  3. gradcheck (reward, z)           : passed")


def test_sequential_opf() -> None:

    ramp = RampConfig()
    agent = make_agent("case30", seed=0, ramp=ramp)
    L = 3
    states = agent.sampler.sample(2 * L)
    ref = compute_reference(states, agent.uncertainty, agent.action_config,
                            False, ramp=ramp, episode_length=L, case="case30")
    assert ref["converged"].all()

    units = list(agent.action_config.G_ns) + [agent.action_config.G_s]
    R = np.array([ramp.R_u(u) for u in units])
    P = np.column_stack([ref["p_g"], ref["p_g_s"]]).reshape(2, L, -1)
    assert np.all(np.abs(np.diff(P, axis=1)) <= R + 1e-4)

    check = check_opf_reference(agent, ref)
    assert check["cost_mismatch_max"] < 0.01, check
    assert check["violation_max"] < 1e-6, check

    print(f"  4. sequential OPF reference       : OK (mean {np.mean(ref['cost']):.1f} $/h; "
          f"OPF@critic cost err {check['cost_mismatch_max']:.1e}, z {check['violation_max']:.1e})")


def test_training_smoke() -> None:

    agent = make_agent("case30", seed=0, ramp=RampConfig(),
                       cpo_config=CPOConfig(lr=1e-3, batch_size=8, window=4))
    s = torch.as_tensor(agent.sampler.sample(8 * 4).reshape(8, 4, -1), dtype=torch.float32)
    l0 = agent.cpo.loss(agent.cpo.evaluate(s)).item()
    for _ in range(20):
        agent.cpo.train_step(s)
    l1 = agent.cpo.loss(agent.cpo.evaluate(s)).item()
    assert np.isfinite(l1) and l1 < l0, (l0, l1)
    print(f"  5. training smoke run (20 steps)  : OK (loss {l0:.3f} -> {l1:.3f})")


def main() -> None:

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("IEEE 30-bus tests")

    agent = make_agent("case30", seed=0,
                       solver_config=HelmSolverConfig(n_max=60, epsilon=1e-10))
    test_dimensions(agent)
    test_vs_pandapower(agent, rng)
    test_gradcheck(agent, rng)
    test_sequential_opf()
    test_training_smoke()
    print("All 30-bus tests passed.")


if __name__ == "__main__":
    main()
