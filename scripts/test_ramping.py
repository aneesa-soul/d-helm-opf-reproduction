"""
Tests for the Eq. (4) action order and Eq. (3) ramping.

1. Eq. (4) layout: a = [p_g of G_ns, |v| of B_vs]; p_first=False keeps
   the old voltage-first layout.
2. Ramp window (Eq. 3 via Eq. 18): a_hat = -1 / +1 maps to
   max(P_min, p_prev - R_d) / min(P_max, p_prev + R_u); without a
   previous dispatch the static [P_min, P_max] is used.
3. Rollout: |p_{g,t} - p_{g,t-1}| <= R for every non-slack unit, and the
   slack ramp (16b) is scored inside the rollout.
4. Gradient through time: z_t depends on a_{t-1}.
5. Refinement keeps the ramp window.
6. Sequential OPF reference: every unit (slack included) respects the
   ramp around the OPF's own previous dispatch, and the OPF solutions
   pass through the ramp-aware critic with matching cost and z ~ 0.

Run:
    python scripts/test_ramping.py
"""

from __future__ import annotations

import numpy as np
import torch

from dhelm_opf.algorithms.case5_setup import make_case5_agent
from dhelm_opf.algorithms.refine import RefineConfig, refine_actions
from dhelm_opf.models.action_config import RampConfig
from dhelm_opf.utils.evaluation import (
    check_opf_reference,
    evaluate_policy,
    ramp_binding,
)
from dhelm_opf.utils.opf_reference import compute_reference

F64 = torch.float64


def test_layout() -> None:

    new = make_case5_agent(seed=0).action_config
    old = make_case5_agent(seed=0, p_first=False).action_config

    assert new.generator_action_indices == (0, 1, 2, 3)
    assert new.voltage_action_indices == (4, 5, 6, 7)
    assert old.voltage_action_indices == (0, 1, 2, 3)
    assert old.generator_action_indices == (4, 5, 6, 7)

    print("  1. Eq. (4) layout (p_g first)       : OK")


def test_window(agent) -> None:

    cfg, ramp, critic = agent.action_config, agent.dhelm.ramp, agent.critic
    idx_p = list(cfg.generator_action_indices)
    P_min, P_max = cfg.P_min, cfg.P_max
    R_u = np.array([ramp.R_u(u) for u in cfg.G_ns])
    R_d = np.array([ramp.R_d(u) for u in cfg.G_ns])

    s = torch.as_tensor(agent.sampler.sample(3), dtype=F64)
    # previous dispatch: one near P_min, one in the middle, one near P_max
    prev = np.stack([P_min + 1.0, 0.5 * (P_min + P_max), P_max - 1.0])
    prev_t = torch.as_tensor(prev, dtype=F64)

    for value, expect in (
        (-1.0, np.maximum(P_min, prev - R_d)),
        (+1.0, np.minimum(P_max, prev + R_u)),
    ):
        a = torch.zeros(3, cfg.action_dim, dtype=F64)
        a[:, idx_p] = value
        out = critic(s, a, p_g_prev=prev_t)
        assert np.allclose(out.p_g.detach().numpy(), expect, atol=1e-9)

    a = torch.zeros(3, cfg.action_dim, dtype=F64)
    a[:, idx_p] = 1.0
    out = critic(s, a)                      # no previous step
    assert np.allclose(out.p_g.detach().numpy(), np.tile(P_max, (3, 1)))

    print("  2. ramp window endpoints, no prev   : OK")


def test_rollout(agent) -> None:

    cfg, ramp = agent.action_config, agent.dhelm.ramp
    R = np.array([ramp.R_u(u) for u in cfg.G_ns])   # R_u = R_d here

    s = torch.as_tensor(agent.sampler.sample(8 * 25).reshape(8, 25, -1),
                        dtype=torch.float32)
    with torch.no_grad():
        ev = agent.cpo.evaluate(s)
    p = ev.p_g.reshape(8, 25, -1).numpy()
    dp = np.abs(np.diff(p, axis=1))
    assert np.all(dp <= R + 1e-9), dp.max(axis=(0, 1))

    # Slack (16b) is scored against the previous slack output.
    k = agent.critic.constraint_names.index("p_g[slack]")
    p_s = ev.p_g_s.reshape(8, 25).numpy()
    R_s = ramp.R_u(cfg.G_s)
    eta = 1.0 / (cfg.G_s.P_max - cfg.G_s.P_min)
    over = np.maximum(np.abs(np.diff(p_s, axis=1)) - R_s, 0.0)
    z_s = ev.z[:, k].reshape(8, 25).numpy()[:, 1:]
    static = np.maximum(p_s - cfg.G_s.P_max, 0) + np.maximum(cfg.G_s.P_min - p_s, 0)
    assert np.all(z_s + 1e-9 >= eta * over)          # ramp part is included
    assert np.all(z_s <= eta * (over + static[:, 1:]) + 1e-9)

    print(f"  3. rollout |dp_g| <= R, slack 16b   : OK "
          f"(max |dp_g|/R = {np.max(dp / R):.2f})")


def test_gradient_through_time(agent) -> None:

    cfg, critic = agent.action_config, agent.critic
    s = torch.as_tensor(agent.sampler.sample(4), dtype=F64)
    a_prev = torch.zeros(4, cfg.action_dim, dtype=F64, requires_grad=True)
    out_prev = critic(s, a_prev)
    a_t = torch.full((4, cfg.action_dim), 0.9, dtype=F64)
    out_t = critic(s, a_t, p_g_prev=out_prev.p_g, p_g_s_prev=out_prev.p_g_s)
    (out_t.f_sum.sum() + out_t.z.sum()).backward()
    assert a_prev.grad.abs().sum() > 0

    print("  4. gradient flows to a_{t-1}        : OK")


def test_refine_keeps_window(agent) -> None:

    cfg, ramp, critic = agent.action_config, agent.dhelm.ramp, agent.critic
    s = torch.as_tensor(agent.sampler.sample(16), dtype=F64)
    prev = torch.as_tensor(
        np.random.default_rng(1).uniform(cfg.P_min, cfg.P_max, (16, cfg.n_g)),
        dtype=F64)
    prev_s = torch.full((16,), 100.0, dtype=F64)
    a = torch.zeros(16, cfg.action_dim, dtype=F64)

    r = refine_actions(critic, agent.cpo.lam, s, a,
                       RefineConfig(mode="r2", K=10, lr=0.1),
                       p_g_prev=prev, p_g_s_prev=prev_s)
    R = torch.as_tensor([ramp.R_u(u) for u in cfg.G_ns], dtype=F64)
    assert torch.all((r.p_g - prev).abs() <= R + 1e-9)

    print("  5. refinement stays in ramp window  : OK")


def test_sequential_opf() -> None:

    # 30% (the paper-protocol value).  Tighter ramps make some of these
    # independent states infeasible (net demand outside the reachable
    # window), which the OPF then correctly fails to solve.
    ramp = RampConfig(frac_u=0.30, frac_d=0.30)
    agent = make_case5_agent(seed=0, ramp=ramp)
    cfg, L = agent.action_config, 5

    states = agent.sampler.sample(2 * L)
    ref = compute_reference(states, agent.uncertainty, cfg, False,
                            ramp=ramp, episode_length=L)
    assert ref["converged"].all()
    assert not ref["has_prev"][0] and not ref["has_prev"][L]
    assert ref["has_prev"][1:L].all()

    units = list(cfg.G_ns) + [cfg.G_s]
    R = np.array([ramp.R_u(u) for u in units])
    P = np.column_stack([ref["p_g"], ref["p_g_s"]]).reshape(2, L, -1)
    assert np.all(np.abs(np.diff(P, axis=1)) <= R + 1e-4)

    check = check_opf_reference(agent, ref)
    # The OPF meets its bounds to its own tolerance (~1e-4 MW), and a_hat
    # is clipped to [-1, 1]: a few 1e-2 $/h on ~8,500 $/h is round-off.
    assert check["cost_mismatch_max"] < 0.1, check
    assert check["violation_max"] < 1e-6, check

    bind = ramp_binding(agent, ref["p_g"], ref["p_g_s"], L)

    # The ramped evaluation path runs end to end on this reference.
    res = evaluate_policy(agent, states, L, ref,
                          RefineConfig(mode="r1", K=3, lr=0.03))
    assert np.isfinite(res["summary"]["gap_mean"])

    print(f"  6. sequential OPF respects ramps    : OK "
          f"(binds on {bind['ramp_bind_step_share']:.0%} of steps; "
          f"OPF@critic cost err {check['cost_mismatch_max']:.1e}, "
          f"z {check['violation_max']:.1e})")


def main() -> None:

    torch.manual_seed(0)
    agent = make_case5_agent(seed=0, ramp=RampConfig())

    print("Ramping tests (Eq. 3, Eq. 4 layout)")
    test_layout()
    test_window(agent)
    test_rollout(agent)
    test_gradient_through_time(agent)
    test_refine_keeps_window(agent)
    test_sequential_opf()
    print("All ramping tests passed.")


if __name__ == "__main__":
    main()
