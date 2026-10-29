"""
Evaluate trained policies (optionally refined) against the OPF reference.

Used by scripts/evaluate_vs_opf.py (one run) and scripts/run_seeds.py
(many seeds and variants), so both report exactly the same numbers.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from dhelm_opf.algorithms.case5_setup import Case5Agent, make_case5_agent
from dhelm_opf.algorithms.dhelm_cpo import CPOConfig
from dhelm_opf.algorithms.refine import RefineConfig, refine_actions
from dhelm_opf.environment.state_sampler import StateSampler
from dhelm_opf.models.action_config import RampConfig

FEASIBLE_TOL = 1e-6
TEST_SEED = 20_000
RAMP_BIND_TOL_MW = 1.0      # "at a ramp limit": within 1 MW of the edge


def ramp_from_args(args: dict) -> RampConfig | None:
    """The RampConfig a run was trained with (None: no ramping)."""

    frac = args.get("ramp_frac")
    return RampConfig(frac_u=frac, frac_d=frac) if frac else None


def load_agent(run_dir: str | Path, checkpoint: str = "best"):
    """Rebuild the agent of a training run and load a checkpoint."""

    run_dir = Path(run_dir)
    fname = {"best": "policy_best.pt", "last": "policy.pt"}[checkpoint]
    path = run_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found (runs trained before best-checkpoint "
            "support only have policy.pt; use --checkpoint last)."
        )

    ckpt = torch.load(path, weights_only=False)
    args = ckpt["args"]

    # Checkpoints written before the rename to paper symbols use the
    # old configuration keys; map them so old runs still load.
    renamed = {"lambda_init": "lam_0", "eta_lambda": "eta_lam",
               "lambda_max": "lam_max"}
    cpo_config = {renamed.get(k, k): v for k, v in ckpt["cpo_config"].items()}
    lam = ckpt["lam"] if "lam" in ckpt else ckpt["lambdas"]

    # Runs made before the Eq. (4) order change (p_g first) have no
    # "p_first" entry and used the voltage-first layout.
    agent = make_case5_agent(
        seed=args["seed"], cpo_config=CPOConfig(**cpo_config),
        p_first=args.get("p_first", False), ramp=ramp_from_args(args),
    )
    agent.policy.load_state_dict(ckpt["policy"])
    agent.policy.eval()
    agent.cpo.lam = lam.to(torch.float64)

    return agent, ckpt


def test_states(agent: Case5Agent, n: int = 100, seed: int = TEST_SEED):
    """The held-out test states (same draw for every run)."""

    sampler = StateSampler(agent.dhelm.n_d, agent.dhelm.n_w,
                           agent.uncertainty, seed=seed)
    return sampler.sample(n)


def check_opf_reference(agent: Case5Agent, ref: dict) -> dict:
    """
    Pass the OPF solutions through the critic: cost must match the OPF
    cost and violations must be ~0 (the OPF point is feasible for the
    critic).  With ramping, steps t > 0 are checked inside the ramp
    window around the OPF's own previous dispatch.
    """

    ok = ref["converged"]
    has_prev = ref.get("has_prev", np.zeros_like(ok)) & ok
    first = ok & ~has_prev

    f_sum = np.full(len(ok), np.nan)
    z_sum = np.full(len(ok), np.nan)

    with torch.no_grad():
        for sel, prev in ((first, False), (has_prev, True)):
            if not sel.any():
                continue
            kw = {}
            if prev and agent.critic.ramping:
                kw = dict(
                    p_g_prev=torch.as_tensor(ref["p_g_prev"][sel]),
                    p_g_s_prev=torch.as_tensor(ref["p_g_s_prev"][sel]),
                )
            out = agent.critic(torch.as_tensor(ref["states"][sel]),
                               torch.as_tensor(ref["action"][sel]), **kw)
            f_sum[sel] = out.f_sum.numpy()
            z_sum[sel] = out.z.sum(dim=1).numpy()

    return {
        "opf_converged": int(ok.sum()),
        "cost_mismatch_max": float(np.max(np.abs(f_sum[ok] - ref["cost"][ok]))),
        "violation_max": float(np.max(z_sum[ok])),
    }


def ramp_binding(agent: Case5Agent, p_g, p_g_s, episode_length: int,
                 tol: float = RAMP_BIND_TOL_MW) -> dict:
    """
    How often the Eq. (3) ramp limits bind, over steps t > 0.

    A unit "binds" at a step when its ramp window edge is tighter than
    the static limit AND its dispatch is within `tol` MW of that edge.
    Slack included (Eq. 16b).  Returns the share of unit-steps and the
    share of steps with at least one binding unit.
    """

    cfg, ramp = agent.action_config, agent.dhelm.ramp
    units = list(cfg.G_ns) + [cfg.G_s]
    P = np.column_stack([np.asarray(p_g), np.asarray(p_g_s)])
    P = P.reshape(-1, episode_length, len(units))
    prev, cur = P[:, :-1], P[:, 1:]

    P_min = np.array([u.P_min for u in units])
    P_max = np.array([u.P_max for u in units])
    R_u = np.array([ramp.R_u(u) for u in units])
    R_d = np.array([ramp.R_d(u) for u in units])

    up_edge, dn_edge = prev + R_u, prev - R_d
    up = (up_edge < P_max - tol) & (cur >= up_edge - tol)
    dn = (dn_edge > P_min + tol) & (cur <= dn_edge + tol)
    bind = up | dn

    return {
        "ramp_bind_unit_share": float(bind.mean()) if bind.size else 0.0,
        "ramp_bind_step_share": float(bind.any(axis=2).mean()) if bind.size else 0.0,
        "ramp_bind_by_unit": [float(x) for x in bind.reshape(-1, len(units)).mean(axis=0)],
    }


def evaluate_policy(
    agent: Case5Agent,
    states: np.ndarray,
    window: int,
    ref: dict,
    refine: RefineConfig | None = None,
) -> dict:
    """
    Policy (+ optional R1/R2 refinement) on `states` vs the OPF reference.

    The policy runs over consecutive windows of `window` states, as in
    training.  Returns a summary dict plus per-state arrays.
    """

    S, d = states.shape
    if S % window:
        raise ValueError(f"{S} states cannot be split into windows of {window}.")

    if agent.critic.ramping:
        roll = _rollout_ramped(agent, states, window, refine)
        a_hat, cost, z = roll["a_hat"], roll["cost"], roll["z"]
        t_policy, t_refine = roll["t_policy"], roll["t_refine"]
        n_refined, steps_mean = roll["n_refined"], roll["steps_mean"]
    else:
        windows = torch.as_tensor(states.reshape(-1, window, d), dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            ev = agent.cpo.evaluate(windows)
        t_policy = time.perf_counter() - t0

        a_hat = ev.a_hat.reshape(S, -1).to(torch.float64)
        cost = ev.f_sum.numpy()
        z = ev.z.numpy()
        t_refine = 0.0
        n_refined = 0
        steps_mean = 0.0

        if refine is not None:
            r = refine_actions(agent.critic, agent.cpo.lam,
                               torch.as_tensor(states), a_hat, refine)
            a_hat, cost, z = r.a_hat, r.f_sum.numpy(), r.z.numpy()
            t_refine = r.seconds
            n_refined = int(r.refined.sum())
            steps_mean = float(r.steps_taken[r.refined].double().mean()) if n_refined else 0.0

    viol = z.sum(axis=1)
    feas = viol <= FEASIBLE_TOL

    opf = ref["cost"]
    ok = ref["converged"]
    gap = (cost - opf) / opf
    sel = ok & feas

    def stat(f, x):
        return float(f(x)) if len(x) else float("nan")

    summary = {
        "states": int(S),
        "opf_converged": int(ok.sum()),
        "feasible": float(feas.mean()),
        "violation_mean": float(viol.mean()),
        "cost_mean": float(cost[ok].mean()),
        "opf_cost_mean": float(opf[ok].mean()),
        "gap_mean": stat(np.mean, gap[ok]),
        "gap_median": stat(np.median, gap[ok]),
        "gap_p95": stat(lambda x: np.percentile(x, 95), gap[ok]),
        "gap_max": stat(np.max, gap[ok]),
        "gap_feasible_mean": stat(np.mean, gap[sel]),
        "refined_states": n_refined,
        "refine_steps_mean": steps_mean,
        "ms_policy_total": 1e3 * t_policy,
        "ms_refine_total": 1e3 * t_refine,
        "ms_total_per_state": 1e3 * (t_policy + t_refine) / S,
    }

    if agent.critic.ramping:
        summary.update(ramp_binding(agent, roll["p_g"], roll["p_g_s"], window))
        if "p_g" in ref and ok.all():
            opf_bind = ramp_binding(agent, ref["p_g"], ref["p_g_s"], window)
            summary.update({f"opf_{k}": v for k, v in opf_bind.items()})

    per_state = {
        "cost": cost, "opf_cost": opf, "gap": gap, "violation": viol,
        "z": z, "a_hat": np.asarray(a_hat),
    }

    return {"summary": summary, "per_state": per_state}


def _rollout_ramped(agent: Case5Agent, states: np.ndarray, window: int,
                    refine: RefineConfig | None) -> dict:
    """
    Step-by-step rollout with ramping, optionally refining each step.

    At every t the policy acts given the previous dispatch, the action
    is (optionally) refined inside the same ramp window, and the
    REFINED dispatch becomes the previous dispatch for t + 1 (both the
    ramp window and the policy's extra input), as it would online.
    """

    S, d = states.shape
    E = S // window
    s_all = torch.as_tensor(states.reshape(E, window, d), dtype=torch.float64)

    policy, critic = agent.policy, agent.critic
    P_min = torch.as_tensor(agent.action_config.P_min, dtype=torch.float64)
    P_max = torch.as_tensor(agent.action_config.P_max, dtype=torch.float64)

    hc = None
    prev_in = torch.zeros(E, policy.n_g, dtype=policy.state_low.dtype)
    p_g_prev = p_g_s_prev = None

    a_hat, cost, z, p_g, p_g_s = [], [], [], [], []
    t_policy = t_refine = 0.0
    n_refined, steps = 0, 0

    for t in range(window):
        s_t = s_all[:, t]

        t0 = time.perf_counter()
        with torch.no_grad():
            a_t, hc = policy.step(s_t.to(prev_in.dtype), prev_in, hc)
            a_t = a_t.to(torch.float64)
            out = critic(s_t, a_t, p_g_prev=p_g_prev, p_g_s_prev=p_g_s_prev)
        t_policy += time.perf_counter() - t0

        f_t, z_t, pg_t, pgs_t = out.f_sum, out.z, out.p_g, out.p_g_s

        if refine is not None:
            r = refine_actions(critic, agent.cpo.lam, s_t, a_t, refine,
                               p_g_prev=p_g_prev, p_g_s_prev=p_g_s_prev)
            a_t, f_t, z_t, pg_t, pgs_t = r.a_hat, r.f_sum, r.z, r.p_g, r.p_g_s
            t_refine += r.seconds
            n_refined += int(r.refined.sum())
            steps += int(r.steps_taken[r.refined].sum())

        a_hat.append(a_t.detach())
        cost.append(f_t.detach())
        z.append(z_t.detach())
        p_g.append(pg_t.detach())
        p_g_s.append(pgs_t.detach())

        p_g_prev, p_g_s_prev = pg_t.detach(), pgs_t.detach()
        prev_in = (2.0 * (p_g_prev - P_min) / (P_max - P_min) - 1.0).to(prev_in.dtype)

    def flat(xs):
        x = torch.stack(xs, dim=1)              # (E, L, ...) -> order (e, t)
        return x.reshape(S, *x.shape[2:]).numpy()

    return {
        "a_hat": flat(a_hat), "cost": flat(cost), "z": flat(z),
        "p_g": flat(p_g), "p_g_s": flat(p_g_s),
        "t_policy": t_policy, "t_refine": t_refine,
        "n_refined": n_refined,
        "steps_mean": steps / n_refined if n_refined else 0.0,
    }


def opf_cache_path(n: int, seed: int, fixed_slack_voltage: bool,
                   root: str | Path = "results/opf_reference",
                   ramp: RampConfig | None = None,
                   episode_length: int | None = None) -> Path:
    suffix = "_fixed_slack" if fixed_slack_voltage else ""
    if ramp is not None:
        suffix += f"_ramp{ramp.frac_u:g}-{ramp.frac_d:g}_L{episode_length}"
    return Path(root) / f"case5_test_seed{seed}_n{n}{suffix}.npz"
