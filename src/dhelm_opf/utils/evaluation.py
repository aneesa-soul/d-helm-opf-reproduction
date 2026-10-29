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

FEASIBLE_TOL = 1e-6
TEST_SEED = 20_000


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

    agent = make_case5_agent(
        seed=args["seed"], cpo_config=CPOConfig(**cpo_config)
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
    critic).
    """

    ok = ref["converged"]
    with torch.no_grad():
        out = agent.critic(torch.as_tensor(ref["states"][ok]),
                           torch.as_tensor(ref["action"][ok]))
    return {
        "opf_converged": int(ok.sum()),
        "cost_mismatch_max": float(np.max(np.abs(out.f_sum.numpy()
                                                 - ref["cost"][ok]))),
        "violation_max": float(out.z.sum(dim=1).max()),
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

    per_state = {
        "cost": cost, "opf_cost": opf, "gap": gap, "violation": viol,
        "z": z, "a_hat": np.asarray(a_hat),
    }

    return {"summary": summary, "per_state": per_state}


def opf_cache_path(n: int, seed: int, fixed_slack_voltage: bool,
                   root: str | Path = "results/opf_reference") -> Path:
    suffix = "_fixed_slack" if fixed_slack_voltage else ""
    return Path(root) / f"case5_test_seed{seed}_n{n}{suffix}.npz"
