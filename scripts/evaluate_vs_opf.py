"""
Evaluate one trained D-HELM policy against pandapower's AC OPF.

For each held-out test state (seed 20000, never used in training):
    policy           action -> D-HELM critic -> cost, violations z
    + refinement     optional R1 / R2 gradient steps on the action
                     (algorithms/refine.py, paper Remark 5)
    OPF              pandapower runopp, matched to the paper's problem
                     and cached (utils/opf_reference.py)
    OPF@critic       the OPF solutions passed through the critic: their
                     cost must equal the OPF cost and z must be ~0

Reported: feasibility (sum z <= 1e-6), optimality gap
(policy - OPF) / OPF, and timing.

Examples:
    python scripts/evaluate_vs_opf.py results/runs/case5_L1_ep100
    python scripts/evaluate_vs_opf.py results/runs/case5_L1_ep100 --checkpoint last
    python scripts/evaluate_vs_opf.py results/runs/case5_L1_ep100 --refine r2 --refine-K 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dhelm_opf.algorithms.refine import RefineConfig
from dhelm_opf.utils.evaluation import (
    TEST_SEED,
    check_opf_reference,
    evaluate_policy,
    load_agent,
    opf_cache_path,
    test_states,
)
from dhelm_opf.utils.opf_reference import load_or_compute


def main() -> None:

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", choices=("best", "last"), default="best")
    p.add_argument("--refine", choices=("none", "r1", "r2"), default="none")
    p.add_argument("--refine-K", type=int, default=20,
                   help="refinement step budget K")
    p.add_argument("--refine-lr", type=float, default=0.03)
    p.add_argument("--n-states", type=int, default=1000,
                   help="test states (a multiple of the window)")
    p.add_argument("--seed", type=int, default=TEST_SEED)
    p.add_argument("--fixed-slack-voltage", action="store_true",
                   help="pandapower default: hold the slack at its setpoint")
    p.add_argument("--opf-cache", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    agent, ckpt = load_agent(run_dir, args.checkpoint)
    window = ckpt["args"]["window"]

    states = test_states(agent, args.n_states, args.seed)

    ramp = agent.dhelm.ramp
    cache = args.opf_cache or opf_cache_path(
        args.n_states, args.seed, args.fixed_slack_voltage,
        ramp=ramp, episode_length=window)
    ref = load_or_compute(cache, states, agent.uncertainty,
                          agent.action_config, args.fixed_slack_voltage,
                          ramp=ramp, episode_length=window)
    check = check_opf_reference(agent, ref)

    refine = None
    if args.refine != "none":
        refine = RefineConfig(mode=args.refine, K=args.refine_K,
                              lr=args.refine_lr)

    result = evaluate_policy(agent, states, window, ref, refine)
    s, ps = result["summary"], result["per_state"]

    def pct(x):
        return f"{100 * x:+.2f}%"

    label = args.refine if refine is None else f"{args.refine} (K={args.refine_K})"

    print("=" * 68)
    print(f"{run_dir.name}  |  checkpoint {args.checkpoint} "
          f"(step {ckpt.get('step', '?')})  |  refinement {label}")
    print(f"{s['states']} test states, window {window}, ramp "
          f"{ckpt['args'].get('ramp_frac') or 'off'}, OPF slack voltage "
          f"{'fixed' if args.fixed_slack_voltage else 'optimized (matches policy)'}")
    print("=" * 68)
    print(f"OPF converged                 : {check['opf_converged']}/{s['states']}")
    print(f"OPF@critic cost mismatch      : {check['cost_mismatch_max']:.2e} $/h (max)")
    print(f"OPF@critic total violation    : {check['violation_max']:.2e} (max)")
    print()
    print(f"Feasible                      : {s['feasible']:.1%} (Σz <= 1e-6)")
    print(f"Mean violation Σz             : {s['violation_mean']:.2e}")
    print(f"Mean cost   policy / OPF      : {s['cost_mean']:.1f} / "
          f"{s['opf_cost_mean']:.1f} $/h")
    print(f"Optimality gap                : mean {pct(s['gap_mean'])}, "
          f"median {pct(s['gap_median'])}, 95th pct {pct(s['gap_p95'])}, "
          f"max {pct(s['gap_max'])}")
    print(f"Optimality gap, feasible only : mean {pct(s['gap_feasible_mean'])}")
    if refine is not None:
        print(f"Refined states                : {s['refined_states']} "
              f"(mean {s['refine_steps_mean']:.1f} steps)")
    print()
    if "ramp_bind_step_share" in s:
        print(f"Ramp limits bind (t > 0)      : policy {s['ramp_bind_step_share']:.1%} "
              f"of steps / {s['ramp_bind_unit_share']:.1%} of unit-steps"
              + (f";  OPF {s['opf_ramp_bind_step_share']:.1%} / "
                 f"{s['opf_ramp_bind_unit_share']:.1%}"
                 if "opf_ramp_bind_step_share" in s else ""))
        print()
    print(f"Time policy, all states       : {s['ms_policy_total']:.1f} ms")
    if refine is not None:
        print(f"Time refinement, all states   : {s['ms_refine_total']:.1f} ms")
    print(f"Time per state (total)        : {s['ms_total_per_state']:.2f} ms")

    names = agent.critic.constraint_names
    counts = (ps["z"] > 1e-6).sum(axis=0)
    if counts.any():
        print("\nViolations by constraint (states):")
        for n, c in zip(names, counts):
            if c:
                print(f"  {n:16s} {c}")

    tag = f"{args.checkpoint}_{args.refine}"
    if refine is not None:
        tag += f"_K{args.refine_K}"
    if args.fixed_slack_voltage:
        tag += "_fixed_slack"

    with open(run_dir / f"eval_{tag}.json", "w") as f:
        json.dump({**s, **{f"opf_check_{k}": v for k, v in check.items()},
                   "checkpoint_step": ckpt.get("step")}, f, indent=2)

    with open(run_dir / f"eval_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "cost", "opf_cost", "gap", "violation"])
        for k in range(s["states"]):
            w.writerow([k, ps["cost"][k], ps["opf_cost"][k], ps["gap"][k],
                        ps["violation"][k]])


if __name__ == "__main__":
    main()
