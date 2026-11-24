"""
Train and evaluate the baseline agents (DDPG, SAC, PPO) on one test
system over several seeds: the paper's Fig. 7 learning curves and
Table 2 comparison.

    train      scripts/train_baseline.py per (algo, seed), skipped when
               the run already has a summary.json
    evaluate   deterministic, on the same 2,400 test states (100 profiles
               x 24) and the same cached OPF reference as the proposed
               method (baselines/evaluate.py)

Outputs in results/experiments/<name>/: per_seed.csv, summary.csv and
learning_curves.png (training-episode return, mean over seeds).

Defaults = the paper's training length per system (Sec. 4.2):
1000 / 2000 / 3000 episodes for case5 / case30 / case118.

Examples:
    python scripts/run_baselines.py --case case5 --name bl_case5
    python scripts/run_baselines.py --case case30 --name bl_case30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dhelm_opf.algorithms.setup import make_agent
from dhelm_opf.baselines.agents import ALGOS
from dhelm_opf.baselines.evaluate import evaluate_baseline, load_baseline
from dhelm_opf.environment.cases import get_case
from dhelm_opf.models.action_config import RampConfig
from dhelm_opf.utils.evaluation import TEST_SEED, opf_cache_path, test_states
from dhelm_opf.utils.opf_reference import load_or_compute


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", default="case5")
    p.add_argument("--name", required=True)
    p.add_argument("--algos", nargs="+", default=list(ALGOS), choices=ALGOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=None,
                   help="default: the paper's Table 2 training length")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-states", type=int, default=2400)
    p.add_argument("--runs", default="results/baselines")
    p.add_argument("--out", default="results/experiments")
    args = p.parse_args()

    spec = get_case(args.case)
    episodes = args.episodes or spec.table2_episodes

    # ---- train ---------------------------------------------------------
    run_dirs = {}
    for algo in args.algos:
        for seed in args.seeds:
            name = f"{args.name}_{algo}_seed{seed}"
            run_dir = Path(args.runs) / name
            run_dirs[algo, seed] = run_dir
            if (run_dir / "summary.json").exists():
                print(f"[train] {name}: exists, skipped")
                continue
            print(f"[train] {name} ...", flush=True)
            cmd = [sys.executable, "scripts/train_baseline.py", "--algo", algo,
                   "--case", args.case, "--episodes", str(episodes),
                   "--seed", str(seed), "--device", args.device,
                   "--out", args.runs, "--name", name, "--quiet"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-2000:], res.stderr[-4000:])
                raise SystemExit(f"training {name} failed")
            print("       ", res.stdout.strip().splitlines()[-1])

    # ---- OPF reference (same cache as the proposed method) --------------
    ramp = RampConfig()
    ref_agent = make_agent(args.case, seed=0, ramp=ramp)
    states = test_states(ref_agent, args.n_states, TEST_SEED)
    ref = load_or_compute(
        opf_cache_path(args.n_states, TEST_SEED, False, ramp=ramp,
                       episode_length=24, case=args.case),
        states, ref_agent.uncertainty, ref_agent.action_config, False,
        ramp=ramp, episode_length=24, case=args.case)
    print(f"[opf] {int(ref['converged'].sum())}/{len(states)} converged; "
          f"mean cost {np.nanmean(ref['cost']):.1f} $/h")

    # ---- evaluate ------------------------------------------------------
    rows = []
    for (algo, seed), run_dir in run_dirs.items():
        policy, env, _ = load_baseline(run_dir)
        s = evaluate_baseline(policy, env, states, ref)
        rows.append({"algo": algo, "seed": seed, **s})
        print(f"[eval] {algo:4s} seed {seed}  feasible {100 * s['feasible']:5.1f}%  "
              f"gap {100 * s['gap_mean']:+8.2f}%  reward/profile {s['reward_penalized_per_profile_mean']:9.1f}")

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for algo in args.algos:
        sel = [r for r in rows if r["algo"] == algo]

        def ms(key):
            v = np.array([r[key] for r in sel], dtype=float)
            return float(v.mean()), float(v.std())

        g, gs = ms("gap_mean")
        fe, fs = ms("feasible")
        rp, rps = ms("reward_penalized_per_profile_mean")
        r5, _ = ms("reward_per_profile_mean")
        c, _ = ms("cost_per_profile_mean")
        summary.append({"algo": algo, "seeds": len(sel), "gap_mean": g, "gap_std": gs,
                        "feasible_mean": fe, "feasible_std": fs,
                        "reward_penalized_per_profile": rp, "reward_penalized_std": rps,
                        "reward_eq5_per_profile": r5, "cost_per_profile": c})
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    print(f"\n{args.name} ({args.case}): {len(args.seeds)} seeds, {episodes} episodes, "
          f"reward = Eq. 5 - 10 sum(z)")
    print(f"{'algo':6s} {'gap (mean ± std)':>20s} {'feasible':>16s} "
          f"{'reward/profile':>18s} {'Eq.5 reward':>12s} {'cost/profile':>13s}")
    for s in summary:
        print(f"{s['algo']:6s} {100 * s['gap_mean']:+9.2f}% ± {100 * s['gap_std']:5.2f}"
              f"   {100 * s['feasible_mean']:6.1f}% ± {100 * s['feasible_std']:4.1f}"
              f"   {s['reward_penalized_per_profile']:9.1f} ± {s['reward_penalized_std']:5.1f}"
              f" {s['reward_eq5_per_profile']:12.2f} {s['cost_per_profile']:13.4g}")

    _plot_curves(run_dirs, args, out_dir)
    print(f"\nSaved to {out_dir}/")


def _plot_curves(run_dirs, args, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for algo in args.algos:
        curves = {}
        for seed in args.seeds:
            path = run_dirs[algo, seed] / "metrics.csv"
            if not path.exists():
                continue
            with open(path) as f:
                rd = list(csv.DictReader(f))
            for key in ("return", "f_sum", "feasible"):
                curves.setdefault(key, []).append([float(r[key]) for r in rd])
        for k, key in enumerate(("return", "f_sum", "feasible")):
            if key not in curves:
                continue
            n = min(len(c) for c in curves[key])
            m = np.mean([c[:n] for c in curves[key]], axis=0)
            w = max(1, n // 50)
            m = np.convolve(m, np.ones(w) / w, mode="valid")
            ax[k].plot(np.arange(len(m)) + w, m, label=algo.upper())
    for a, t in zip(ax, ("episode return (Eq. 5 - 10 Σz)", "mean cost ($/h)", "feasible share")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("episode")
    ax[0].legend()
    fig.suptitle(f"{args.case}: baselines, mean over seeds {args.seeds}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)


if __name__ == "__main__":
    main()
