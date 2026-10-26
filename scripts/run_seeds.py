"""
Train one configuration over several seeds and evaluate every seed
against the same cached OPF reference.

For each seed:
    train          scripts/train_dhelm_case5.py (skipped if the run exists)
    evaluate       last checkpoint, best checkpoint, and the best
                   checkpoint refined with R1 / R2 at several budgets K

Outputs (results/experiments/<name>/):
    per_seed.csv   one row per (seed, variant)
    summary.csv    mean and std over seeds per variant
    summary.png    gap and feasibility per variant, one dot per seed

Examples:
    python scripts/run_seeds.py --name paper_2500 --episodes 100
    python scripts/run_seeds.py --name paper_5000 --episodes 200
    python scripts/run_seeds.py --name cosine_2500 --episodes 100 --lr-schedule cosine
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
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

TRAIN_SCRIPT = Path(__file__).with_name("train_dhelm_case5.py")


def parse_variants(text: str):
    """'last,best,best+r1:20,best+r2:50' -> [(ckpt, mode, K), ...]"""

    out = []
    for item in text.split(","):
        item = item.strip()
        if "+" in item:
            ckpt, ref = item.split("+")
            mode, k = ref.split(":")
            out.append((ckpt, mode, int(k)))
        else:
            out.append((item, "none", 0))
    return out


def variant_label(ckpt, mode, k):
    return ckpt if mode == "none" else f"{ckpt}+{mode.upper()} K={k}"


def main() -> None:

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"),
                   default="constant")
    p.add_argument("--variants", type=str,
                   default="last,best,best+r1:20,best+r2:5,best+r2:20,"
                           "best+r2:50,best+r2:200")
    p.add_argument("--refine-lr", type=float, default=0.03)
    p.add_argument("--n-states", type=int, default=100)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--runs-dir", type=str, default="results/runs")
    p.add_argument("--out", type=str, default="results/experiments")
    args = p.parse_args()

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variants)

    # ---- train ------------------------------------------------------
    run_dirs = {}
    for seed in args.seeds:
        run_name = f"{args.name}_seed{seed}"
        run_dir = Path(args.runs_dir) / run_name
        run_dirs[seed] = run_dir

        if (run_dir / "summary.json").exists() and not args.retrain:
            print(f"[train] {run_name}: exists, skipped")
            continue

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--episodes", str(args.episodes),
            "--window", str(args.window),
            "--lr-schedule", args.lr_schedule,
            "--seed", str(seed),
            "--name", run_name,
            "--out", args.runs_dir,
            "--quiet",
        ]
        print(f"[train] {run_name} ...", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:])
            raise RuntimeError(f"training failed for seed {seed}")
        print("        " + res.stdout.strip().splitlines()[-1])

    # ---- shared test set and OPF reference --------------------------
    agent, _ = load_agent(run_dirs[args.seeds[0]], "last")
    states = test_states(agent, args.n_states, TEST_SEED)
    ref = load_or_compute(opf_cache_path(args.n_states, TEST_SEED, False),
                          states, agent.uncertainty, agent.action_config)
    check = check_opf_reference(agent, ref)
    print(f"[opf] {check['opf_converged']}/{args.n_states} converged; "
          f"OPF@critic cost mismatch {check['cost_mismatch_max']:.1e} $/h, "
          f"violation {check['violation_max']:.1e}")

    # ---- evaluate ---------------------------------------------------
    rows = []
    loaded = {}
    for seed in args.seeds:
        for ckpt, mode, k in variants:
            key = (seed, ckpt)
            if key not in loaded:
                loaded[key] = load_agent(run_dirs[seed], ckpt)
            agent, meta = loaded[key]

            refine = None if mode == "none" else RefineConfig(
                mode=mode, steps=k, lr=args.refine_lr)

            res = evaluate_policy(agent, states, meta["args"]["window"],
                                  ref, refine)
            s = res["summary"]
            rows.append({
                "seed": seed,
                "variant": variant_label(ckpt, mode, k),
                "checkpoint_step": meta.get("step"),
                **s,
            })
            print(f"[eval] seed {seed}  {variant_label(ckpt, mode, k):18s} "
                  f"feasible {s['feasible']:6.1%}  gap {100 * s['gap_mean']:+6.2f}%  "
                  f"{s['ms_total_per_state']:6.2f} ms/state")

    with open(out_dir / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate --------------------------------------------------
    labels = [variant_label(*v) for v in variants]
    summary = []
    for lab in labels:
        sel = [r for r in rows if r["variant"] == lab]

        def ms(key):
            x = np.array([r[key] for r in sel], dtype=float)
            return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

        g, gs = ms("gap_mean")
        fe, fs = ms("feasible")
        t, _ = ms("ms_total_per_state")
        summary.append({"variant": lab, "seeds": len(sel),
                        "gap_mean": g, "gap_std": gs,
                        "feasible_mean": fe, "feasible_std": fs,
                        "ms_per_state": t})

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n{args.name}: {len(args.seeds)} seeds, {args.episodes} episodes "
          f"({args.episodes * 25} steps), lr schedule {args.lr_schedule}")
    print(f"{'variant':20s} {'gap (mean ± std)':>20s} {'feasible':>16s} {'ms/state':>9s}")
    for s in summary:
        print(f"{s['variant']:20s} {100 * s['gap_mean']:+8.2f}% ± {100 * s['gap_std']:5.2f}"
              f"   {100 * s['feasible_mean']:6.1f}% ± {100 * s['feasible_std']:4.1f}"
              f" {s['ms_per_state']:9.2f}")

    plot(rows, labels, out_dir / "summary.png", args.name)
    print(f"\nSaved to {out_dir}/")


# ----------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"


def plot(rows, labels, path, title):
    """Two panels (gap, feasibility); one dot per seed, a tick for the mean."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    x = np.arange(len(labels))

    for ax, key, scale, ttl in (
        (axes[0], "gap_mean", 100, "Optimality gap vs OPF (%)"),
        (axes[1], "feasible", 100, "Feasible test states (%)"),
    ):
        for i, lab in enumerate(labels):
            vals = np.array([r[key] for r in rows if r["variant"] == lab]) * scale
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), vals, s=36, color=SERIES_1,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
            m = vals.mean()
            ax.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=2, zorder=4)
            ax.annotate(f"{m:.1f}", (i + 0.27, m), color=INK_2, fontsize=8,
                        va="center")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8,
                           color=INK_2)
        ax.set_title(ttl, color=INK, fontsize=10, loc="left")
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)

    axes[0].axhline(0, color=INK_2, lw=1)
    fig.suptitle(f"{title} — dots: seeds, bar: mean", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
