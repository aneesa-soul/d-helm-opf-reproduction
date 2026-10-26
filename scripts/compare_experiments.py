"""
Side-by-side table of several run_seeds.py experiments.

Reads results/experiments/<name>/summary.csv for each name and prints
gap (mean ± std over seeds), feasibility and time per state for every
variant.

Example:
    python scripts/compare_experiments.py paper_2500 paper_5000 cosine_2500
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:

    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--root", default="results/experiments")
    args = p.parse_args()

    tables = {}
    for name in args.names:
        path = Path(args.root) / name / "summary.csv"
        with open(path) as f:
            tables[name] = {r["variant"]: r for r in csv.DictReader(f)}

    variants = list(next(iter(tables.values())).keys())

    for metric, label, fmt in (
        ("gap", "Optimality gap vs OPF (mean ± std over seeds)",
         lambda r: f"{100 * float(r['gap_mean']):+6.2f}% ± {100 * float(r['gap_std']):4.2f}"),
        ("feasible", "Feasible test states (mean over seeds)",
         lambda r: f"{100 * float(r['feasible_mean']):6.1f}%"),
        ("time", "Time per state, policy + refinement (ms)",
         lambda r: f"{float(r['ms_per_state']):7.2f}"),
    ):
        print(f"\n{label}")
        print(f"{'variant':20s}" + "".join(f"{n:>20s}" for n in args.names))
        for v in variants:
            cells = "".join(
                f"{fmt(tables[n][v]) if v in tables[n] else '-':>20s}"
                for n in args.names
            )
            print(f"{v:20s}{cells}")


if __name__ == "__main__":
    main()
