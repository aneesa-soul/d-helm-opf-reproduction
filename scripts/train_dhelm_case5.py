"""
Train the D-HELM policy on the IEEE 5-bus case.

Loop (paper Fig. 2b; Table 1 for the 5-bus hyperparameters):

    for each episode:
        draw an episode of states and add it to the states-only buffer
        for each step of the episode:
            sample a minibatch of state windows from the buffer
            policy -> actions -> D-HELM critic -> r, z
            policy update on  -mean(r) + sum_n lambda_n mean(z_n)   (19)
            multiplier update lambda_n += eta * mean(z_n)          (20)

Every `--eval-every` steps the policy is evaluated on a fixed held-out
set of states, so the curves are not affected by minibatch noise.

Outputs (in results/runs/<name>/):
    metrics.csv   one row per evaluation
    policy.pt     policy weights, multipliers and the run configuration
    training.png  learning curves

Examples:
    python scripts/train_dhelm_case5.py --window 1 --name case5_L1
    python scripts/train_dhelm_case5.py --window 5 --name case5_L5
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dhelm_opf.algorithms.case5_setup import make_case5_agent
from dhelm_opf.algorithms.dhelm_cpo import CPOConfig
from dhelm_opf.environment.state_sampler import StateBuffer, StateSampler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=25)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta-lambda", type=float, default=1.0)
    p.add_argument("--lambda-max", type=float, default=1000.0)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--out", type=str, default="results/runs")
    return p.parse_args()


def main() -> None:

    args = parse_args()
    name = args.name or f"case5_L{args.window}_seed{args.seed}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cpo_config = CPOConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        window=args.window,
        eta_lambda=args.eta_lambda,
        lambda_max=args.lambda_max,
    )

    agent = make_case5_agent(seed=args.seed, cpo_config=cpo_config)
    cpo, sampler = agent.cpo, agent.sampler

    # Held-out evaluation windows (separate seed, never trained on).
    eval_sampler = StateSampler(
        agent.dhelm.n_load, agent.dhelm.n_wind, agent.uncertainty,
        seed=10_000 + args.seed,
    )
    eval_states = eval_sampler.sample(args.eval_size * args.window).reshape(
        args.eval_size, args.window, -1
    )

    buffer = StateBuffer()
    rng = np.random.default_rng(args.seed)

    rows = []
    step = 0
    start = time.perf_counter()

    def log(step, episode):
        m = cpo.evaluate_metrics(eval_states)
        m.update(step=step, episode=episode,
                 seconds=time.perf_counter() - start)
        rows.append(m)
        return m

    m = log(0, 0)
    print(f"step {0:4d}  reward {m['reward']:.4f}  cost {m['cost']:8.1f}  "
          f"violation {m['violation']:.4f}  feasible {m['feasible']:.2f}")

    for episode in range(1, args.episodes + 1):

        buffer.add_episode(sampler.episode(args.episode_length))

        for _ in range(args.episode_length):
            windows = buffer.sample_windows(args.batch_size, args.window, rng)
            cpo.train_step(windows)
            step += 1

            if step % args.eval_every == 0:
                m = log(step, episode)

        print(f"step {step:4d}  reward {m['reward']:.4f}  cost {m['cost']:8.1f}  "
              f"violation {m['violation']:.4f}  feasible {m['feasible']:.2f}  "
              f"max lambda {cpo.lambdas.max().item():7.2f}  "
              f"(episode {episode})")

    # ---- save -------------------------------------------------------
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "lambdas": cpo.lambdas,
            "cpo_config": asdict(cpo_config),
            "args": vars(args),
            "constraint_names": agent.critic.constraint_names,
        },
        out_dir / "policy.pt",
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "cpo_config": asdict(cpo_config)},
                  f, indent=2)

    plot(rows, agent.critic.constraint_names, out_dir / "training.png", name)

    print(f"\nSaved to {out_dir}/  ({time.perf_counter() - start:.1f} s)")


# ----------------------------------------------------------------------
# Learning-curve figure: small multiples, one measure per panel
# ----------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3


def plot(rows, names, path, title):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([r["step"] for r in rows])

    def col(key):
        return np.array([r[key] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    panels = [
        ("reward", "Reward (Eq. 5)", None),
        ("cost", "Operating cost ($/h)", None),
        ("violation", "Total violation  Σz (Eq. 6)", None),
        ("feasible", "Feasible share of eval states", (0, 1.02)),
        ("helm_terms", "D-HELM terms N", None),
    ]

    for ax, (key, label, ylim) in zip(axes.flat, panels):
        ax.plot(steps, col(key), color=SERIES[0], lw=2)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        if ylim:
            ax.set_ylim(*ylim)

    # Multipliers: the three that grew most, plus a legend.
    ax = axes.flat[5]
    lam = {n: col(f"lambda/{n}") for n in names}
    top = sorted(names, key=lambda n: lam[n][-1], reverse=True)[:3]
    for color, n in zip(SERIES, top):
        ax.plot(steps, lam[n], color=color, lw=2, label=n)
        ax.annotate(f"{lam[n][-1]:.1f}", (steps[-1], lam[n][-1]),
                    xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center")
    ax.set_title("λ, three largest (Eq. 20)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlabel("training step", color=INK_2, fontsize=8)

    fig.suptitle(f"D-HELM policy training — {title}", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
