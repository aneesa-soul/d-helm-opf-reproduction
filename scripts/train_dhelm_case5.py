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
VALIDATION set of states, so the curves are not affected by minibatch
noise.  The same evaluations select the best checkpoint: the lowest
validation cost among evaluations with >= `--best-feasible` feasible
states (fallback: lowest validation loss).

Outputs (in results/runs/<name>/):
    metrics.csv      one row per evaluation
    policy.pt        last policy, multipliers and the run configuration
    policy_best.pt   best checkpoint (see above)
    summary.json     last vs best step, cost, feasibility
    training.png     learning curves (best step marked)

Examples:
    python scripts/train_dhelm_case5.py --window 1 --name case5_L1
    python scripts/train_dhelm_case5.py --episodes 100 --lr-schedule cosine
"""

from __future__ import annotations

import argparse
import copy
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-length", type=int, default=25)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta-lam", type=float, default=1.0,
                   help="eta_lam, Eq. (20)")
    p.add_argument("--lam-max", type=float, default=1000.0,
                   help="lam_max, Eq. (20)")
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"),
                   default="constant",
                   help="constant = paper (Table 1); cosine decays to "
                        "lr * lr-min-factor over the run")
    p.add_argument("--lr-min-factor", type=float, default=0.1)
    p.add_argument("--best-feasible", type=float, default=0.99,
                   help="validation feasible share required for the "
                        "best checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--out", type=str, default="results/runs")
    p.add_argument("--quiet", action="store_true",
                   help="print only the final summary line")
    return p.parse_args(argv)


def main(argv=None) -> Path:

    args = parse_args(argv)
    name = args.name or f"case5_L{args.window}_seed{args.seed}"
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cpo_config = CPOConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        window=args.window,
        eta_lam=args.eta_lam,
        lam_max=args.lam_max,
    )

    agent = make_case5_agent(seed=args.seed, cpo_config=cpo_config)
    cpo, sampler = agent.cpo, agent.sampler

    total_steps = args.episodes * args.episode_length
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            cpo.optimizer, T_max=total_steps,
            eta_min=args.lr * args.lr_min_factor,
        )

    # Validation windows: never trained on, and distinct from the test
    # states used by evaluate_vs_opf.py (seed 20000).  They draw the
    # learning curves and select the best checkpoint.
    eval_sampler = StateSampler(
        agent.dhelm.n_d, agent.dhelm.n_w, agent.uncertainty,
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

    # Best checkpoint: lowest validation cost among evaluations with at
    # least `best_feasible` feasible validation states.  Fallback (if no
    # evaluation qualifies): lowest validation loss.
    best = {"feasible": None, "loss": None}

    def snapshot(m):
        return {
            "policy": copy.deepcopy(agent.policy.state_dict()),
            "lam": cpo.lam.clone(),
            "step": m["step"],
            "f_sum": m["f_sum"],
            "feasible": m["feasible"],
            "loss": m["loss"],
        }

    def log(step, episode):
        m = cpo.evaluate_metrics(eval_states)
        m.update(step=step, episode=episode,
                 seconds=time.perf_counter() - start,
                 lr=cpo.optimizer.param_groups[0]["lr"])
        rows.append(m)

        if step > 0:
            if m["feasible"] >= args.best_feasible and (
                best["feasible"] is None or m["f_sum"] < best["feasible"]["f_sum"]
            ):
                best["feasible"] = snapshot(m)
            if best["loss"] is None or m["loss"] < best["loss"]["loss"]:
                best["loss"] = snapshot(m)
        return m

    def report(m, episode=None):
        if args.quiet:
            return
        tail = (f"  max lam {cpo.lam.max().item():7.2f}  (episode {episode})"
                if episode else "")
        print(f"step {m['step']:4d}  r {m['r']:.4f}  "
              f"f_sum {m['f_sum']:8.1f}  z_sum {m['z_sum']:.4f}  "
              f"feasible {m['feasible']:.2f}{tail}")

    m = log(0, 0)
    report(m)

    for episode in range(1, args.episodes + 1):

        buffer.add_episode(sampler.episode(args.episode_length))

        for _ in range(args.episode_length):
            windows = buffer.sample_windows(args.batch_size, args.window, rng)
            cpo.train_step(windows)
            step += 1
            if scheduler is not None:
                scheduler.step()

            if step % args.eval_every == 0:
                m = log(step, episode)

        report(m, episode)

    # ---- save -------------------------------------------------------
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    common = {
        "cpo_config": asdict(cpo_config),
        "args": vars(args),
        "constraint_names": agent.critic.constraint_names,
    }

    torch.save(
        {"policy": agent.policy.state_dict(), "lam": cpo.lam,
         "step": step, **common},
        out_dir / "policy.pt",
    )

    if best["feasible"] is not None:
        chosen, criterion = best["feasible"], (
            f"lowest validation cost with feasible >= {args.best_feasible}"
        )
    else:
        chosen, criterion = best["loss"], "lowest validation loss (fallback)"

    torch.save(
        {"policy": chosen["policy"], "lam": chosen["lam"],
         "step": chosen["step"], "criterion": criterion, **common},
        out_dir / "policy_best.pt",
    )

    summary = {
        "last": {k: rows[-1][k] for k in ("step", "f_sum", "feasible", "loss")},
        "best": {k: chosen[k] for k in ("step", "f_sum", "feasible", "loss")},
        "best_criterion": criterion,
        "seconds": time.perf_counter() - start,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "cpo_config": asdict(cpo_config)},
                  f, indent=2)

    plot(rows, agent.critic.constraint_names, out_dir / "training.png", name,
         best_step=chosen["step"])

    print(f"{name}: last step {step} f_sum {rows[-1]['f_sum']:.1f} "
          f"feasible {rows[-1]['feasible']:.2f} | best step {chosen['step']} "
          f"f_sum {chosen['f_sum']:.1f} feasible {chosen['feasible']:.2f} "
          f"| saved to {out_dir}/ ({time.perf_counter() - start:.1f} s)")

    return out_dir


# ----------------------------------------------------------------------
# Learning-curve figure: small multiples, one measure per panel
# ----------------------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3


def plot(rows, names, path, title, best_step=None):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([r["step"] for r in rows])

    def col(key):
        return np.array([r[key] for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), facecolor=SURFACE)
    panels = [
        ("r", "Reward r (Eq. 5)", None),
        ("f_sum", "Operating cost  Σ f_g(p_g)  ($/h)", None),
        ("z_sum", "Total violation  Σz (Eq. 6)", None),
        ("feasible", "Feasible share of validation states", (0, 1.02)),
        ("N", "D-HELM terms N", None),
    ]

    for ax, (key, label, ylim) in zip(axes.flat, panels):
        ax.plot(steps, col(key), color=SERIES[0], lw=2)
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        if best_step is not None and key in ("f_sum", "feasible"):
            ax.axvline(best_step, color=INK_2, lw=1, ls="--")
            ax.annotate("best", (best_step, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -10), textcoords="offset points",
                        color=INK_2, fontsize=8)

    # Multipliers: the three that grew most, plus a legend.
    ax = axes.flat[5]
    lam = {n: col(f"lam/{n}") for n in names}
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
