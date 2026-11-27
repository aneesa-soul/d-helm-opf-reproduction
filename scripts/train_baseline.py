"""
Train one baseline agent (DDPG, SAC or PPO; Tianshou 0.5.1) on one test
system, for the paper's Fig. 7 learning curves and Table 2 comparison.

Episodes are 24-step profiles with ramping (as for the proposed method);
the reward is Eq. (5) minus a fixed penalty rho * sum(z) (rho = 10, see
baselines/env.py).  Per training episode the script logs the episode
return, mean cost, mean violation and the share of feasible steps.

Examples:
    python scripts/train_baseline.py --algo sac --case case5 --episodes 1000 --seed 0
    python scripts/train_baseline.py --algo ppo --case case30 --episodes 2000 --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv

from dhelm_opf.baselines.agents import ALGOS, BaselineConfig, build
from dhelm_opf.baselines.env import OPFEnv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--case", default="case5")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=10.0, help="fixed violation penalty")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/baselines")
    p.add_argument("--name", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def episode_stats(buffer, idx):
    """Return, cost, violation and feasible share of the steps idx.

    Read from buffer.rew / buffer.info directly: buffer[idx] stacks
    `info` into stack_num-long windows padded with the episode's first
    step (like the observations), which biased these averages toward
    step 0 (and made the r_eq5 sum ~24x too large) before this fix.
    """
    info = buffer.info
    return {
        "return": float(np.sum(buffer.rew[idx])),
        "f_sum": float(np.mean(info.f_sum[idx])),
        "z_sum": float(np.mean(info.z_sum[idx])),
        "feasible": float(np.mean(info.feasible[idx])),
        "r_eq5": float(np.sum(info.r_eq5[idx])),
    }


def last_episode_stats(buffer, n_steps):
    idx = (buffer.last_index[0] - np.arange(n_steps)[::-1]) % buffer.maxsize
    return episode_stats(buffer, idx)


def check_finite(policy, name) -> None:
    """Stop with a clear message instead of training on NaN weights."""
    for k, v in policy.state_dict().items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise FloatingPointError(f"{name}: non-finite weights in {k}")


def main(argv=None) -> None:
    args = parse_args(argv)
    name = args.name or f"{args.case}_{args.algo}_seed{args.seed}"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = BaselineConfig()
    L = 24

    env = OPFEnv(args.case, seed=args.seed, rho=args.rho)
    envs = DummyVectorEnv([lambda: env])
    envs.seed(args.seed)

    total = (args.episodes + cfg.warmup_episodes) * L
    policy, buffer, batch = build(args.algo, args.case, env.obs_dim,
                                  env.action_space.shape[0], env.action_space,
                                  total, cfg, args.device)
    collector = Collector(policy, envs, buffer, exploration_noise=True)

    rows, t0 = [], time.time()

    if args.algo in ("ddpg", "sac"):
        collector.collect(n_episode=cfg.warmup_episodes, random=True)
        for ep in range(1, args.episodes + 1):
            policy.train()
            collector.collect(n_episode=1)
            for _ in range(L):                         # one update per step
                policy.update(batch, buffer)
            if ep % 10 == 0:
                check_finite(policy, name)
            rows.append({"episode": ep, **last_episode_stats(buffer, L)})
            if not args.quiet and ep % 50 == 0:
                print(f"  ep {ep:5d}  return {rows[-1]['return']:9.2f}  "
                      f"cost {rows[-1]['f_sum']:10.1f}  feasible {rows[-1]['feasible']:.2f}")
    else:                                              # PPO, on-policy
        ep = 0
        while ep < args.episodes:
            policy.train()
            n = min(cfg.ppo_episodes_per_update, args.episodes - ep)
            collector.collect(n_episode=n)
            for k in range(n):
                idx = np.arange(k * L, (k + 1) * L)
                ep += 1
                rows.append({"episode": ep, **episode_stats(buffer, idx)})
            policy.update(0, buffer, batch_size=batch, repeat=cfg.ppo_repeat)
            buffer.reset()
            check_finite(policy, name)
            if not args.quiet and ep % 48 < n:
                print(f"  ep {ep:5d}  return {rows[-1]['return']:9.2f}  "
                      f"cost {rows[-1]['f_sum']:10.1f}  feasible {rows[-1]['feasible']:.2f}")

    seconds = time.time() - t0
    torch.save({"policy": policy.state_dict(), "args": vars(args),
                "config": cfg.__dict__}, out / "policy.pt")
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    last = rows[-min(50, len(rows)):]
    summary = {"name": name, "algo": args.algo, "case": args.case,
               "episodes": args.episodes, "seed": args.seed, "seconds": seconds,
               "last50_return": float(np.mean([r["return"] for r in last])),
               "last50_feasible": float(np.mean([r["feasible"] for r in last])),
               "last50_cost": float(np.mean([r["f_sum"] for r in last]))}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{name}: last-50 return {summary['last50_return']:.2f}, cost "
          f"{summary['last50_cost']:.1f}, feasible {summary['last50_feasible']:.2f} "
          f"| saved to {out}/ ({seconds:.0f} s)")


if __name__ == "__main__":
    main()
