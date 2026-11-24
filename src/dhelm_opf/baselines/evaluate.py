"""
Deterministic evaluation of a trained baseline (paper Sec. 4.2, Table 2):
"disabling random noise in DDPG and using the mean value of action
distributions ... for MBPO, SAC, and PPO", on 100 profiles x 24 samples.

The same test states, ramp sequencing and cached OPF reference as for
the proposed method (utils/evaluation.py, utils/opf_reference.py), so
gaps and feasibility are directly comparable.  The 100 profiles run as
one batch; each step builds the environment's observation, asks the
policy for its action (carrying the LSTM state) and scores it with the
critic, with ramp windows around the previous step's dispatch.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tianshou.data import Batch

from dhelm_opf.baselines.agents import BaselineConfig, build
from dhelm_opf.baselines.env import FEASIBLE_TOL, OPFEnv

F64 = torch.float64


def load_baseline(run_dir: str | Path):
    ckpt = torch.load(Path(run_dir) / "policy.pt", weights_only=False,
                      map_location="cpu")
    a = ckpt["args"]
    env = OPFEnv(a["case"], seed=a["seed"], rho=a["rho"])
    policy, _, _ = build(a["algo"], a["case"], env.obs_dim,
                         env.action_space.shape[0], env.action_space, 10,
                         BaselineConfig(**ckpt["config"]), "cpu")
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, env, ckpt


def evaluate_baseline(policy, env: OPFEnv, states: np.ndarray, ref: dict,
                      window: int = 24) -> dict:
    critic, acfg = env.critic, env.acfg
    S, d = states.shape
    P = S // window
    s_all = states.reshape(P, window, d)

    low, high = env._low, env._high
    P_min, P_max = env._P_min, env._P_max
    G_s = acfg.G_s

    hidden, p_g_prev, p_g_s_prev = None, None, None
    cost = np.zeros((P, window))
    z = np.zeros((P, window, len(critic.constraint_names)))
    r_eq5 = np.zeros((P, window))
    t0 = time.perf_counter()

    for t in range(window):
        s_t = s_all[:, t]
        obs = 2.0 * (s_t - low) / (high - low) - 1.0
        if p_g_prev is None:
            prev = np.zeros((P, env.n_g + int(env.prev_slack)))
        else:
            prev = 2.0 * (p_g_prev - P_min) / (P_max - P_min) - 1.0
            if env.prev_slack:
                ps = np.clip(2.0 * (p_g_s_prev - G_s.P_min) / (G_s.P_max - G_s.P_min) - 1.0, -1, 1)
                prev = np.concatenate([prev, ps[:, None]], axis=1)
        obs = np.concatenate([obs, prev], axis=1).astype(np.float32)

        with torch.no_grad():
            out_pol = policy(Batch(obs=obs, info={}), state=hidden)
            hidden = out_pol.state
            act = out_pol.act
            act = act.cpu().numpy() if torch.is_tensor(act) else np.asarray(act)
            act = np.clip(act, -1.0, 1.0)

            kw = {}
            if p_g_prev is not None:
                kw = dict(p_g_prev=torch.as_tensor(p_g_prev, dtype=F64),
                          p_g_s_prev=torch.as_tensor(p_g_s_prev, dtype=F64))
            out = critic(torch.as_tensor(s_t, dtype=F64),
                         torch.as_tensor(act, dtype=F64), **kw)

        cost[:, t] = out.f_sum.numpy()
        z[:, t] = out.z.numpy()
        r_eq5[:, t] = out.r.numpy()
        p_g_prev, p_g_s_prev = out.p_g.numpy(), out.p_g_s.numpy()

    seconds = time.perf_counter() - t0
    cost_f, z_f = cost.reshape(S), z.reshape(S, -1)
    viol = z_f.sum(axis=1)
    feas = np.linalg.norm(z_f, axis=1) <= FEASIBLE_TOL

    ok = ref["converged"]
    opf = ref["cost"]
    gap = np.full(S, np.nan)
    gap[ok] = (cost_f[ok] - opf[ok]) / opf[ok]

    reward_pen = (r_eq5 - env.rho * z.sum(axis=2)).sum(axis=1)
    return {
        "states": S, "profiles": P,
        "feasible": float(feas.mean()),
        "feasible_strict": float((viol <= 1e-6).mean()),
        "violation_mean": float(viol.mean()),
        "gap_mean": float(np.nanmean(gap)),
        "gap_median": float(np.nanmedian(gap)),
        "cost_mean": float(cost_f.mean()),
        "opf_cost_mean": float(np.nanmean(opf[ok])),
        "reward_per_profile_mean": float(r_eq5.sum(axis=1).mean()),
        "reward_penalized_per_profile_mean": float(reward_pen.mean()),
        "reward_penalized_per_profile_std": float(reward_pen.std()),
        "cost_per_profile_mean": float(cost.sum(axis=1).mean()),
        "ms_per_state": 1e3 * seconds / S,
    }
