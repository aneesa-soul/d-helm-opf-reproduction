"""
Tests for the baseline environment, networks and evaluation.

1. Environment: gymnasium's env checker passes; observation / action
   sizes equal Table 1's policy input / output for 5-, 30- and 118-bus.
2. Reward = Eq. (5) - rho * sum(z) from the critic, with the Eq. (3)
   ramp window around the previous step's dispatch (|dp_g| <= R).
3. Networks: Table 1 actors (LSTM width per case), critic sizes per the
   paper, SAC's conditioned sigma network, PPO's state-independent sigma.
4. Recurrence: while collecting the actor steps one observation with its
   carried LSTM state; while training it receives (B, 24, obs) windows.
5. Deterministic evaluation (batched, baselines/evaluate.py) gives the
   same costs and violations as stepping the environment.

Run:
    python scripts/test_baselines.py
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv

from dhelm_opf.baselines.agents import ALGOS, build
from dhelm_opf.baselines.env import OPFEnv
from dhelm_opf.baselines.evaluate import evaluate_baseline
from dhelm_opf.baselines.nets import CRITIC_SIZES, QCritic, TableOneNet

F64 = torch.float64
TABLE1 = {"case5": (8 + 4, 4 + 4), "case30": (44 + 5, 6 + 5), "case118": (205 + 54, 54 + 53)}


def test_env_dims() -> None:
    for case, (n_in, n_out) in TABLE1.items():
        env = OPFEnv(case, seed=0)
        check_env(env, skip_render_check=True)
        assert env.observation_space.shape == (n_in,)
        assert env.action_space.shape == (n_out,)
    print("  1. env checker, Table 1 sizes       : OK (5/30/118-bus)")


def test_reward_and_ramp() -> None:
    env = OPFEnv("case5", seed=0)
    ramp, acfg = env.agent.dhelm.ramp, env.acfg
    R = np.array([ramp.R_u(u) for u in acfg.G_ns])
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    prev = None
    for t in range(24):
        s = torch.as_tensor(env._episode[t], dtype=F64)[None]
        a = rng.uniform(-1, 1, env.action_space.shape)
        kw = {} if env.p_g_prev is None else dict(
            p_g_prev=torch.as_tensor(env.p_g_prev, dtype=F64)[None],
            p_g_s_prev=torch.as_tensor([env.p_g_s_prev], dtype=F64))
        with torch.no_grad():
            out = env.critic(s, torch.as_tensor(a, dtype=F64)[None], **kw)
        _, r, term, trunc, info = env.step(a)
        expect = float(out.r[0]) - 10.0 * float(out.z[0].sum())
        assert abs(r - expect) < 1e-9, (r, expect)
        if prev is not None:
            assert np.all(np.abs(env.p_g_prev - prev) <= R + 1e-9)
        prev = env.p_g_prev.copy()
    assert trunc and not term
    print("  2. reward = Eq.5 - 10 Σz, ramping   : OK (24 steps, truncated at the end)")


def test_networks() -> None:
    for case in ("case5", "case30", "case118"):
        env = OPFEnv(case, seed=0)
        for algo in ALGOS:
            pol, _, bs = build(algo, case, env.obs_dim, env.action_space.shape[0],
                               env.action_space, 100)
            lstms = [m for m in pol.actor.modules() if isinstance(m, torch.nn.LSTM)]
            assert all(m.hidden_size == env.agent.case.hidden for m in lstms)
            if algo == "sac":
                assert len(lstms) == 2                     # mu + conditioned sigma
                crit = pol.critic1
            elif algo == "ppo":
                assert len(lstms) == 1 and hasattr(pol.actor, "log_sigma")
                crit = pol.critic
            else:
                crit = pol.critic
            sizes = [m.out_features for m in crit.net if isinstance(m, torch.nn.Linear)][:-1]
            assert tuple(sizes) == CRITIC_SIZES[case], (case, algo, sizes)
            assert bs == env.agent.case.batch_size
    print("  3. networks per Table 1 / Sec. 4.2  : OK (actor widths, critic sizes, sigma nets)")


def test_recurrence() -> None:
    seen = {}
    orig = TableOneNet.forward

    def spy(self, obs, state=None):
        key = "train" if (obs.dim() == 3 and obs.shape[1] > 1) else "collect"
        seen.setdefault(key, set()).add((tuple(obs.shape), state is not None))
        return orig(self, obs, state)

    TableOneNet.forward = spy
    try:
        for algo in ALGOS:
            seen.clear()
            env = OPFEnv("case5", seed=0)
            pol, buf, bs = build(algo, "case5", env.obs_dim, 8, env.action_space, 2000)
            Collector(pol, DummyVectorEnv([lambda: env]), buf,
                      exploration_noise=True).collect(n_episode=2)
            if algo == "ppo":
                pol.update(0, buf, batch_size=bs, repeat=1)
            else:
                pol.update(bs, buf)
            assert ((1, 12), True) in seen["collect"]          # carried LSTM state
            assert any(s[1:] == (24, 12) for s, _ in seen["train"])
    finally:
        TableOneNet.forward = orig
    print("  4. recurrence                       : OK (state carried while collecting, "
          "(B, 24, obs) windows in training)")


def test_eval_matches_env() -> None:
    env = OPFEnv("case30", seed=0)
    pol, _, _ = build("sac", "case30", env.obs_dim, env.action_space.shape[0],
                      env.action_space, 100)
    pol.eval()
    profiles = np.stack([env.sampler.episode(24) for _ in range(3)])
    ref = {"converged": np.ones(72, bool), "cost": np.ones(72)}

    costs = []
    for k in range(3):
        env.set_profile(profiles[k])
        obs, _ = env.reset()
        state = None
        for t in range(24):
            out = pol(Batch(obs=obs[None], info={}), state=state)
            state = out.state
            obs, _, _, _, info = env.step(np.clip(out.act.detach().numpy()[0], -1, 1))
            costs.append(info["f_sum"])
    res = evaluate_baseline(pol, env, profiles.reshape(72, -1), ref)
    assert abs(res["cost_mean"] - np.mean(costs)) < 1e-6 * np.mean(costs)
    print("  5. batched evaluation = env rollout  : OK (3 profiles, deterministic SAC)")


def main() -> None:
    torch.manual_seed(0)
    print("Baseline tests")
    test_env_dims()
    test_reward_and_ramp()
    test_networks()
    test_recurrence()
    test_eval_matches_env()
    print("All baseline tests passed.")


if __name__ == "__main__":
    main()
