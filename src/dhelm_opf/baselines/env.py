"""
Gymnasium OPF environment for the baseline DRL agents (paper Sec. 4.1).

The paper: "The power system environment is standardized as OpenAI's
gymnasium library, where the power flow calculations are based on the
HELM procedure, defined in Algorithm 1."  Here the power flow, costs and
violations come from the same validated D-HELM critic the proposed
method uses (evaluated without gradients), so every agent is scored by
identical physics.

One episode = one 24-sample profile (Sec. 4.2), with the Eq. (3) ramp
limits between consecutive steps, exactly as for the proposed method.

Observation (Table 1 "policy input"): the uncertainty state normalized to
[-1, 1], then the previous p_g of the non-slack units normalized by their
static limits (and the previous slack output, clipped to [-1, 1], for
118/300-bus); zeros at t = 0.

Action: a_hat in [-1, 1]^n_a (Eq. 18 scaling inside the D-HELM).

Reward (our choice; the paper gives no penalty weight for the
baselines): r_t = w_0 - w_r f_sum - rho * sum_n z_n with a FIXED
rho = 10 (= lambda_0 of the proposed method).  The paper notes that
adaptive (CPO-style) penalties make the replay buffer unstable for
off-policy agents, so a fixed penalty is used.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from dhelm_opf.algorithms.setup import make_agent
from dhelm_opf.models.action_config import RampConfig

F64 = torch.float64
FEASIBLE_TOL = 1e-4          # ||z||_2 <= 1e-4, Sec. 4.4


class OPFEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        case: str = "case5",
        seed: int = 0,
        rho: float = 10.0,
        episode_length: int = 24,
        ramp: RampConfig | None = RampConfig(),
    ) -> None:
        super().__init__()

        self.agent = make_agent(case, seed=seed, ramp=ramp)
        self.critic = self.agent.critic
        self.acfg = self.agent.action_config
        self.sampler = self.agent.sampler
        self.prev_slack = self.agent.case.prev_slack

        self.rho = float(rho)
        self.L = int(episode_length)
        self.n_g = self.acfg.n_g
        self.state_dim = self.agent.dhelm.state_dim
        self.obs_dim = self.state_dim + self.n_g + int(self.prev_slack)

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.acfg.action_dim,), np.float32)

        self._low, self._high = self.sampler.low, self.sampler.high
        self._P_min = np.asarray(self.acfg.P_min, dtype=float)
        self._P_max = np.asarray(self.acfg.P_max, dtype=float)
        self._episode = None
        self._fixed = None
        self.t = 0
        self.p_g_prev = self.p_g_s_prev = None

    # ------------------------------------------------------------------
    def set_profile(self, states: np.ndarray | None) -> None:
        """Use the given (L, state_dim) profile at the next reset (test)."""

        self._fixed = None if states is None else np.asarray(states, dtype=float)

    def _obs(self) -> np.ndarray:
        s = self._episode[self.t]
        s_n = 2.0 * (s - self._low) / (self._high - self._low) - 1.0
        if self.p_g_prev is None:
            prev = np.zeros(self.n_g + int(self.prev_slack))
        else:
            prev = 2.0 * (self.p_g_prev - self._P_min) / (self._P_max - self._P_min) - 1.0
            if self.prev_slack:
                G_s = self.acfg.G_s
                ps = 2.0 * (self.p_g_s_prev - G_s.P_min) / (G_s.P_max - G_s.P_min) - 1.0
                prev = np.append(prev, np.clip(ps, -1.0, 1.0))
        return np.concatenate([s_n, prev]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.sampler.rng = np.random.default_rng(seed)
        self._episode = (self._fixed if self._fixed is not None
                         else self.sampler.episode(self.L))
        self.t = 0
        self.p_g_prev = self.p_g_s_prev = None
        return self._obs(), {}

    def step(self, action):
        a = torch.as_tensor(np.clip(np.asarray(action, dtype=float), -1.0, 1.0),
                            dtype=F64).reshape(1, -1)
        s = torch.as_tensor(self._episode[self.t], dtype=F64).reshape(1, -1)
        kw = {}
        if self.p_g_prev is not None:
            kw = dict(p_g_prev=torch.as_tensor(self.p_g_prev, dtype=F64).reshape(1, -1),
                      p_g_s_prev=torch.as_tensor([self.p_g_s_prev], dtype=F64))
        with torch.no_grad():
            out = self.critic(s, a, **kw)

        r_eq5 = float(out.r[0])
        z = out.z[0].numpy()
        z_sum = float(z.sum())
        reward = r_eq5 - self.rho * z_sum

        self.p_g_prev = out.p_g[0].numpy().copy()
        self.p_g_s_prev = float(out.p_g_s[0])
        self.t += 1
        truncated = self.t >= self.L          # time limit, not a terminal state

        info = {
            "r_eq5": r_eq5,
            "f_sum": float(out.f_sum[0]),
            "z_sum": z_sum,
            "feasible": float(np.linalg.norm(z) <= FEASIBLE_TOL),
            "helm_converged": float(out.helm.converged[0]),
        }
        obs = (np.zeros(self.obs_dim, np.float32) if truncated else self._obs())
        return obs, reward, False, truncated, info
