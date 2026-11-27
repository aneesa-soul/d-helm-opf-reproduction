"""
Baseline agents in Tianshou 0.5.1 (paper Sec. 4.1-4.2): DDPG, SAC
("improved": entropy maximization, auto-tuned temperature, two critics
and two targets, prioritized replay) and PPO.

From the paper: actors per Table 1; critics per Sec. 4.2 (ReLU); learning
rate 1e-4 and target smoothing tau = 1e-3; minibatch per Table 1.

Our choices (not in the paper), all Tianshou defaults or the values of
Tianshou's own examples:
* discount gamma = 0.99;
* DDPG exploration: Gaussian noise, sigma = 0.1;
* SAC: target entropy = -|A|, temperature lr = 3e-4, prioritized replay
  alpha = 0.6, beta = 0.4;
* PPO: clip 0.2, GAE lambda 0.95, value coefficient 0.5, entropy
  coefficient 0.01, advantage normalization; 8 episodes per update,
  10 passes; reward normalization and gradient-norm clipping at 0.5
  (both from Tianshou's PPO examples); tanh-bounded mean (Tianshou's
  ActorProb default; see nets.py); log probability ratio clamped to
  +-20 before exp() (StablePPOPolicy, against float32 overflow);
* off-policy: one gradient update per environment step after 10 random
  warm-up episodes; replay buffer holding all transitions;
* recurrent training windows: stack_num = 24 (the whole episode, as for
  the proposed method).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.exploration import GaussianNoise
from tianshou.policy import DDPGPolicy, PPOPolicy, SACPolicy
from torch.distributions import Independent, Normal

from dhelm_opf.baselines.nets import (
    CRITIC_SIZES,
    DeterministicActor,
    GaussianActor,
    QCritic,
    VCritic,
)
from dhelm_opf.environment.cases import get_case

ALGOS = ("ddpg", "sac", "ppo")


@dataclass(frozen=True)
class BaselineConfig:
    lr: float = 1e-4              # paper
    tau: float = 1e-3             # paper ("smoothing factor")
    gamma: float = 0.99
    ddpg_noise: float = 0.1
    sac_alpha_lr: float = 3e-4
    per_alpha: float = 0.6
    per_beta: float = 0.4
    ppo_episodes_per_update: int = 8
    ppo_repeat: int = 10
    warmup_episodes: int = 10
    stack_num: int = 24


def dist_fn(*logits):
    return Independent(Normal(*logits), 1)


LOG_RATIO_MAX = 20.0


class StablePPOPolicy(PPOPolicy):
    """Tianshou 0.5.1 PPOPolicy.learn with one change: the log probability
    ratio is clamped to +-LOG_RATIO_MAX before exp().

    Within a single update the 11-dimensional (30-bus) policy sometimes
    moved so far that log(pi_new / pi_old) reached 45-82; exp() of that
    gave a clip loss of ~5e17, an infinite gradient norm, and (past ~88,
    float32 overflow) NaN weights.  exp(20) ~ 5e8 is still far outside the
    clip range [0.8, 1.2], so normal updates are unchanged; only the
    overflow is removed.  Our departure from Tianshou, not in the paper.
    """

    def learn(self, batch, batch_size, repeat, **kwargs):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                log_ratio = dist.log_prob(minibatch.act) - minibatch.logp_old
                ratio = log_ratio.clamp(-LOG_RATIO_MAX, LOG_RATIO_MAX).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip,
                                    1.0 + self._eps_clip) * minibatch.adv
                clip_loss = -torch.min(surr1, surr2).mean()
                value = self.critic(minibatch.obs).flatten()
                vf_loss = (minibatch.returns - value).pow(2).mean()
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:
                    nn.utils.clip_grad_norm_(self._actor_critic.parameters(),
                                             max_norm=self._grad_norm)
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        return {"loss": losses, "loss/clip": clip_losses,
                "loss/vf": vf_losses, "loss/ent": ent_losses}


def build(algo: str, case: str, obs_dim: int, act_dim: int, action_space,
          total_steps: int, cfg: BaselineConfig = BaselineConfig(),
          device: str = "cpu"):
    """Return (policy, buffer) for one algorithm on one case."""

    spec = get_case(case)
    h, sizes, bs = spec.hidden, CRITIC_SIZES[case], spec.batch_size

    if algo == "ddpg":
        actor = DeterministicActor(obs_dim, act_dim, h, device).to(device)
        critic = QCritic(obs_dim, act_dim, sizes, device).to(device)
        policy = DDPGPolicy(
            actor, torch.optim.Adam(actor.parameters(), lr=cfg.lr),
            critic, torch.optim.Adam(critic.parameters(), lr=cfg.lr),
            tau=cfg.tau, gamma=cfg.gamma,
            exploration_noise=GaussianNoise(sigma=cfg.ddpg_noise),
            action_space=action_space, action_scaling=False,
            action_bound_method="clip",
        )
        buffer = VectorReplayBuffer(total_steps, 1, stack_num=cfg.stack_num)

    elif algo == "sac":
        actor = GaussianActor(obs_dim, act_dim, h, conditioned_sigma=True,
                              device=device).to(device)
        c1 = QCritic(obs_dim, act_dim, sizes, device).to(device)
        c2 = QCritic(obs_dim, act_dim, sizes, device).to(device)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = (-float(act_dim), log_alpha,
                 torch.optim.Adam([log_alpha], lr=cfg.sac_alpha_lr))
        policy = SACPolicy(
            actor, torch.optim.Adam(actor.parameters(), lr=cfg.lr),
            c1, torch.optim.Adam(c1.parameters(), lr=cfg.lr),
            c2, torch.optim.Adam(c2.parameters(), lr=cfg.lr),
            tau=cfg.tau, gamma=cfg.gamma, alpha=alpha,
            deterministic_eval=True,
            action_space=action_space, action_scaling=False,
        )
        buffer = PrioritizedVectorReplayBuffer(
            total_steps, 1, alpha=cfg.per_alpha, beta=cfg.per_beta,
            stack_num=cfg.stack_num)

    elif algo == "ppo":
        actor = GaussianActor(obs_dim, act_dim, h, conditioned_sigma=False,
                              device=device, bounded_mean=True).to(device)
        critic = VCritic(obs_dim, sizes, device).to(device)
        params = list(actor.parameters()) + list(critic.parameters())
        policy = StablePPOPolicy(
            actor, critic, torch.optim.Adam(params, lr=cfg.lr), dist_fn,
            discount_factor=cfg.gamma, eps_clip=0.2, gae_lambda=0.95,
            vf_coef=0.5, ent_coef=0.01, advantage_normalization=True,
            reward_normalization=True, max_grad_norm=0.5,
            deterministic_eval=True,
            action_space=action_space, action_scaling=False,
            action_bound_method="clip",
        )
        buffer = VectorReplayBuffer(cfg.ppo_episodes_per_update * 24, 1,
                                    stack_num=cfg.stack_num)
    else:
        raise ValueError(f"algo must be one of {ALGOS}")

    return policy, buffer, bs
