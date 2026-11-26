"""
Networks for the baseline agents (paper Sec. 4.2, Table 1).

Actors: "The policy network of all test cases is designed as in Table 1"
-> the same LSTM(h) -> Dense h (SELU) -> Dense h (SELU) -> output network
as the proposed method, h = 64 / 64 / 128 / 256.

* DeterministicActor (DDPG): output -> tanh.
* GaussianActor (SAC, PPO): mean from one Table 1 network.
    - SAC: "stochastic actors with conditioned sigma (learnable standard
      deviation), where an additional network with the same structure as
      the actor is employed to predict sigma values" -> a second Table 1
      network for log sigma.
    - PPO: the paper does not say; a state-independent learnable log
      sigma (Tianshou's default for PPO) is used, and the mean is bounded
      with tanh (Tianshou's ActorProb default, unbounded=False).  With an
      unbounded mean PPO pushed it far outside the clipped action range
      (|mu| ~ 8 within 80 updates) until the probability ratios
      overflowed to NaN.  SAC keeps an unbounded mean: it squashes the
      sampled action with tanh itself.

Critics: "The Q function is approximated by two critic networks, which
accept the actions and states" with hidden sizes 64/64, 128/64,
256/256/128 (printed 256/265/128) and 512/128/128, ReLU.  PPO's value
function V(s) uses the same layer sizes.

Recurrence follows Tianshou 0.5.1's convention (utils.net.common.
Recurrent): obs of shape (B, d) with a hidden-state dict while
collecting, (B, T, d) stacked windows (buffer stack_num) while training.
Critics read the last frame of a stacked window.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

CRITIC_SIZES = {
    "case5": (64, 64),
    "case30": (128, 64),
    "case118": (256, 256, 128),
    "case300": (512, 128, 128),
}

LOG_SIG_MIN, LOG_SIG_MAX = -20.0, 2.0


class TableOneNet(nn.Module):
    """LSTM(h) -> Dense h SELU -> Dense h SELU -> Linear(out) (Table 1)."""

    def __init__(self, obs_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.SELU()

    def forward(self, obs, state=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if state is None:
            y, (h, c) = self.lstm(obs)
        else:
            y, (h, c) = self.lstm(obs, (state["hidden"].transpose(0, 1).contiguous(),
                                        state["cell"].transpose(0, 1).contiguous()))
        x = self.act(self.fc1(y[:, -1]))
        x = self.act(self.fc2(x))
        return self.out(x), {"hidden": h.transpose(0, 1).detach(),
                             "cell": c.transpose(0, 1).detach()}


def _as_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


class DeterministicActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden, device="cpu") -> None:
        super().__init__()
        self.net = TableOneNet(obs_dim, act_dim, hidden)
        self.device = device

    def forward(self, obs, state=None, info={}):
        y, hidden = self.net(_as_tensor(obs, self.device), _split(state, "mu"))
        return torch.tanh(y), _join(hidden, None)


class GaussianActor(nn.Module):
    """(mu, sigma) for Tianshou's SAC / PPO policies."""

    def __init__(self, obs_dim, act_dim, hidden, conditioned_sigma, device="cpu",
                 bounded_mean=False) -> None:
        super().__init__()
        self.mu = TableOneNet(obs_dim, act_dim, hidden)
        self.bounded_mean = bounded_mean
        self.conditioned_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = TableOneNet(obs_dim, act_dim, hidden)
        else:
            self.log_sigma = nn.Parameter(torch.full((act_dim,), -0.5))
        self.device = device

    def forward(self, obs, state=None, info={}):
        obs = _as_tensor(obs, self.device)
        mu, h_mu = self.mu(obs, _split(state, "mu"))
        if self.bounded_mean:
            mu = torch.tanh(mu)
        if self.conditioned_sigma:
            log_sigma, h_sig = self.sigma(obs, _split(state, "sigma"))
        else:
            log_sigma, h_sig = self.log_sigma.expand_as(mu), None
        sigma = torch.clamp(log_sigma, LOG_SIG_MIN, LOG_SIG_MAX).exp()
        return (mu, sigma), _join(h_mu, h_sig)


def _split(state, key):
    if state is None:
        return None
    return {"hidden": state[f"{key}_hidden"], "cell": state[f"{key}_cell"]}


def _join(h_mu, h_sig):
    out = {"mu_hidden": h_mu["hidden"], "mu_cell": h_mu["cell"]}
    if h_sig is not None:
        out.update(sigma_hidden=h_sig["hidden"], sigma_cell=h_sig["cell"])
    return out


def mlp(in_dim: int, sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class QCritic(nn.Module):
    """Q(s, a) with ReLU layers (last frame of a stacked window)."""

    def __init__(self, obs_dim, act_dim, sizes, device="cpu") -> None:
        super().__init__()
        self.net = mlp(obs_dim + act_dim, sizes, 1)
        self.device = device

    def forward(self, obs, act, info={}):
        obs = _as_tensor(obs, self.device)
        if obs.dim() == 3:
            obs = obs[:, -1]
        act = _as_tensor(act, self.device)
        return self.net(torch.cat([obs, act], dim=1))


class VCritic(nn.Module):
    """V(s) for PPO, same layer sizes."""

    def __init__(self, obs_dim, sizes, device="cpu") -> None:
        super().__init__()
        self.net = mlp(obs_dim, sizes, 1)
        self.device = device

    def forward(self, obs, **kwargs):
        obs = _as_tensor(obs, self.device)
        if obs.dim() == 3:
            obs = obs[:, -1]
        return self.net(obs)
