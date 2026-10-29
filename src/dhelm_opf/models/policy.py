"""
D-HELM policy network.

Paper basis
-----------
* Deterministic policy (Fig. 2b): the critic is exact, so there is no
  learned Q-function and no stochastic actor.
* Table 1, 5-bus: LSTM(64) -> Dense(64) -> Dense(64), SELU activations;
  policy input 8 + 4, policy output 4 + 4.
* Eq. (18): a = a_min + (a_max - a_min)(tanh(logit) + 1) / 2.  The
  network outputs tanh(logit) in [-1, 1]; the D-HELM's linear
  assignment (same map as ActionDecoder) applies a_min / a_max.
* Sec. 3.1: the LSTM accepts a sequence of states, so that the
  previous step's generator outputs p_{g,t-1} are available.

Reproduction choices
--------------------
* The "+4" extra inputs are the previous NORMALIZED active-power
  actions of the non-slack generators (|G / G_s| = 4).  Table 1's
  symbol is unreadable in the PDF; |G / G_s| fits the 5- and 30-bus
  columns exactly.
* Autoregressive: over a window of L states, the policy feeds its own
  previous generator action back as input.  At the start of a window
  the previous action is 0 (the middle of each generator's range).
* States are scaled to [-1, 1] with the sampling ranges before entering
  the network.
"""

from __future__ import annotations

import torch
from torch import nn


class DHelmPolicy(nn.Module):
    """
    LSTM policy producing a_hat = tanh(logit) in [-1, 1] (Eq. 18).

    Parameters
    ----------
    state_dim:
        Dimension of s_t (8 for 5-bus).
    n_v, n_g:
        Action layout (ActionConfig.n_v / n_g).
    state_low, state_high:
        Sampling ranges used to scale states to [-1, 1].
    hidden:
        Width of the LSTM and dense layers (Table 1: 64 for 5-bus).
    idx_p:
        Positions of the p_g actions in the output (ActionConfig.
        generator_action_indices).  Default: after the voltages (the
        pre-Eq.-4 layout).
    """

    def __init__(
        self,
        state_dim: int,
        n_v: int,
        n_g: int,
        state_low,
        state_high,
        hidden: int = 64,
        idx_p=None,
    ) -> None:

        super().__init__()

        self.state_dim = state_dim
        self.n_v = n_v
        self.n_g = n_g
        self.action_dim = n_v + n_g
        self.input_dim = state_dim + n_g
        self.idx_p = list(idx_p) if idx_p is not None else list(range(n_v, n_v + n_g))

        self.register_buffer(
            "state_low", torch.as_tensor(state_low, dtype=torch.float32)
        )
        self.register_buffer(
            "state_high", torch.as_tensor(state_high, dtype=torch.float32)
        )

        self.lstm = nn.LSTM(self.input_dim, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, self.action_dim)
        self.act = nn.SELU()

    def scale_state(self, s: torch.Tensor) -> torch.Tensor:
        """Map the state's sampling range -> [-1, 1]."""

        return (
            2.0 * (s - self.state_low)
            / (self.state_high - self.state_low)
            - 1.0
        )

    def step(self, s_t, a_hat_g_prev, hc=None):
        """
        One time step.

        s_t:           (B, state_dim)  state s_t (Eq. 2)
        a_hat_g_prev:  (B, n_g)        normalized p_g actions of step t-1
        hc:            LSTM state or None

        Returns (a_hat_t = tanh(logit_t), new hc); a_hat_t in [-1, 1]
        is mapped to the physical action a_t by Eq. (18).
        """

        x = torch.cat([self.scale_state(s_t), a_hat_g_prev], dim=1)

        h, hc = self.lstm(x.unsqueeze(1), hc)
        h = h.squeeze(1)

        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))

        logit = self.out(h)

        return torch.tanh(logit), hc

    def forward(self, s: torch.Tensor, a_hat_g_prev=None):
        """
        Run the policy over windows of states, autoregressively.

        s:             (B, L, state_dim)  (or (B, state_dim) for L = 1)
        a_hat_g_prev:  (B, n_g) p_g actions before the window;
                       zeros if None.

        Returns a_hat: (B, L, action_dim) normalized actions.
        """

        if s.dim() == 2:
            s = s.unsqueeze(1)

        s = s.to(self.state_low.dtype)
        B, L, _ = s.shape

        if a_hat_g_prev is None:
            prev = s.new_zeros(B, self.n_g)
        else:
            prev = a_hat_g_prev.to(s.dtype)

        hc = None
        a_hat = []

        for t in range(L):
            a_hat_t, hc = self.step(s[:, t], prev, hc)
            a_hat.append(a_hat_t)
            prev = a_hat_t[:, self.idx_p]

        return torch.stack(a_hat, dim=1)
