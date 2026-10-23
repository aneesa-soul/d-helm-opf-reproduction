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
    LSTM policy producing normalized actions in [-1, 1].

    Parameters
    ----------
    state_dim:
        Dimension of s_t (8 for 5-bus).
    n_voltage, n_generator:
        Action layout (ActionConfig.n_voltage / n_generator).
    state_low, state_high:
        Sampling ranges used to scale states to [-1, 1].
    hidden:
        Width of the LSTM and dense layers (Table 1: 64 for 5-bus).
    """

    def __init__(
        self,
        state_dim: int,
        n_voltage: int,
        n_generator: int,
        state_low,
        state_high,
        hidden: int = 64,
    ) -> None:

        super().__init__()

        self.state_dim = state_dim
        self.n_voltage = n_voltage
        self.n_generator = n_generator
        self.action_dim = n_voltage + n_generator
        self.input_dim = state_dim + n_generator

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

    def scale_state(self, state: torch.Tensor) -> torch.Tensor:
        """Map sampling range -> [-1, 1]."""

        return (
            2.0 * (state - self.state_low)
            / (self.state_high - self.state_low)
            - 1.0
        )

    def step(self, state_t, prev_gen_action, hc=None):
        """
        One time step.

        state_t:          (B, state_dim)
        prev_gen_action:  (B, n_generator) normalized, from step t-1
        hc:               LSTM state or None

        Returns (action_t (B, action_dim) in [-1, 1], new hc).
        """

        x = torch.cat([self.scale_state(state_t), prev_gen_action], dim=1)

        h, hc = self.lstm(x.unsqueeze(1), hc)
        h = h.squeeze(1)

        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))

        return torch.tanh(self.out(h)), hc

    def forward(self, states: torch.Tensor, prev_gen_action=None):
        """
        Run the policy over windows of states, autoregressively.

        states:           (B, L, state_dim)  (or (B, state_dim) for L = 1)
        prev_gen_action:  (B, n_generator) action before the window;
                          zeros if None.

        Returns (B, L, action_dim) normalized actions.
        """

        if states.dim() == 2:
            states = states.unsqueeze(1)

        states = states.to(self.state_low.dtype)
        B, L, _ = states.shape

        if prev_gen_action is None:
            prev = states.new_zeros(B, self.n_generator)
        else:
            prev = prev_gen_action.to(states.dtype)

        hc = None
        actions = []

        for t in range(L):
            a_t, hc = self.step(states[:, t], prev, hc)
            actions.append(a_t)
            prev = a_t[:, self.n_voltage:]

        return torch.stack(actions, dim=1)
