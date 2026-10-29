"""
State sampling and the states-only replay buffer.

Paper basis
-----------
* States s_t = (p_d, q_d, p_w) (Eq. 2).  Demands are sampled in
  [60%, 110%] and wind in [40%, 100%] of rated output (Sec. 4.1).
* The replay buffer stores ONLY system states, not actions, rewards or
  violations (Sec. 3, Fig. 2b): the exact critic recomputes those for
  every update.
* The LSTM policy accepts a sequence of states (Sec. 3.1), so the buffer
  keeps states in episode order and returns windows of consecutive
  states.

Reproduction choices
--------------------
* States are drawn independently at every step (the paper uses no
  time-series dataset).
* Episode length (default 25) is inferred from "20 episodes
  (<= 500 steps)" in Sec. 4.2.
"""

from __future__ import annotations

import numpy as np

from dhelm_opf.environment.uncertainty import UncertaintyConfig


class StateSampler:
    """
    Draw states in the paper's ranges.

    State layout (matches scenario_to_state and DHelmNetwork):

        [p_d scale (n_d), q_d scale (n_d), p_w scale (n_w)]
    """

    def __init__(
        self,
        n_d: int,
        n_w: int,
        config: UncertaintyConfig | None = None,
        seed: int | None = None,
    ) -> None:

        self.config = config or UncertaintyConfig()
        self.n_d = n_d
        self.n_w = n_w
        self.state_dim = 2 * n_d + n_w
        self.rng = np.random.default_rng(seed)

        c = self.config
        self.low = np.array(
            [c.demand_min] * (2 * n_d) + [c.wind_min] * n_w
        )
        self.high = np.array(
            [c.demand_max] * (2 * n_d) + [c.wind_max] * n_w
        )

    def sample(self, n: int) -> np.ndarray:
        """(n, state_dim) independent states."""

        return self.rng.uniform(self.low, self.high, (n, self.state_dim))

    def episode(self, length: int) -> np.ndarray:
        """(length, state_dim) states of one episode."""

        return self.sample(length)


class StateBuffer:
    """
    Replay buffer that stores only states, grouped by episode.

    Episodes are stored as rows of a (n_episodes, length, state_dim)
    array; all episodes must have the same length.
    """

    def __init__(self, capacity_episodes: int = 10_000) -> None:

        self.capacity = capacity_episodes
        self._episodes: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def n_states(self) -> int:
        return sum(len(e) for e in self._episodes)

    def add_episode(self, states: np.ndarray) -> None:

        states = np.asarray(states, dtype=float)

        if self._episodes and states.shape != self._episodes[0].shape:
            raise ValueError(
                "All episodes must have the same shape; got "
                f"{states.shape} vs {self._episodes[0].shape}."
            )

        self._episodes.append(states)

        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def sample_windows(
        self,
        batch: int,
        window: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        (batch, window, state_dim) windows of consecutive states.

        Each window comes from one episode, at a random start.  The
        policy runs over the window autoregressively, so the first state
        of a window is treated like the start of an episode.
        """

        if not self._episodes:
            raise RuntimeError("The buffer is empty.")

        length = self._episodes[0].shape[0]

        if window > length:
            raise ValueError(
                f"window {window} is longer than the episodes ({length})."
            )

        data = np.stack(self._episodes)              # (E, T, d)
        ep = rng.integers(0, len(data), size=batch)
        start = rng.integers(0, length - window + 1, size=batch)

        offsets = start[:, None] + np.arange(window)[None, :]

        return data[ep[:, None], offsets]
