"""Random-action baseline agent."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Chooses a discrete action uniformly at random."""

    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_actions = int(n_actions)
        self._rng = np.random.default_rng(seed)

    def act(self, observation=None) -> int:
        _ = observation
        return int(self._rng.integers(0, self.n_actions))
