"""PPO policy wrapper around a saved Stable-Baselines3 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    """Loads a trained PPO checkpoint and exposes a simple ``act`` API."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        deterministic: bool = True,
    ):
        path = Path(model_path)
        # SB3 accepts path with or without .zip
        load_path = path.with_suffix("") if path.suffix == ".zip" else path
        self.model = PPO.load(str(load_path))
        self.deterministic = bool(deterministic)

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)
