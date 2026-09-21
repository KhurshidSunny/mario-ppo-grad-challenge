"""Gymnasium wrapper around the custom platformer engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_rl.env.observations import OBS_DIM, build_observation
from mario_rl.env.rewards import compute_reward
from mario_rl.game.constants import ACTION_RIGHT_JUMP
from mario_rl.game.engine import GameEngine

DEFAULT_LEVEL = Path(__file__).resolve().parents[3] / "levels" / "level_1.txt"
N_ACTIONS = ACTION_RIGHT_JUMP + 1


class MarioEnv(gym.Env):
    """Discrete-action Mario-style environment with feature observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_path: str | Path | None = None,
        max_episode_steps: int = 2000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.level_path = Path(level_path) if level_path is not None else DEFAULT_LEVEL
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode

        self.engine = GameEngine(self.level_path)
        self._elapsed_steps = 0
        self._prev_x = 0.0

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        reset_seed = 0 if seed is None else int(seed)
        self.engine.reset(seed=reset_seed)
        self._elapsed_steps = 0
        self._prev_x = self.engine.player.x
        obs = build_observation(self.engine)
        return obs, self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        prev_x = self.engine.player.x
        was_alive = self.engine.alive
        had_goal = self.engine.reached_goal_flag

        self.engine.step(int(action))
        self._elapsed_steps += 1

        reached_goal = self.engine.reached_goal_flag and not had_goal
        died = (not self.engine.alive) and was_alive

        reward = compute_reward(
            prev_x,
            self.engine.player.x,
            reached_goal=reached_goal,
            died=died,
        )

        terminated = (not self.engine.alive) or self.engine.reached_goal_flag
        truncated = (not terminated) and self._elapsed_steps >= self.max_episode_steps

        obs = build_observation(self.engine)
        self._prev_x = self.engine.player.x
        return obs, reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        player = self.engine.player
        return {
            "x": player.x,
            "y": player.y,
            "x_progress": player.x - self.engine.level.player_start[0],
            "steps": self._elapsed_steps,
            "alive": self.engine.alive,
            "reached_goal": self.engine.reached_goal_flag,
            "death_cause": self.engine.death_cause,
            "on_ground": player.on_ground,
        }
