"""API checks for the Gymnasium Mario environment."""

from pathlib import Path

import numpy as np
import pytest

from mario_rl.env.mario_env import MarioEnv
from mario_rl.env.observations import OBS_DIM
from mario_rl.game.constants import ACTION_NOOP, ACTION_RIGHT

ROOT = Path(__file__).resolve().parents[1]
LEVEL_PATH = ROOT / "levels" / "level_1.txt"


@pytest.fixture
def env() -> MarioEnv:
    return MarioEnv(level_path=LEVEL_PATH, max_episode_steps=200)


def test_spaces(env: MarioEnv):
    assert env.action_space.n == 5
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32


def test_reset_returns_obs_and_info(env: MarioEnv):
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "x" in info
    assert "reached_goal" in info
    assert info["alive"] is True


def test_step_tuple_length_and_types(env: MarioEnv):
    env.reset(seed=0)
    result = env.step(ACTION_RIGHT)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reset_seed_is_deterministic(env: MarioEnv):
    obs_a, _ = env.reset(seed=3)
    path_a = [obs_a.copy()]
    for action in [ACTION_RIGHT, ACTION_NOOP, ACTION_RIGHT]:
        obs, _, terminated, truncated, _ = env.step(action)
        path_a.append(obs.copy())
        if terminated or truncated:
            break

    obs_b, _ = env.reset(seed=3)
    path_b = [obs_b.copy()]
    for action in [ACTION_RIGHT, ACTION_NOOP, ACTION_RIGHT]:
        obs, _, terminated, truncated, _ = env.step(action)
        path_b.append(obs.copy())
        if terminated or truncated:
            break

    assert len(path_a) == len(path_b)
    for a, b in zip(path_a, path_b):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_truncation_at_max_steps():
    env = MarioEnv(level_path=LEVEL_PATH, max_episode_steps=5)
    env.reset(seed=0)
    terminated = False
    truncated = False
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(ACTION_NOOP)
        if terminated or truncated:
            break
    assert truncated is True or terminated is True
    if not terminated:
        assert truncated is True


def test_info_tracks_progress(env: MarioEnv):
    _, info0 = env.reset(seed=0)
    start_x = info0["x"]
    _, _, _, _, info1 = env.step(ACTION_RIGHT)
    assert info1["steps"] == 1
    assert info1["x"] >= start_x - 1e-6
