"""Tests for rendering / plotting helpers."""

from pathlib import Path

import numpy as np

from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.render_frames import render_rgb

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "levels" / "level_1.txt"


def test_render_rgb_shape_and_dtype():
    env = MarioEnv(level_path=LEVEL, max_episode_steps=50)
    env.reset(seed=0)
    frame = render_rgb(env.engine, scale=2)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.shape[0] > 0 and frame.shape[1] > 0
    env.close()
