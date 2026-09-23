"""Tests for the PPO agent wrapper and short training smoke path."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "levels" / "level_1.txt"

torch = pytest.importorskip("torch")
sb3 = pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402

from mario_rl.agents.ppo_agent import PPOAgent  # noqa: E402
from mario_rl.env.mario_env import MarioEnv  # noqa: E402
from mario_rl.utils.eval_metrics import run_episode  # noqa: E402


def test_ppo_short_train_save_load_act(tmp_path: Path):
    env = MarioEnv(level_path=LEVEL, max_episode_steps=200)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        verbose=0,
        seed=0,
    )
    model.learn(total_timesteps=128)
    save_path = tmp_path / "ppo_smoke"
    model.save(str(save_path))
    env.close()

    agent = PPOAgent(save_path.with_suffix(".zip"), deterministic=True)
    eval_env = MarioEnv(level_path=LEVEL, max_episode_steps=50)
    obs, _ = eval_env.reset(seed=1)
    action = agent.act(obs)
    assert 0 <= action < eval_env.action_space.n

    ep = run_episode(eval_env, agent.act, seed=1)
    assert ep["steps"] >= 1
    assert ep["terminated"] or ep["truncated"]
    eval_env.close()
