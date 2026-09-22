"""Tests for the random agent and evaluation helpers."""

from pathlib import Path

from mario_rl.agents.random_agent import RandomAgent
from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.eval_metrics import run_episode, summarize_episodes

ROOT = Path(__file__).resolve().parents[1]
LEVEL = ROOT / "levels" / "level_1.txt"


def test_random_agent_is_deterministic_with_seed():
    a = RandomAgent(n_actions=5, seed=7)
    b = RandomAgent(n_actions=5, seed=7)
    actions_a = [a.act() for _ in range(20)]
    actions_b = [b.act() for _ in range(20)]
    assert actions_a == actions_b
    assert all(0 <= x < 5 for x in actions_a)


def test_run_episode_and_summary():
    env = MarioEnv(level_path=LEVEL, max_episode_steps=50)
    agent = RandomAgent(n_actions=env.action_space.n, seed=0)
    ep = run_episode(env, agent.act, seed=0)
    assert "return" in ep
    assert ep["steps"] >= 1
    assert ep["terminated"] or ep["truncated"]

    summary = summarize_episodes([ep, ep])
    assert summary["episodes"] == 2
    assert 0.0 <= summary["success_rate"] <= 1.0
    assert 0.0 <= summary["death_rate"] <= 1.0
