"""Helpers for running and summarizing evaluation episodes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from mario_rl.env.mario_env import MarioEnv


def run_episode(
    env: MarioEnv,
    act_fn: Callable[[np.ndarray], int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run one episode and return summary fields."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while True:
        action = int(act_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "return": total_reward,
        "steps": steps,
        "x_progress": float(info.get("x_progress", 0.0)),
        "final_x": float(info.get("x", 0.0)),
        "reached_goal": bool(info.get("reached_goal", False)),
        "alive": bool(info.get("alive", True)),
        "death_cause": info.get("death_cause"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mean reward, distance, death rate, and success rate."""
    n = len(episodes)
    if n == 0:
        return {
            "episodes": 0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "mean_x_progress": 0.0,
            "mean_steps": 0.0,
            "success_rate": 0.0,
            "death_rate": 0.0,
            "truncation_rate": 0.0,
        }

    returns = np.array([e["return"] for e in episodes], dtype=np.float64)
    progress = np.array([e["x_progress"] for e in episodes], dtype=np.float64)
    steps = np.array([e["steps"] for e in episodes], dtype=np.float64)
    successes = sum(1 for e in episodes if e["reached_goal"])
    deaths = sum(1 for e in episodes if (not e["alive"]) and (not e["reached_goal"]))
    truncations = sum(1 for e in episodes if e["truncated"])

    return {
        "episodes": n,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=0)),
        "mean_x_progress": float(progress.mean()),
        "mean_steps": float(steps.mean()),
        "success_rate": successes / n,
        "death_rate": deaths / n,
        "truncation_rate": truncations / n,
    }
