"""Default progress-shaped reward for Level 1."""

from __future__ import annotations

# Tunable weights used by MarioEnv (Day 7 light tune).
PROGRESS_SCALE = 0.12
TIME_PENALTY = 0.01
GOAL_BONUS = 10.0
DEATH_PENALTY = 1.5


def compute_reward(
    prev_x: float,
    curr_x: float,
    *,
    reached_goal: bool,
    died: bool,
) -> float:
    """Reward forward progress, penalize time and death, bonus on goal."""
    reward = (curr_x - prev_x) * PROGRESS_SCALE
    reward -= TIME_PENALTY
    if reached_goal:
        reward += GOAL_BONUS
    if died:
        reward -= DEATH_PENALTY
    return float(reward)
