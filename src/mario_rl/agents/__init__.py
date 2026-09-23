"""Public agent exports."""

from mario_rl.agents.random_agent import RandomAgent

__all__ = ["RandomAgent"]

try:
    from mario_rl.agents.ppo_agent import PPOAgent

    __all__.append("PPOAgent")
except ImportError:
    # torch / stable-baselines3 not installed yet
    pass
