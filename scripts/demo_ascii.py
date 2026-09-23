"""ASCII demo: run an agent for a few frames and print level snapshots.

Examples:
  python scripts/demo_ascii.py --agent random
  python scripts/demo_ascii.py --agent ppo --model models/ppo_mario_level1_latest.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.agents.ppo_agent import PPOAgent
from mario_rl.agents.random_agent import RandomAgent
from mario_rl.env.mario_env import MarioEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="ASCII play demo")
    parser.add_argument("--agent", choices=["random", "ppo"], default="ppo")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "ppo_mario_level1_latest.zip",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--every", type=int, default=5, help="Print every N steps")
    parser.add_argument(
        "--level",
        type=Path,
        default=ROOT / "levels" / "level_1.txt",
    )
    args = parser.parse_args()

    env = MarioEnv(level_path=args.level, max_episode_steps=2000)
    obs, info = env.reset(seed=args.seed)

    if args.agent == "random":
        agent = RandomAgent(n_actions=env.action_space.n, seed=args.seed)
    else:
        model_path = args.model
        if not model_path.exists():
            raise SystemExit(
                f"Model not found: {model_path}\n"
                "Train first: python scripts/train.py --timesteps 20000"
            )
        agent = PPOAgent(model_path, deterministic=True)

    print(f"agent={args.agent} seed={args.seed}")
    print(
        f"start x={info['x']:.2f} y={info['y']:.2f} "
        f"alive={info['alive']} goal={info['reached_goal']}"
    )
    print(env.engine.ascii_snapshot())
    print()

    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if step % args.every == 0 or terminated or truncated:
            status = "alive"
            if info["reached_goal"]:
                status = "GOAL"
            elif not info["alive"]:
                status = f"dead ({info['death_cause']})"
            print(
                f"step={step:03d} action={action} reward={reward:.3f} "
                f"x={info['x']:.2f} status={status}"
            )
            print(env.engine.ascii_snapshot())
            print()
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
