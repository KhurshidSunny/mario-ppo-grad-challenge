"""Record a gameplay GIF of an agent on Level 1.

Examples:
  python scripts/record_demo.py --agent ppo --model models/ppo_mario_level1_latest.zip
  python scripts/record_demo.py --agent random --out results/demos/random.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.agents.ppo_agent import PPOAgent
from mario_rl.agents.random_agent import RandomAgent
from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.render_frames import render_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Mario agent demo GIF")
    parser.add_argument("--agent", choices=["random", "ppo"], default="ppo")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "ppo_mario_level1_latest.zip",
    )
    parser.add_argument(
        "--level",
        type=Path,
        default=ROOT / "levels" / "level_1.txt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--every", type=int, default=2, help="Keep every Nth frame")
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "demos" / "best_agent.gif",
    )
    args = parser.parse_args()

    env = MarioEnv(level_path=args.level, max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)

    if args.agent == "random":
        agent = RandomAgent(n_actions=env.action_space.n, seed=args.seed)
    else:
        if not args.model.exists():
            raise SystemExit(
                f"Model not found: {args.model}\n"
                "Train first: python scripts/train.py"
            )
        agent = PPOAgent(args.model, deterministic=True)

    frames: list[Image.Image] = []
    frames.append(Image.fromarray(render_rgb(env.engine, scale=args.scale)))

    for step in range(1, args.max_steps + 1):
        action = agent.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        if step % args.every == 0 or terminated or truncated:
            frames.append(Image.fromarray(render_rgb(env.engine, scale=args.scale)))
        if terminated or truncated:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=False,
    )
    env.close()
    print(
        f"agent={args.agent} frames={len(frames)} steps={step} "
        f"final_x={info.get('x', 0):.1f} status="
        f"{'goal' if info.get('reached_goal') else ('dead:' + str(info.get('death_cause')) if not info.get('alive') else 'alive')}"
    )
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
