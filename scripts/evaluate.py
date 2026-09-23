"""Evaluate agents on Level 1 and save baseline metrics.

Examples:
  python scripts/evaluate.py --agent random --episodes 30 --seed 42
  python scripts/evaluate.py --agent ppo --model models/ppo_mario_level1_latest.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.agents.ppo_agent import PPOAgent
from mario_rl.agents.random_agent import RandomAgent
from mario_rl.env.mario_env import MarioEnv
from mario_rl.utils.eval_metrics import run_episode, summarize_episodes


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mario RL agents")
    parser.add_argument("--agent", choices=["random", "ppo"], default="random")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "ppo_mario_level1_latest.zip",
        help="Path to saved PPO model (.zip) when --agent ppo",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
    )
    parser.add_argument(
        "--level",
        type=Path,
        default=ROOT / "levels" / "level_1.txt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "metrics",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    max_steps = int(cfg.get("max_episode_steps", 2000))

    env = MarioEnv(level_path=args.level, max_episode_steps=max_steps)
    if args.agent == "random":
        agent = RandomAgent(n_actions=env.action_space.n, seed=seed)
    elif args.agent == "ppo":
        if not args.model.exists():
            raise SystemExit(
                f"Model not found: {args.model}\n"
                "Train first: python scripts/train.py"
            )
        agent = PPOAgent(args.model, deterministic=True)
    else:
        raise ValueError(f"Unsupported agent: {args.agent}")

    episodes = []
    for i in range(args.episodes):
        ep_seed = seed + i
        summary = run_episode(env, agent.act, seed=ep_seed)
        episodes.append(summary)

    metrics = summarize_episodes(episodes)
    payload = {
        "agent": args.agent,
        "model": str(args.model.as_posix()) if args.agent == "ppo" else None,
        "level": str(args.level.as_posix()),
        "base_seed": seed,
        "max_episode_steps": max_steps,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "episodes": episodes,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"{args.agent}_seed{seed}_ep{args.episodes}_{stamp}.json"
    csv_path = args.out_dir / f"{args.agent}_seed{seed}_ep{args.episodes}_{stamp}.csv"
    latest_json = args.out_dir / f"{args.agent}_baseline_latest.json"
    latest_csv = args.out_dir / f"{args.agent}_baseline_latest.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_episode_csv(csv_path, episodes)
    _write_episode_csv(latest_csv, episodes)

    print(f"agent={args.agent} episodes={metrics['episodes']} seed={seed}")
    print(f"mean_return={metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"mean_x_progress={metrics['mean_x_progress']:.4f}")
    print(f"mean_steps={metrics['mean_steps']:.2f}")
    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"death_rate={metrics['death_rate']:.4f}")
    print(f"truncation_rate={metrics['truncation_rate']:.4f}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


def _write_episode_csv(path: Path, episodes: list[dict]) -> None:
    if not episodes:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "seed",
        "return",
        "steps",
        "x_progress",
        "final_x",
        "reached_goal",
        "alive",
        "death_cause",
        "terminated",
        "truncated",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in episodes:
            writer.writerow({k: row.get(k) for k in fields})


if __name__ == "__main__":
    main()
