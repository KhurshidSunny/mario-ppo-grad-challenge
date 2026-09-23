"""Evaluate agents on Level 1 and save baseline / comparison metrics.

Examples:
  python scripts/evaluate.py --agent random --episodes 30 --seed 42
  python scripts/evaluate.py --agent ppo --model models/ppo_mario_level1_latest.zip
  python scripts/evaluate.py --compare --episodes 30 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
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


def evaluate_agent(env: MarioEnv, act_fn, *, episodes: int, seed: int) -> tuple[dict, list]:
    episodes_out = []
    for i in range(episodes):
        episodes_out.append(run_episode(env, act_fn, seed=seed + i))
    return summarize_episodes(episodes_out), episodes_out


def save_agent_payload(
    *,
    out_dir: Path,
    agent_name: str,
    model: Path | None,
    level: Path,
    seed: int,
    max_steps: int,
    metrics: dict,
    episodes: list,
) -> tuple[Path, Path]:
    payload = {
        "agent": agent_name,
        "model": str(model.as_posix()) if model is not None else None,
        "level": str(level.as_posix()),
        "base_seed": seed,
        "max_episode_steps": max_steps,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "episodes": episodes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{agent_name}_seed{seed}_ep{len(episodes)}_{stamp}.json"
    csv_path = out_dir / f"{agent_name}_seed{seed}_ep{len(episodes)}_{stamp}.csv"
    latest_json = out_dir / f"{agent_name}_baseline_latest.json"
    latest_csv = out_dir / f"{agent_name}_baseline_latest.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_episode_csv(csv_path, episodes)
    _write_episode_csv(latest_csv, episodes)
    return json_path, csv_path


def print_metrics(agent_name: str, metrics: dict, seed: int) -> None:
    print(f"agent={agent_name} episodes={metrics['episodes']} seed={seed}")
    print(f"mean_return={metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"mean_x_progress={metrics['mean_x_progress']:.4f}")
    print(f"mean_steps={metrics['mean_steps']:.2f}")
    print(f"success_rate={metrics['success_rate']:.4f}")
    print(f"death_rate={metrics['death_rate']:.4f}")
    print(f"truncation_rate={metrics['truncation_rate']:.4f}")


def write_comparison(
    *,
    out_dir: Path,
    plots_dir: Path,
    seed: int,
    episodes: int,
    random_m: dict,
    ppo_m: dict,
    model: Path,
) -> Path:
    rows = [
        {
            "metric": "mean_return",
            "random": random_m["mean_return"],
            "ppo": ppo_m["mean_return"],
            "delta_ppo_minus_random": ppo_m["mean_return"] - random_m["mean_return"],
            "ppo_wins": ppo_m["mean_return"] > random_m["mean_return"],
        },
        {
            "metric": "mean_x_progress",
            "random": random_m["mean_x_progress"],
            "ppo": ppo_m["mean_x_progress"],
            "delta_ppo_minus_random": ppo_m["mean_x_progress"] - random_m["mean_x_progress"],
            "ppo_wins": ppo_m["mean_x_progress"] > random_m["mean_x_progress"],
        },
        {
            "metric": "success_rate",
            "random": random_m["success_rate"],
            "ppo": ppo_m["success_rate"],
            "delta_ppo_minus_random": ppo_m["success_rate"] - random_m["success_rate"],
            "ppo_wins": ppo_m["success_rate"] > random_m["success_rate"],
        },
        {
            "metric": "death_rate",
            "random": random_m["death_rate"],
            "ppo": ppo_m["death_rate"],
            "delta_ppo_minus_random": ppo_m["death_rate"] - random_m["death_rate"],
            "ppo_wins": ppo_m["death_rate"] < random_m["death_rate"],
        },
        {
            "metric": "mean_steps",
            "random": random_m["mean_steps"],
            "ppo": ppo_m["mean_steps"],
            "delta_ppo_minus_random": ppo_m["mean_steps"] - random_m["mean_steps"],
            "ppo_wins": ppo_m["mean_steps"] > random_m["mean_steps"],
        },
    ]
    payload = {
        "seed": seed,
        "episodes": episodes,
        "model": str(model.as_posix()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "headline": {
            "ppo_beats_random_on_return": ppo_m["mean_return"] > random_m["mean_return"],
            "ppo_beats_random_on_x_progress": ppo_m["mean_x_progress"]
            > random_m["mean_x_progress"],
        },
        "random": random_m,
        "ppo": ppo_m,
        "table": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_latest.json"
    md_path = out_dir / "comparison_latest.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Random vs PPO (Level 1)",
        "",
        f"- episodes: {episodes}",
        f"- seed: {seed}",
        f"- model: `{model.as_posix()}`",
        "",
        "| Metric | Random | PPO | Delta (PPO-Random) | PPO better? |",
        "|--------|-------:|----:|------------------:|:-----------:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['metric']} | {r['random']:.4f} | {r['ppo']:.4f} | "
            f"{r['delta_ppo_minus_random']:.4f} | {'yes' if r['ppo_wins'] else 'no'} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Bar chart for headline metrics
    labels = ["mean_return", "mean_x_progress"]
    random_vals = [random_m["mean_return"], random_m["mean_x_progress"]]
    ppo_vals = [ppo_m["mean_return"], ppo_m["mean_x_progress"]]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - width / 2 for i in x], random_vals, width, label="Random", color="#c0392b")
    ax.bar([i + width / 2 for i in x], ppo_vals, width, label="PPO", color="#1f6aa5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(f"Random vs PPO — Level 1 (seed {seed}, {episodes} eps)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    plot_path = plots_dir / "comparison_random_vs_ppo.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\n=== COMPARISON TABLE ===")
    print(md_path.read_text(encoding="utf-8").encode("ascii", "replace").decode("ascii"))
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    print(f"saved: {plot_path}")
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mario RL agents")
    parser.add_argument("--agent", choices=["random", "ppo"], default="random")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Evaluate random and PPO, write one comparison table + plot",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "ppo_mario_level1_latest.zip",
        help="Path to saved PPO model (.zip) when --agent ppo or --compare",
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
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "results" / "plots",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    max_steps = int(cfg.get("max_episode_steps", 2000))

    env = MarioEnv(level_path=args.level, max_episode_steps=max_steps)

    if args.compare:
        if not args.model.exists():
            raise SystemExit(
                f"Model not found: {args.model}\n"
                "Train first: python scripts/train.py"
            )
        random_agent = RandomAgent(n_actions=env.action_space.n, seed=seed)
        ppo_agent = PPOAgent(args.model, deterministic=True)

        random_m, random_eps = evaluate_agent(
            env, random_agent.act, episodes=args.episodes, seed=seed
        )
        ppo_m, ppo_eps = evaluate_agent(
            env, ppo_agent.act, episodes=args.episodes, seed=seed
        )

        print_metrics("random", random_m, seed)
        save_agent_payload(
            out_dir=args.out_dir,
            agent_name="random",
            model=None,
            level=args.level,
            seed=seed,
            max_steps=max_steps,
            metrics=random_m,
            episodes=random_eps,
        )
        print_metrics("ppo", ppo_m, seed)
        save_agent_payload(
            out_dir=args.out_dir,
            agent_name="ppo",
            model=args.model,
            level=args.level,
            seed=seed,
            max_steps=max_steps,
            metrics=ppo_m,
            episodes=ppo_eps,
        )
        write_comparison(
            out_dir=args.out_dir,
            plots_dir=args.plots_dir,
            seed=seed,
            episodes=args.episodes,
            random_m=random_m,
            ppo_m=ppo_m,
            model=args.model,
        )
        return

    if args.agent == "random":
        agent = RandomAgent(n_actions=env.action_space.n, seed=seed)
        model_path = None
    elif args.agent == "ppo":
        if not args.model.exists():
            raise SystemExit(
                f"Model not found: {args.model}\n"
                "Train first: python scripts/train.py"
            )
        agent = PPOAgent(args.model, deterministic=True)
        model_path = args.model
    else:
        raise ValueError(f"Unsupported agent: {args.agent}")

    metrics, episodes = evaluate_agent(
        env, agent.act, episodes=args.episodes, seed=seed
    )
    print_metrics(args.agent, metrics, seed)
    json_path, csv_path = save_agent_payload(
        out_dir=args.out_dir,
        agent_name=args.agent,
        model=model_path,
        level=args.level,
        seed=seed,
        max_steps=max_steps,
        metrics=metrics,
        episodes=episodes,
    )
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
