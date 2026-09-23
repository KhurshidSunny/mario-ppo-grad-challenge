"""Plot PPO learning curves from Monitor CSV logs.

Examples:
  python scripts/plot_learning_curves.py
  python scripts/plot_learning_curves.py --monitor results/train_logs/monitor/monitor.monitor.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_monitor(path: Path) -> pd.DataFrame:
    """Load SB3 Monitor CSV (first line is a JSON comment)."""
    if not path.exists():
        raise FileNotFoundError(f"Monitor log not found: {path}")
    df = pd.read_csv(path, comment="#")
    if not {"r", "l", "t"}.issubset(df.columns):
        raise ValueError(f"Unexpected monitor columns in {path}: {list(df.columns)}")
    df = df.copy()
    df["timesteps"] = df["l"].cumsum()
    return df


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(float)
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO learning curves")
    parser.add_argument(
        "--monitor",
        type=Path,
        default=ROOT / "results" / "train_logs" / "monitor" / "monitor.monitor.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "plots",
    )
    parser.add_argument("--window", type=int, default=50, help="Rolling mean window")
    parser.add_argument(
        "--random-return",
        type=float,
        default=7.84,
        help="Random baseline mean return for reference line",
    )
    args = parser.parse_args()

    df = load_monitor(args.monitor)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rewards = df["r"].to_numpy(dtype=float)
    steps = df["timesteps"].to_numpy(dtype=float)
    smooth = rolling_mean(rewards, args.window)

    # Reward vs timesteps
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, color="#9aa7b5", alpha=0.35, linewidth=0.8, label="Episode return")
    ax.plot(steps, smooth, color="#1f6aa5", linewidth=2.0, label=f"Rolling mean (w={args.window})")
    ax.axhline(
        args.random_return,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Random baseline ({args.random_return:.2f})",
    )
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode return")
    ax.set_title("PPO learning curve — Level 1 (seed 42)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    reward_path = args.out_dir / "learning_curve_reward.png"
    fig.savefig(reward_path, dpi=150)
    plt.close(fig)

    # Episode length vs timesteps (proxy for survival / progress)
    lengths = df["l"].to_numpy(dtype=float)
    length_smooth = rolling_mean(lengths, args.window)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, lengths, color="#9aa7b5", alpha=0.35, linewidth=0.8, label="Episode length")
    ax.plot(steps, length_smooth, color="#2d8a5d", linewidth=2.0, label=f"Rolling mean (w={args.window})")
    ax.set_xlabel("Environment timesteps")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title("PPO episode length — Level 1 (seed 42)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    length_path = args.out_dir / "learning_curve_ep_length.png"
    fig.savefig(length_path, dpi=150)
    plt.close(fig)

    print(f"episodes={len(df)} final_timesteps={int(steps[-1]) if len(steps) else 0}")
    print(f"saved: {reward_path}")
    print(f"saved: {length_path}")


if __name__ == "__main__":
    main()
