"""Train a PPO agent on Level 1 with Stable-Baselines3.

Examples:
  python scripts/train.py
  python scripts/train.py --timesteps 20000
  python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.env.mario_env import MarioEnv


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_env(level: Path, max_steps: int, seed: int, monitor_dir: Path | None = None):
    env = MarioEnv(level_path=level, max_episode_steps=max_steps)
    env.reset(seed=seed)
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_dir / "monitor"))
    else:
        env = Monitor(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on Mario Level 1")
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
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "results" / "train_logs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ppo_mario_level1",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    timesteps = (
        args.timesteps
        if args.timesteps is not None
        else int(cfg.get("total_timesteps", 200_000))
    )
    max_steps = int(cfg.get("max_episode_steps", 2000))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.log_dir / "checkpoints"
    eval_dir = args.log_dir / "eval"
    monitor_dir = args.log_dir / "monitor"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.level, max_steps, seed, monitor_dir=monitor_dir)
    eval_env = make_env(args.level, max_steps, seed + 10_000)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        n_steps=int(cfg.get("n_steps", 2048)),
        batch_size=int(cfg.get("batch_size", 64)),
        gamma=float(cfg.get("gamma", 0.99)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        verbose=1,
        seed=seed,
        tensorboard_log=None,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000, timesteps // 5),
        save_path=str(checkpoint_dir),
        name_prefix=args.model_name,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=max(5_000, timesteps // 10),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print(f"Training PPO for {timesteps} timesteps (seed={seed})")
    model.learn(total_timesteps=timesteps, callback=[checkpoint_cb, eval_cb])
    elapsed_s = time.perf_counter() - started

    final_path = args.out_dir / args.model_name
    model.save(str(final_path))
    # Prefer best eval model as "latest" if it exists
    best_zip = eval_dir / "best_model.zip"
    latest_path = args.out_dir / f"{args.model_name}_latest"
    if best_zip.exists():
        best = PPO.load(str(best_zip.with_suffix("")))
        best.save(str(latest_path))
        used_best = True
    else:
        model.save(str(latest_path))
        used_best = False

    meta = {
        "algorithm": "PPO",
        "library": "stable-baselines3",
        "policy": "MlpPolicy",
        "level": str(args.level.as_posix()),
        "seed": seed,
        "total_timesteps": timesteps,
        "max_episode_steps": max_steps,
        "config": {
            "learning_rate": float(cfg.get("learning_rate", 3e-4)),
            "n_steps": int(cfg.get("n_steps", 2048)),
            "batch_size": int(cfg.get("batch_size", 64)),
            "gamma": float(cfg.get("gamma", 0.99)),
            "clip_range": float(cfg.get("clip_range", 0.2)),
            "ent_coef": float(cfg.get("ent_coef", 0.01)),
        },
        "model_path": str((args.out_dir / f"{args.model_name}.zip").as_posix()),
        "latest_path": str((args.out_dir / f"{args.model_name}_latest.zip").as_posix()),
        "used_best_eval_model": used_best,
        "wall_clock_seconds": round(elapsed_s, 2),
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
    }
    meta_path = args.out_dir / f"{args.model_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()

    print(f"saved model: {final_path}.zip")
    print(f"saved latest: {latest_path}.zip")
    print(f"saved meta: {meta_path}")
    print(f"wall_clock_seconds={elapsed_s:.1f}")


if __name__ == "__main__":
    main()
