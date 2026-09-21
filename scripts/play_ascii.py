"""Print a few static ASCII frames to sanity-check the level."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.game.constants import ACTION_JUMP, ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT
from mario_rl.game.engine import GameEngine


def _print(title: str, engine: GameEngine) -> None:
    p = engine.player
    status = "alive"
    if not engine.alive:
        status = f"dead ({engine.death_cause})"
    elif engine.reached_goal_flag:
        status = "goal"
    print(title)
    print(
        f"   x={p.x:.2f}  y={p.y:.2f}  on_ground={p.on_ground}  status={status}"
    )
    print(engine.ascii_snapshot())
    print()


def main() -> None:
    engine = GameEngine(ROOT / "levels" / "level_1.txt")
    engine.reset(seed=0)
    _print("1) After reset", engine)

    for _ in range(5):
        engine.step(ACTION_RIGHT)
    _print("2) After RIGHT x5", engine)

    engine.step(ACTION_JUMP)
    _print("3) After JUMP", engine)

    for _ in range(25):
        engine.step(ACTION_NOOP)
    _print("4) After NOOP x25", engine)

    for _ in range(40):
        engine.step(ACTION_LEFT)
    _print("5) After LEFT x40", engine)


if __name__ == "__main__":
    main()
