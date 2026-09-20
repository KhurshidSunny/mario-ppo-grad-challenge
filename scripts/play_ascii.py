"""Print a few static ASCII frames to sanity-check player physics.

Usage:
  python scripts/play_ascii.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mario_rl.game.constants import ACTION_JUMP, ACTION_LEFT, ACTION_NOOP, ACTION_RIGHT
from mario_rl.game.engine import GameEngine


def main() -> None:
    level = ROOT / "levels" / "level_1.txt"
    engine = GameEngine(level)
    engine.reset(seed=0)

    checks = [
        ("1) After reset - standing on ground", []),
        ("2) After RIGHT x5 - moved right on platform", [ACTION_RIGHT] * 5),
        ("3) After JUMP - in the air", [ACTION_JUMP]),
        ("4) After NOOP x25 - fell and landed again", [ACTION_NOOP] * 25),
        ("5) After LEFT x40 - blocked by left wall", [ACTION_LEFT] * 40),
    ]

    for title, actions in checks:
        for action in actions:
            engine.step(action)
        p = engine.player
        print(title)
        print(
            f"   x={p.x:.2f}  y={p.y:.2f}  vx={p.vx:.2f}  vy={p.vy:.2f}  "
            f"on_ground={p.on_ground}"
        )
        print(engine.ascii_snapshot())
        print()


if __name__ == "__main__":
    main()
