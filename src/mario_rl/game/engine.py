"""Headless game loop: reset + step the platformer world."""

from __future__ import annotations

from pathlib import Path

from mario_rl.game.entities import Player
from mario_rl.game.level import Level, load_level
from mario_rl.game.physics import step_physics


class GameEngine:

    def __init__(self, level_path: str | Path):
        self.level_path = Path(level_path)
        self.level: Level = load_level(self.level_path)
        self.player = Player(0.0, 0.0)
        self.steps = 0
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> Player:
        
        _ = seed  # determinism for enemies/etc. comes in a later feature
        self.level = load_level(self.level_path)
        sx, sy = self.level.player_start
        self.player = Player(x=sx, y=sy, vx=0.0, vy=0.0, on_ground=False)
        # Settle onto the floor with a few gravity ticks (no player input)
        for _ in range(8):
            step_physics(self.player, self.level, action=0)
        self.steps = 0
        return self.player

    def step(self, action: int) -> Player:
        """Advance one frame with the given discrete action."""
        step_physics(self.player, self.level, action=action)
        self.steps += 1
        return self.player

    def ascii_snapshot(self, mark: str = "@") -> str:
        """Render the tile map with the live player marked (start 'P' shown as empty)."""
        rows = [list(r) for r in self.level.raw_rows]
        for row in rows:
            for i, ch in enumerate(row):
                if ch == "P":
                    row[i] = "."
        cx, cy = self.player.center()
        from mario_rl.game.constants import TILE_SIZE

        tx = int(cx // TILE_SIZE)
        ty = int(cy // TILE_SIZE)
        if 0 <= ty < len(rows) and 0 <= tx < len(rows[ty]):
            rows[ty][tx] = mark
        return "\n".join("".join(r) for r in rows)
