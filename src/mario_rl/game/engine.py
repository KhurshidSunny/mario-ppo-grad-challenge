"""Headless game loop: reset + step the platformer world."""

from __future__ import annotations

from pathlib import Path

from mario_rl.game.constants import TILE_SIZE
from mario_rl.game.entities import Enemy, Goal, Player
from mario_rl.game.interactions import (
    fell_into_pit,
    make_enemies,
    make_goal,
    reached_goal,
    step_enemies,
    touching_enemy,
    touching_hazard,
)
from mario_rl.game.level import Level, load_level
from mario_rl.game.physics import step_physics


class GameEngine:
    """Simulates one level: physics, patrol enemies, hazards, pits, and goal."""

    def __init__(self, level_path: str | Path):
        self.level_path = Path(level_path)
        self.level: Level = load_level(self.level_path)
        self.player = Player(0.0, 0.0)
        self.enemies: list[Enemy] = []
        self.goal: Goal | None = None
        self.steps = 0
        self.seed = 0
        self.alive = True
        self.reached_goal_flag = False
        self.death_cause: str | None = None
        self.reset(seed=0)

    def reset(self, seed: int | None = None) -> Player:
        """Restore a clean initial state. Same seed => same enemy start direction."""
        self.seed = 0 if seed is None else int(seed)
        self.level = load_level(self.level_path)
        sx, sy = self.level.player_start
        self.player = Player(x=sx, y=sy, vx=0.0, vy=0.0, on_ground=False)
        self.enemies = make_enemies(self.level, self.seed)
        self.goal = make_goal(self.level)
        self.alive = True
        self.reached_goal_flag = False
        self.death_cause = None
        self.steps = 0

        # Settle onto the floor before gameplay starts.
        for _ in range(8):
            step_physics(self.player, self.level, action=0)
        return self.player

    def step(self, action: int) -> Player:
        """Advance one frame. After death or goal, further steps are no-ops."""
        if not self.alive or self.reached_goal_flag:
            return self.player

        step_physics(self.player, self.level, action=action)
        step_enemies(self.enemies, self.level)
        self.steps += 1
        self._resolve_outcomes()
        return self.player

    def _resolve_outcomes(self) -> None:
        if fell_into_pit(self.player, self.level):
            self.alive = False
            self.death_cause = "pit"
            return
        if touching_hazard(self.player, self.level):
            self.alive = False
            self.death_cause = "hazard"
            return
        if touching_enemy(self.player, self.enemies):
            self.alive = False
            self.death_cause = "enemy"
            return
        if reached_goal(self.player, self.goal):
            self.reached_goal_flag = True

    def snapshot_state(self) -> tuple:
        """Hashable-ish state for determinism checks."""
        enemy_state = tuple(
            (round(e.x, 5), round(e.y, 5), round(e.vx, 5), e.alive) for e in self.enemies
        )
        p = self.player
        return (
            round(p.x, 5),
            round(p.y, 5),
            round(p.vx, 5),
            round(p.vy, 5),
            p.on_ground,
            self.alive,
            self.reached_goal_flag,
            self.death_cause,
            enemy_state,
        )

    def ascii_snapshot(self, mark: str = "@") -> str:
        """Render the tile map with live player/enemy marks."""
        rows = [list(r) for r in self.level.raw_rows]
        for row in rows:
            for i, ch in enumerate(row):
                if ch == "P":
                    row[i] = "."

        def paint(px: float, py: float, ch: str) -> None:
            tx = int(px // TILE_SIZE)
            ty = int(py // TILE_SIZE)
            if 0 <= ty < len(rows) and 0 <= tx < len(rows[ty]):
                rows[ty][tx] = ch

        for enemy in self.enemies:
            if enemy.alive:
                cx, cy = enemy.center()
                paint(cx, cy, "E")

        cx, cy = self.player.center()
        paint(cx, cy, mark)
        return "\n".join("".join(r) for r in rows)
