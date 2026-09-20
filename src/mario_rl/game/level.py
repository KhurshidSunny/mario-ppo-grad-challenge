"""Tile-map loading from plain-text level files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mario_rl.game.constants import PLAYER_HEIGHT, PLAYER_WIDTH, TILE_SIZE

SOLID_CHARS = {"#"}
PLAYER_CHAR = "P"
GOAL_CHAR = "G"
ENEMY_CHAR = "E"
HAZARD_CHAR = "^"
EMPTY_CHARS = {".", " "}


@dataclass
class Level:
    """Axis-aligned tile map. Row 0 is the top of the world."""

    width: int
    height: int
    solids: set[tuple[int, int]]  # (tile_x, tile_y)
    player_start: tuple[float, float]  # pixel top-left of player spawn
    goal_pos: tuple[float, float] | None = None
    enemy_spawns: list[tuple[float, float]] = field(default_factory=list)
    hazard_tiles: set[tuple[int, int]] = field(default_factory=set)
    raw_rows: list[str] = field(default_factory=list)

    def is_solid_tile(self, tx: int, ty: int) -> bool:
        return (tx, ty) in self.solids

    def pixel_bounds(self) -> tuple[float, float]:
        """Return (width_px, height_px)."""
        return self.width * TILE_SIZE, self.height * TILE_SIZE


def load_level(path: str | Path) -> Level:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    map_rows: list[str] = []

    for line in text.splitlines():
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        # Legend / header comments: "# Level" or "#   # = solid"
        if stripped.lstrip().startswith("#") and not _looks_like_map_row(stripped):
            continue
        if _looks_like_map_row(stripped):
            map_rows.append(stripped)

    if not map_rows:
        raise ValueError(f"No map rows found in {path}")

    width = max(len(r) for r in map_rows)
    # Pad short rows
    map_rows = [r.ljust(width, ".") for r in map_rows]
    height = len(map_rows)

    solids: set[tuple[int, int]] = set()
    hazards: set[tuple[int, int]] = set()
    enemies: list[tuple[float, float]] = []
    player_start: tuple[float, float] | None = None
    goal_pos: tuple[float, float] | None = None

    for ty, row in enumerate(map_rows):
        for tx, ch in enumerate(row):
            if ch in SOLID_CHARS:
                solids.add((tx, ty))
            elif ch == HAZARD_CHAR:
                hazards.add((tx, ty))
            elif ch == PLAYER_CHAR:
                # Spawn slightly inset so feet sit on the tile below after settle
                px = tx * TILE_SIZE + (TILE_SIZE - PLAYER_WIDTH) / 2
                py = ty * TILE_SIZE + (TILE_SIZE - PLAYER_HEIGHT)
                player_start = (px, py)
            elif ch == GOAL_CHAR:
                goal_pos = (tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)
            elif ch == ENEMY_CHAR:
                enemies.append((tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2))
            elif ch in EMPTY_CHARS:
                continue
            else:
                # Unknown characters treated as empty air
                continue

    if player_start is None:
        raise ValueError(f"Level {path} is missing player start 'P'")

    return Level(
        width=width,
        height=height,
        solids=solids,
        player_start=player_start,
        goal_pos=goal_pos,
        enemy_spawns=enemies,
        hazard_tiles=hazards,
        raw_rows=map_rows,
    )


def _looks_like_map_row(line: str) -> bool:
   
    allowed = set("#.PGE^ ")
    if len(line) < 4:
        return False
    return all(ch in allowed for ch in line)
