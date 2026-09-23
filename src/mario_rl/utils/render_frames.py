"""Simple RGB frame renderer for demos (no pygame window required)."""

from __future__ import annotations

import numpy as np

from mario_rl.game.constants import (
    ENEMY_HEIGHT,
    ENEMY_WIDTH,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
    TILE_SIZE,
)
from mario_rl.game.engine import GameEngine

# Distinct colors (RGB) — keep in sync with play HUD legend
COLOR_BG = (24, 32, 48)
COLOR_SOLID = (92, 64, 40)      # brown = ground / walls
COLOR_HAZARD = (200, 60, 50)    # red = spikes
COLOR_GOAL = (50, 190, 90)      # green = goal
COLOR_PLAYER = (240, 200, 60)   # yellow = Mario (you)
COLOR_ENEMY = (210, 90, 200)    # pink/purple = enemy
COLOR_AIR = (40, 55, 80)        # dark blue-gray = empty air


def render_rgb(engine: GameEngine, *, scale: int = 4) -> np.ndarray:
    """Return an HxWx3 uint8 image of the current engine state."""
    level = engine.level
    w_px = level.width * TILE_SIZE
    h_px = level.height * TILE_SIZE
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    canvas[:, :] = COLOR_AIR

    for ty in range(level.height):
        for tx in range(level.width):
            x0, y0 = tx * TILE_SIZE, ty * TILE_SIZE
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
            if level.is_solid_tile(tx, ty):
                canvas[y0:y1, x0:x1] = COLOR_SOLID
            elif (tx, ty) in level.hazard_tiles:
                # Spikes sit on the path: red block in the walkway cell (above brown floor)
                canvas[y0:y1, x0:x1] = COLOR_HAZARD
                # Pointy look: darker tip on the top half
                tip = y0 + TILE_SIZE // 2
                canvas[y0:tip, x0:x1] = (230, 90, 70)

    if engine.goal is not None:
        gx = int(engine.goal.x) - TILE_SIZE // 2
        gy = int(engine.goal.y) - TILE_SIZE // 2
        _blit_rect(canvas, gx, gy, TILE_SIZE, TILE_SIZE, COLOR_GOAL)

    for enemy in engine.enemies:
        if enemy.alive:
            _blit_rect(
                canvas,
                int(enemy.x),
                int(enemy.y),
                ENEMY_WIDTH,
                ENEMY_HEIGHT,
                COLOR_ENEMY,
            )

    p = engine.player
    _blit_rect(
        canvas,
        int(p.x),
        int(p.y),
        PLAYER_WIDTH,
        PLAYER_HEIGHT,
        COLOR_PLAYER,
    )

    if scale != 1:
        canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    return canvas


def _blit_rect(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    h_px, w_px = canvas.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_px, x + w)
    y1 = min(h_px, y + h)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = color
