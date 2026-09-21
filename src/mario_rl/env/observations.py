"""Feature-vector observations for the Mario environment."""

from __future__ import annotations

import math

import numpy as np

from mario_rl.game.constants import MAX_FALL_SPEED, MAX_MOVE_SPEED, TILE_SIZE
from mario_rl.game.engine import GameEngine


OBS_DIM = 10


def build_observation(engine: GameEngine) -> np.ndarray:
    level = engine.level
    width_px, height_px = level.pixel_bounds()
    player = engine.player
    cx, cy = player.center()

    goal_dx = 0.0
    goal_dy = 0.0
    if engine.goal is not None:
        gx = engine.goal.x + engine.goal.width / 2.0
        gy = engine.goal.y + engine.goal.height / 2.0
        goal_dx = gx - cx
        goal_dy = gy - cy

    enemy_dx = 0.0
    enemy_dy = 0.0
    nearest = _nearest_alive_enemy(engine, cx, cy)
    if nearest is not None:
        ex, ey = nearest.center()
        enemy_dx = ex - cx
        enemy_dy = ey - cy

    obs = np.array(
        [
            cx / width_px,
            cy / height_px,
            player.vx / MAX_MOVE_SPEED,
            player.vy / MAX_FALL_SPEED,
            1.0 if player.on_ground else 0.0,
            goal_dx / width_px,
            goal_dy / height_px,
            enemy_dx / width_px,
            enemy_dy / height_px,
            1.0 if _ground_ahead(engine, cx, cy) else 0.0,
        ],
        dtype=np.float32,
    )
    return obs


def _nearest_alive_enemy(engine: GameEngine, cx: float, cy: float):
    best = None
    best_dist = math.inf
    for enemy in engine.enemies:
        if not enemy.alive:
            continue
        ex, ey = enemy.center()
        dist = (ex - cx) ** 2 + (ey - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = enemy
    return best


def _ground_ahead(engine: GameEngine, cx: float, cy: float) -> bool:
    """True if solid ground exists a short distance to the right of the player."""
    probe_x = cx + TILE_SIZE * 1.5
    probe_y = cy + TILE_SIZE
    tx = int(probe_x // TILE_SIZE)
    ty = int(probe_y // TILE_SIZE)
    if tx < 0 or ty < 0 or tx >= engine.level.width or ty >= engine.level.height:
        return False
    return engine.level.is_solid_tile(tx, ty)
