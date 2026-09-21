"""Gameplay interactions: pits, hazards, enemies, and the goal."""

from __future__ import annotations

import math
from typing import Protocol

from mario_rl.game.constants import ENEMY_HEIGHT, ENEMY_SPEED, ENEMY_WIDTH, TILE_SIZE
from mario_rl.game.entities import Enemy, Goal, Player
from mario_rl.game.level import Level


class _Box(Protocol):
    left: float
    right: float
    top: float
    bottom: float


def aabb_overlap(a: _Box, b: _Box) -> bool:
    return a.left < b.right and a.right > b.left and a.top < b.bottom and a.bottom > b.top


def fell_into_pit(player: Player, level: Level) -> bool:
    """Death if the player falls below the bottom of the tile map."""
    _, height_px = level.pixel_bounds()
    return player.top > height_px


def touching_hazard(player: Player, level: Level) -> bool:
    eps = 1e-6
    tx0 = int(math.floor((player.left + eps) / TILE_SIZE))
    tx1 = int(math.floor((player.right - eps) / TILE_SIZE))
    ty0 = int(math.floor((player.top + eps) / TILE_SIZE))
    ty1 = int(math.floor((player.bottom - eps) / TILE_SIZE))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            if (tx, ty) in level.hazard_tiles:
                return True
    return False


def touching_enemy(player: Player, enemies: list[Enemy]) -> bool:
    for enemy in enemies:
        if enemy.alive and aabb_overlap(player, enemy):
            return True
    return False


def reached_goal(player: Player, goal: Goal | None) -> bool:
    if goal is None:
        return False
    return aabb_overlap(player, goal)


def step_enemies(enemies: list[Enemy], level: Level) -> None:
    for enemy in enemies:
        if not enemy.alive:
            continue
        enemy.x += enemy.vx
        if _blocked_horizontally(enemy, level) or _ledge_ahead(enemy, level):
            enemy.vx = -enemy.vx
            enemy.x += enemy.vx


def _blocked_horizontally(enemy: Enemy, level: Level) -> bool:
    probe_x = enemy.right + 0.5 if enemy.vx > 0 else enemy.left - 0.5
    sample_y = enemy.y + enemy.height * 0.5
    tx = int(math.floor(probe_x / TILE_SIZE))
    ty = int(math.floor(sample_y / TILE_SIZE))
    if tx < 0 or tx >= level.width or ty < 0 or ty >= level.height:
        return True
    return level.is_solid_tile(tx, ty)


def _ledge_ahead(enemy: Enemy, level: Level) -> bool:
    """Turn around when the next footstep would hang over empty space."""
    foot_x = enemy.right - 1.0 if enemy.vx > 0 else enemy.left + 1.0
    foot_y = enemy.bottom + 0.5
    tx = int(math.floor(foot_x / TILE_SIZE))
    ty = int(math.floor(foot_y / TILE_SIZE))
    if tx < 0 or tx >= level.width or ty < 0 or ty >= level.height:
        return True
    return not level.is_solid_tile(tx, ty)


def make_enemies(level: Level, seed: int) -> list[Enemy]:
    direction = ENEMY_SPEED if seed % 2 == 0 else -ENEMY_SPEED
    enemies: list[Enemy] = []
    for cx, cy in level.enemy_spawns:
        enemies.append(
            Enemy(
                x=cx - ENEMY_WIDTH / 2.0,
                y=cy - ENEMY_HEIGHT / 2.0,
                vx=direction,
            )
        )
    return enemies


def make_goal(level: Level) -> Goal | None:
    if level.goal_pos is None:
        return None
    gx, gy = level.goal_pos
    return Goal(x=gx - 5.0, y=gy - 7.0)
