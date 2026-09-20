"""Gravity, acceleration, and AABB vs tile collisions."""

from __future__ import annotations

import math

from mario_rl.game.constants import (
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_RIGHT_JUMP,
    AIR_FRICTION,
    GRAVITY,
    GROUND_FRICTION,
    JUMP_VELOCITY,
    MAX_FALL_SPEED,
    MAX_MOVE_SPEED,
    MOVE_ACCEL,
    TILE_SIZE,
)
from mario_rl.game.entities import Player
from mario_rl.game.level import Level


def apply_controls(player: Player, action: int) -> None:
    """Update velocities from a discrete action. Jump only works on the ground."""
    want_left = action == ACTION_LEFT
    want_right = action in (ACTION_RIGHT, ACTION_RIGHT_JUMP)
    want_jump = action in (ACTION_JUMP, ACTION_RIGHT_JUMP)

    if want_left:
        player.vx -= MOVE_ACCEL
    elif want_right:
        player.vx += MOVE_ACCEL
    else:
        friction = GROUND_FRICTION if player.on_ground else AIR_FRICTION
        if player.vx > 0:
            player.vx = max(0.0, player.vx - friction)
        elif player.vx < 0:
            player.vx = min(0.0, player.vx + friction)

    player.vx = max(-MAX_MOVE_SPEED, min(MAX_MOVE_SPEED, player.vx))

    if want_jump and player.on_ground:
        player.vy = JUMP_VELOCITY
        player.on_ground = False

    if action == ACTION_NOOP:
        # Friction already applied when not pressing left/right.
        pass


def apply_gravity(player: Player) -> None:
    player.vy = min(MAX_FALL_SPEED, player.vy + GRAVITY)


def integrate_and_collide(player: Player, level: Level) -> None:
    """Move the player with axis-separated collision against solid tiles."""
    # --- horizontal ---
    player.x += player.vx
    _resolve_axis(player, level, axis="x")

    # --- vertical ---
    player.y += player.vy
    _resolve_axis(player, level, axis="y")
    player.on_ground = _ground_beneath(player, level)


def step_physics(player: Player, level: Level, action: int) -> None:
    """One full physics tick: controls → gravity → integrate + collide."""
    apply_controls(player, action)
    apply_gravity(player)
    integrate_and_collide(player, level)


def _resolve_axis(player: Player, level: Level, axis: str) -> None:
    overlapping = _overlapping_tiles(player, level)
    for tx, ty in overlapping:
        if not level.is_solid_tile(tx, ty):
            continue
        tile_left = tx * TILE_SIZE
        tile_right = tile_left + TILE_SIZE
        tile_top = ty * TILE_SIZE
        tile_bottom = tile_top + TILE_SIZE

        if axis == "x":
            if player.vx > 0:
                player.x = tile_left - player.width
                player.vx = 0.0
            elif player.vx < 0:
                player.x = tile_right
                player.vx = 0.0
        else:
            if player.vy > 0:
                # Landing on top of a tile
                player.y = tile_top - player.height
                player.vy = 0.0
                player.on_ground = True
            elif player.vy < 0:
                # Hitting ceiling
                player.y = tile_bottom
                player.vy = 0.0


def _overlapping_tiles(player: Player, level: Level) -> list[tuple[int, int]]:
    """Return solid-candidate tile indices that the player AABB overlaps."""
    # Shrink epsilon so resting exactly on a tile edge does not re-hit it.
    eps = 1e-6
    left = player.left + eps
    right = player.right - eps
    top = player.top + eps
    bottom = player.bottom - eps

    tx0 = max(0, int(math.floor(left / TILE_SIZE)))
    tx1 = min(level.width - 1, int(math.floor(right / TILE_SIZE)))
    ty0 = max(0, int(math.floor(top / TILE_SIZE)))
    ty1 = min(level.height - 1, int(math.floor(bottom / TILE_SIZE)))

    tiles: list[tuple[int, int]] = []
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tiles.append((tx, ty))
    return tiles


def _ground_beneath(player: Player, level: Level) -> bool:
    """True if a solid tile sits immediately under the player's feet."""
    probe = 0.5
    feet_y = player.bottom + probe
    # Sample a few x points across the feet so corners still count as grounded
    samples = (
        player.left + 1.0,
        player.x + player.width / 2.0,
        player.right - 1.0,
    )
    ty = int(math.floor(feet_y / TILE_SIZE))
    if ty < 0 or ty >= level.height:
        return False
    for sx in samples:
        tx = int(math.floor(sx / TILE_SIZE))
        if 0 <= tx < level.width and level.is_solid_tile(tx, ty):
            return True
    return False
