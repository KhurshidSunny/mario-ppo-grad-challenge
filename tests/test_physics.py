"""Unit tests for gravity, ground contact, and wall blocking."""

from pathlib import Path

import pytest

from mario_rl.game.constants import (
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    GRAVITY,
    TILE_SIZE,
)
from mario_rl.game.engine import GameEngine
from mario_rl.game.entities import Player
from mario_rl.game.level import load_level
from mario_rl.game.physics import apply_gravity, step_physics

ROOT = Path(__file__).resolve().parents[1]
LEVEL_PATH = ROOT / "levels" / "level_1.txt"


@pytest.fixture
def engine() -> GameEngine:
    return GameEngine(LEVEL_PATH)


def test_level_loads_player_and_solids():
    level = load_level(LEVEL_PATH)
    assert level.width >= 10
    assert level.height >= 5
    assert level.player_start is not None
    assert len(level.solids) > 0
    assert level.goal_pos is not None


def test_gravity_increases_downward_velocity():
    player = Player(x=0.0, y=0.0, vx=0.0, vy=0.0)
    apply_gravity(player)
    assert player.vy == pytest.approx(GRAVITY)
    apply_gravity(player)
    assert player.vy == pytest.approx(2 * GRAVITY)


def test_player_stands_on_ground_after_reset(engine: GameEngine):
    player = engine.reset(seed=0)
    assert player.on_ground is True
    assert player.vy == pytest.approx(0.0)


def test_cannot_walk_through_left_wall(engine: GameEngine):
    player = engine.reset(seed=0)
    start_x = player.x
    # Hold left for many frames; should stop against the left wall tiles
    for _ in range(80):
        engine.step(ACTION_LEFT)
    assert engine.player.x >= TILE_SIZE - 1e-3
    assert engine.player.x <= start_x + 1e-3 or engine.player.vx == 0.0
    # Must remain inside the solid-border corridor
    assert engine.player.x >= TILE_SIZE - 1.0


def test_moving_right_increases_x(engine: GameEngine):
    player = engine.reset(seed=0)
    start_x = player.x
    # Short run so we stay on the starting platform (a gap exists further right)
    for _ in range(5):
        engine.step(ACTION_RIGHT)
    assert engine.player.x > start_x
    assert engine.player.on_ground is True


def test_jump_leaves_ground_then_can_land(engine: GameEngine):
    engine.reset(seed=0)
    assert engine.player.on_ground is True
    engine.step(ACTION_JUMP)
    assert engine.player.on_ground is False
    assert engine.player.vy < 0  # moving upward
    # Fall until grounded again
    landed = False
    for _ in range(60):
        engine.step(ACTION_NOOP)
        if engine.player.on_ground:
            landed = True
            break
    assert landed is True


def test_physics_step_is_deterministic(engine: GameEngine):
    engine.reset(seed=0)
    actions = [ACTION_RIGHT, ACTION_RIGHT, ACTION_JUMP, ACTION_NOOP, ACTION_LEFT] * 5
    path_a = []
    for a in actions:
        engine.step(a)
        p = engine.player
        path_a.append((round(p.x, 5), round(p.y, 5), round(p.vx, 5), round(p.vy, 5)))

    engine.reset(seed=0)
    path_b = []
    for a in actions:
        engine.step(a)
        p = engine.player
        path_b.append((round(p.x, 5), round(p.y, 5), round(p.vx, 5), round(p.vy, 5)))

    assert path_a == path_b


def test_step_physics_helper_matches_engine_action_noop(engine: GameEngine):
    engine.reset(seed=0)
    p = Player(x=engine.player.x, y=engine.player.y, vx=0.0, vy=0.0, on_ground=True)
    step_physics(p, engine.level, ACTION_NOOP)
    assert p.on_ground is True
