"""Gameplay outcomes: pits, hazards, enemies, goal, reset, determinism."""

from pathlib import Path

import pytest

from mario_rl.game.constants import (
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_RIGHT_JUMP,
    TILE_SIZE,
)
from mario_rl.game.engine import GameEngine

ROOT = Path(__file__).resolve().parents[1]
LEVEL_PATH = ROOT / "levels" / "level_1.txt"


@pytest.fixture
def engine() -> GameEngine:
    return GameEngine(LEVEL_PATH)


def test_reset_restores_clean_state(engine: GameEngine):
    engine.reset(seed=0)
    for _ in range(30):
        engine.step(ACTION_RIGHT)
    engine.reset(seed=0)
    assert engine.alive is True
    assert engine.reached_goal_flag is False
    assert engine.death_cause is None
    assert engine.steps == 0
    assert engine.player.on_ground is True
    assert len(engine.enemies) >= 1


def test_pit_death(engine: GameEngine):
    engine.reset(seed=0)
    # Walk off the starting platform into the gap.
    for _ in range(80):
        engine.step(ACTION_RIGHT)
        if not engine.alive:
            break
    assert engine.alive is False
    assert engine.death_cause == "pit"


def test_hazard_death(engine: GameEngine):
    engine.reset(seed=0)
    assert engine.level.hazard_tiles
    tx, ty = next(iter(engine.level.hazard_tiles))
    engine.player.x = tx * TILE_SIZE + 2
    engine.player.y = ty * TILE_SIZE + 2
    engine.player.vx = 0.0
    engine.player.vy = 0.0
    engine.step(ACTION_NOOP)
    assert engine.alive is False
    assert engine.death_cause == "hazard"


def test_enemy_contact_kills(engine: GameEngine):
    engine.reset(seed=0)
    enemy = engine.enemies[0]
    # Place the player on top of the enemy.
    engine.player.x = enemy.x
    engine.player.y = enemy.y
    engine.player.vx = 0.0
    engine.player.vy = 0.0
    engine.step(ACTION_NOOP)
    assert engine.alive is False
    assert engine.death_cause == "enemy"


def test_goal_success(engine: GameEngine):
    engine.reset(seed=0)
    assert engine.goal is not None
    engine.player.x = engine.goal.x
    engine.player.y = engine.goal.y
    engine.player.vx = 0.0
    engine.player.vy = 0.0
    engine.step(ACTION_NOOP)
    assert engine.reached_goal_flag is True
    assert engine.alive is True


def test_same_seed_same_trajectory(engine: GameEngine):
    actions = [ACTION_RIGHT, ACTION_RIGHT, ACTION_RIGHT_JUMP, ACTION_NOOP, ACTION_LEFT] * 8

    engine.reset(seed=7)
    path_a = [engine.snapshot_state()]
    for a in actions:
        engine.step(a)
        path_a.append(engine.snapshot_state())

    engine.reset(seed=7)
    path_b = [engine.snapshot_state()]
    for a in actions:
        engine.step(a)
        path_b.append(engine.snapshot_state())

    assert path_a == path_b


def test_different_seed_can_change_enemy_direction(engine: GameEngine):
    engine.reset(seed=0)
    vx0 = engine.enemies[0].vx
    engine.reset(seed=1)
    vx1 = engine.enemies[0].vx
    assert vx0 == -vx1


def test_headless_smoke_1000_steps(engine: GameEngine):
    engine.reset(seed=0)
    actions = [ACTION_NOOP, ACTION_LEFT, ACTION_RIGHT, ACTION_RIGHT_JUMP]
    for i in range(1000):
        engine.step(actions[i % len(actions)])
    # Survived the loop without crashing; episode may have ended earlier.
    assert engine.steps <= 1000
