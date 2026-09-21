"""Game entities used by the platformer engine."""

from __future__ import annotations

from dataclasses import dataclass

from mario_rl.game.constants import (
    ENEMY_HEIGHT,
    ENEMY_WIDTH,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
)


@dataclass
class Player:
    """Axis-aligned player body in pixel coordinates (y increases downward)."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    width: float = PLAYER_WIDTH
    height: float = PLAYER_HEIGHT

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass
class Enemy:
    """Simple left/right patrol enemy."""

    x: float
    y: float
    vx: float = 1.0
    width: float = ENEMY_WIDTH
    height: float = ENEMY_HEIGHT
    alive: bool = True

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0


@dataclass
class Goal:
    """Win target represented as a small axis-aligned box around the flag point."""

    x: float
    y: float
    width: float = 10.0
    height: float = 14.0

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height
