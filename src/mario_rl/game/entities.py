"""Game entities used by the platformer engine."""

from __future__ import annotations

from dataclasses import dataclass

from mario_rl.game.constants import PLAYER_HEIGHT, PLAYER_WIDTH


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

    def copy_from(self, other: "Player") -> None:
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.on_ground = other.on_ground
        self.width = other.width
        self.height = other.height
