"""Shared game constants (pixel units, one simulation step = one frame)."""

# Tile / body sizes
TILE_SIZE = 16
PLAYER_WIDTH = 12
PLAYER_HEIGHT = 14
ENEMY_WIDTH = 12
ENEMY_HEIGHT = 12

# Vertical motion (y increases downward)
GRAVITY = 0.45
MAX_FALL_SPEED = 8.0
JUMP_VELOCITY = -8.5  # negative = upward

# Horizontal motion
MOVE_ACCEL = 0.55
MAX_MOVE_SPEED = 3.2
GROUND_FRICTION = 0.35
AIR_FRICTION = 0.08
ENEMY_SPEED = 1.0

# Discrete actions (must match docs/DESIGN.md)
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_RIGHT_JUMP = 4
