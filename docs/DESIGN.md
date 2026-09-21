# Design decisions

## Goal
Custom Mario-style platformer + Gymnasium environment + PPO agent that outperforms a random-action baseline.

## Architecture
High-level game structure:

![Game architecture](figures/game_architecture.png)

The engine takes the level map and a player action, runs physics and game rules on world objects, and updates the game state.

## Game rules (Level 1)
- Solid ground and walls from `#` tiles
- Gaps / pits: falling below the map ends the episode (death)
- One patrol enemy (`E`): touching it ends the episode (death)
- One hazard (`^` spikes): overlapping the tile ends the episode (death)
- Goal (`G`): reaching it ends the episode successfully
- `GameEngine.reset(seed)` restores a clean start; same seed + same actions replay the same trajectory

## Observation
Hand-crafted feature vector for the core experiments (pixel observations reserved for optional comparison).

Planned features include player position and velocity, distance to goal, nearest enemy relative position, local ground/pit indicators, and an on-ground flag. Exact feature ordering will match `src/mario_rl/env/observations.py`.

## Action space (discrete)
- 0: NOOP
- 1: LEFT
- 2: RIGHT
- 3: JUMP
- 4: RIGHT+JUMP

## Reward
Progress-shaped default (wired in the Gymnasium wrapper):
- positive signal for moving toward the goal
- small per-step time penalty
- success bonus on reaching the goal
- penalty on death

## Episode end
- terminated: death or goal reached
- truncated: maximum step limit reached

## Determinism
The same seed and action sequence must produce the same trajectory.

## Rendering
- headless mode for training (`render_mode=None`)
- ASCII snapshot script for quick physics/gameplay checks
- optional window or RGB frames for demos later
