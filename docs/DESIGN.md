# Design decisions

## Goal
Custom Mario-style platformer + Gymnasium environment + PPO agent that outperforms a random-action baseline.

## Observation
Hand-crafted feature vector for the core experiments (pixel observations reserved for optional follow-up comparison).

Planned features include player position and velocity, distance to goal, nearest enemy relative position, local ground/pit indicators, and an on-ground flag. Exact feature ordering will match `src/mario_rl/env/observations.py`.

## Action space (discrete)
- 0: NOOP
- 1: LEFT
- 2: RIGHT
- 3: JUMP
- 4: RIGHT+JUMP

## Reward
Progress-shaped default:
- positive signal for moving toward the goal
- small per-step time penalty
- success bonus on reaching the goal
- penalty on death

The implemented formula will live in `src/mario_rl/env/rewards.py` and be mirrored in the writeup.

## Episode end
- terminated: death or goal reached
- truncated: maximum step limit reached

## Determinism
The same seed and action sequence must produce the same trajectory.

## Rendering
- headless mode for training (`render_mode=None`)
- optional window or RGB frames for debugging and demos
