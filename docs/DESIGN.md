# Design decisions (locked for Core)

## Goal
Custom Mario-style platformer + Gymnasium env + PPO agent that beats a random baseline.

## Observation (Core)
Hand-crafted feature vector (not pixels first).
Exact features will be listed here on Day 4 when `observations.py` is implemented.

## Action space (discrete)
- 0: NOOP
- 1: LEFT
- 2: RIGHT
- 3: JUMP
- 4: RIGHT+JUMP

## Reward (Core default)
Progress-shaped:
- reward for moving toward the goal
- small per-step time penalty
- bonus on reaching goal
- penalty on death

Exact formula will be finalized in `src/mario_rl/env/rewards.py` and copied here.

## Episode end
- terminated: death or goal reached
- truncated: max steps exceeded

## Determinism
Same seed + same action sequence => same trajectory.

## Rendering
- headless for training (`render_mode=None`)
- optional window / rgb frames for demos later
