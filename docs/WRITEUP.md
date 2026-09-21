# Writeup

## 1. MDP formulation

**Actions:** discrete `{NOOP, LEFT, RIGHT, JUMP, RIGHT+JUMP}`.

**Observations:** 10-D feature vector (normalized position/velocity, on-ground flag,
relative goal and enemy offsets, ground-ahead indicator). See `docs/DESIGN.md`.

**Reward:** progress-shaped — forward movement reward, small per-step cost,
goal bonus, death penalty.

**Termination:** death or goal. **Truncation:** step limit.

## 2. Method

Game engine design, environment wrapper, network architecture, PPO configuration, and training procedure.

## 3. Results

Learning curves, comparison against the random baseline, and any additional experiments.

## 4. Limitations and next steps

Failure cases, remaining weaknesses, and directions for further work.

## 5. Disclosure

See `docs/DISCLOSURE.md`.
