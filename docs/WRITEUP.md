# Writeup

## 1. MDP formulation

**Actions:** discrete `{NOOP, LEFT, RIGHT, JUMP, RIGHT+JUMP}`.

**Observations:** 10-D feature vector (normalized position/velocity, on-ground flag,
relative goal and enemy offsets, ground-ahead indicator). See `docs/DESIGN.md`.

**Reward:** progress-shaped — forward movement reward, small per-step cost,
goal bonus, death penalty.

**Termination:** death or goal. **Truncation:** step limit.

## 2. Method

Custom game engine wrapped as a Gymnasium environment with feature observations
and a progress-shaped reward. The first baseline is a uniform random-action agent
(`src/mario_rl/agents/random_agent.py`), evaluated by `scripts/evaluate.py`.
PPO training will be added next.

## 3. Results

### Random baseline (Level 1, 30 episodes, seed 42)
| Metric | Value |
|--------|------:|
| Mean return | 7.84 |
| Mean x progress | 93.49 |
| Success rate | 0.00 |
| Death rate | 1.00 |

Source file: `results/metrics/random_baseline_latest.json`.
PPO comparison numbers will be added after training.

## 4. Limitations and next steps

Failure cases, remaining weaknesses, and directions for further work.

## 5. Disclosure

See `docs/DISCLOSURE.md`.
