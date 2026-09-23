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
and a progress-shaped reward. Baselines and learners:

1. **Random agent** — uniform discrete actions (`src/mario_rl/agents/random_agent.py`)
2. **PPO agent** — Stable-Baselines3 `PPO` + `MlpPolicy`, trained by `scripts/train.py`,
   wrapped for evaluation by `src/mario_rl/agents/ppo_agent.py`

Hyperparameters live in `configs/default.yaml`. Training is headless (no GUI).

## 3. Results

### Random baseline (Level 1, 30 episodes, seed 42)
| Metric | Value |
|--------|------:|
| Mean return | 7.84 |
| Mean x progress | 93.49 |
| Success rate | 0.00 |
| Death rate | 1.00 |

Source file: `results/metrics/random_baseline_latest.json`.

### PPO (Level 1, 200k timesteps, 30 eval episodes, seed 42)
| Metric | Random | PPO |
|--------|-------:|----:|
| Mean return | 7.84 | **11.22** |
| Mean x progress | 93.49 | **126.46** |
| Success rate | 0.00 | 0.00 |
| Death rate | 1.00 | 1.00 |

Training: CPU PyTorch, ~9.3 minutes wall-clock (`models/ppo_mario_level1_meta.json`).
Eval source: `results/metrics/ppo_baseline_latest.json`.

PPO already beats random on return and forward progress after 200k steps, but still falls into the first pit (no goal clears yet). Longer training and/or reward shaping tweaks are natural next steps.

## 4. Limitations and next steps

- Level 1 only; pits / enemy / spikes remain hard for short training
- Full training curves and failure-case analysis after a longer run

## 5. Disclosure

See `docs/DISCLOSURE.md`.
