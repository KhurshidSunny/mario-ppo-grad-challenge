# Writeup

## 1. MDP formulation

**Actions:** discrete `{NOOP, LEFT, RIGHT, JUMP, RIGHT+JUMP}`.

**Observations:** 10-D feature vector (normalized position/velocity, on-ground flag,
relative goal and enemy offsets, ground-ahead indicator). See `docs/DESIGN.md`.

**Reward:** progress-shaped — forward movement reward (`0.12` scale), small per-step
cost, goal bonus (`+10`), death penalty (`-1.5`).

**Termination:** death or goal. **Truncation:** step limit.

## 2. Method

Custom game engine wrapped as a Gymnasium environment with feature observations
and a progress-shaped reward. Baselines and learners:

1. **Random agent** — uniform discrete actions (`src/mario_rl/agents/random_agent.py`)
2. **PPO agent** — Stable-Baselines3 `PPO` + `MlpPolicy`, trained by `scripts/train.py`,
   wrapped for evaluation by `src/mario_rl/agents/ppo_agent.py`

Hyperparameters live in `configs/default.yaml` (`ent_coef=0.02`). Training is
headless; demos are recorded as RGB GIFs via `scripts/record_demo.py`.

## 3. Results

Level 1, **30 episodes**, seed **42**, PPO trained **200k** timesteps (CPU, ~6.6 min).

| Metric | Random | PPO | PPO better? |
|--------|-------:|----:|:-----------:|
| Mean return | 9.21 | **13.29** | yes |
| Mean x progress | 93.49 | **126.80** | yes |
| Success rate | 0.00 | 0.00 | no |
| Death rate | 1.00 | 1.00 | no |

Sources:
- `results/metrics/comparison_latest.md`
- `results/metrics/comparison_latest.json`
- Learning curves: `results/plots/learning_curve_reward.png`
- Demo GIF: `results/demos/best_agent.gif`

PPO clearly beats random on **return** and **forward distance**, but still dies in the
first pit (no goal clears yet). That failure mode is the main next engineering target.

## 4. Limitations and next steps

- No graphical interactive window required for Core; optional `scripts/play_human.py` (pygame) is for local keyboard play only
- Agent does not clear pits / spikes / enemy reliably after 200k steps
- Stretch later: reward ablations, longer training, second level / generalization

## 5. Disclosure

See `docs/DISCLOSURE.md`.
