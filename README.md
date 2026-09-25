# Mario-style Platformer + PPO

Ontario Tech — Code & Sorcery Lab  
Graduate Research Assistant Code Challenge  
Applicant: **Khurshid Khan Ahmadzai**  
Status: **Core complete**

![Best PPO agent demo](results/demos/best_agent.gif)

## Overview

This repository implements a **custom** Super Mario Bros–style platformer (not a
ROM / emulator wrapper), exposes it as a Gymnasium environment, trains a PPO
agent, and compares it to a random-action baseline. The submission emphasizes
clean engineering, a clear MDP, reproducible commands, and honest analysis.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the headline result

```bash
# 1) Train (CPU OK, headless — ~6–10 minutes for 200k steps)
python scripts/train.py --timesteps 200000 --seed 42 --reset-monitor

# 2) Compare random vs PPO (30 episodes, seed 42)
python scripts/evaluate.py --compare --episodes 30 --seed 42

# 3) Learning curves + demo GIF
python scripts/plot_learning_curves.py
python scripts/record_demo.py --agent ppo --model models/ppo_mario_level1_latest.zip
```

Quick checks:

```bash
pytest tests/ -q
python scripts/play_ascii.py
```

Optional keyboard play (does **not** affect headless training):

```bash
python scripts/play_human.py
```

Controls: Arrows / WASD move, Space / Up jump, `R` reset, `Esc` quit.  
Colors: yellow = Mario, brown = ground, red = spikes (jump over), pink = enemy, green = goal.

## Headline results (Level 1)

30 evaluation episodes, seed 42, PPO trained 200k timesteps on CPU:

| Metric | Random | PPO |
|--------|-------:|----:|
| Mean return | 15.19 | **52.77** |
| Mean x progress | 138.53 | **367.49** |
| Success rate | 0.10 | **1.00** |
| Death rate | 0.90 | **0.00** |

PPO beats random on every reported metric and reliably reaches the goal under
this evaluation protocol.

Artifacts: `results/metrics/comparison_latest.md`, `results/plots/`,
`results/demos/best_agent.gif`, `models/ppo_mario_level1_latest.zip`.

## Hardware

- Windows 10, **CPU-only** PyTorch (`torch 2.14.0+cpu`)
- full train (200k timesteps, seed 42): ≈ **7.2 minutes** wall-clock
  (`models/ppo_mario_level1_meta.json`)

## Repository layout

- `src/mario_rl/` — game, env, agents, utils
- `scripts/` — train / evaluate / plot / demo / optional human play
- `levels/` — Level 1 tile map
- `configs/default.yaml` — seeds and PPO hyperparameters
- `tests/` — physics, gameplay, env API, PPO smoke tests
- `docs/` — DESIGN, WRITEUP, DISCLOSURE

