# Mario-style Platformer + PPO

Ontario Tech — Code & Sorcery Lab  
Graduate Research Assistant Code Challenge  
Applicant: **Khurshid Khan Ahmadzai**

![Best PPO agent demo](results/demos/best_agent.gif)

## Overview

This repository implements a custom Super Mario Bros–style platformer, wraps it as a reinforcement learning environment, and trains a PPO agent to play it. The goal is a clean, reproducible research engineering submission: game engine, environment API, baseline comparison, training pipeline, and writeup.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Repository layout

- `src/mario_rl/` — game engine, environment wrapper, agents, utilities
- `scripts/` — train, evaluate, plot, and demo entry points
- `levels/` — level maps
- `configs/` — experiment configuration
- `tests/` — unit and integration tests
- `results/` — plots, metrics, and demo artifacts
- `models/` — saved policy weights
- `docs/` — design notes, architecture figure, writeup, disclosure

Quick checks:

```bash
python scripts/play_ascii.py
pytest tests/ -v
```

Optional keyboard play (compact camera window — does **not** affect headless training):

```bash
python scripts/play_human.py
```

Controls: Arrow keys / WASD to move, Space / Up to jump, `R` reset, `Esc` quit.  
Tip: **run right, then jump at the edge** to clear pits and the red spike gap.  
Do not walk into red spikes — jump over that gap like a pit.

## Train / evaluate / demo

```bash
# train (CPU OK, headless — no window)
python scripts/train.py

# random vs PPO comparison table + bar chart
python scripts/evaluate.py --compare --episodes 30 --seed 42

# learning curves from Monitor logs
python scripts/plot_learning_curves.py

# record GIF
python scripts/record_demo.py --agent ppo --model models/ppo_mario_level1_latest.zip
```

## Headline results

Level 1, 30 episodes, seed 42, PPO 200k timesteps:

| Metric | Random | PPO |
|--------|-------:|----:|
| Mean return | 9.21 | **13.29** |
| Mean x progress | 93.49 | **126.80** |
| Success rate | 0.00 | 0.00 |

PPO beats random on return and distance; still fails the first pit.

Artifacts: `results/metrics/comparison_latest.md`, `results/plots/`, `results/demos/best_agent.gif`.

## Hardware

- Windows 10, **CPU-only** PyTorch (`torch 2.14.0+cpu`)

## License

MIT
