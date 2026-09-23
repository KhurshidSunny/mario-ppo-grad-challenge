# Mario-style Platformer + PPO

Ontario Tech — Code & Sorcery Lab  
Graduate Research Assistant Code Challenge  
Applicant: **Khurshid Khan Ahmadzai**


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
- `scripts/` — train, evaluate, and demo entry points
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

Random baseline evaluation:

```bash
python scripts/evaluate.py --agent random --episodes 30 --seed 42
```

PPO training (CPU is fine for Level 1):

```bash
# smoke / short run
python scripts/train.py --timesteps 20000

# full config run (see configs/default.yaml)
python scripts/train.py
```

Evaluate a trained PPO model:

```bash
python scripts/evaluate.py --agent ppo --model models/ppo_mario_level1_latest.zip --episodes 30 --seed 42
```

ASCII demo of a trained policy (terminal “visual” check):

```bash
python scripts/demo_ascii.py --agent ppo --model models/ppo_mario_level1_latest.zip
```

Metrics are written under `results/metrics/`. Models are saved under `models/`.

## Hardware

Default training uses the **CPU** PyTorch build (`torch 2.14.0+cpu`).

Recorded full run (`total_timesteps: 200000`, seed 42):
- wall-clock ≈ **9.3 minutes** on Windows 10
- details in `models/ppo_mario_level1_meta.json`

## License

MIT
