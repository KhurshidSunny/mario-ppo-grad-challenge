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

Training and evaluation entry points live under `scripts/` once the training pipeline is added.

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

## Hardware

Training hardware and approximate wall-clock time will be recorded here after the first full training run.

## License

MIT
