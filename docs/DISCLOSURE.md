# Disclosure

This file records libraries, references, and any AI assistance used during the
project, as required by the challenge guidelines.

Applicant: Khurshid Khan Ahmadzai  
Project: Mario-style platformer + PPO (Ontario Tech Code & Sorcery Lab challenge)

## Libraries
- **numpy** — game math and observation vectors
- **gymnasium** — environment API (`reset` / `step`)
- **pyyaml** — experiment config files
- **pytest** — unit and API tests
- **tqdm**, **matplotlib**, **pandas** — evaluation and plotting
- **pillow** — GIF export for demos
- **pygame** — optional windowed keyboard play (`scripts/play_human.py`);
  training and evaluation remain headless and do not require a display
- **torch** — neural network backend for PPO (CPU build used for headline runs)
- **stable-baselines3** — PPO training, checkpointing, and evaluation helpers

## Tutorials and references
- Gymnasium environment creation docs: https://gymnasium.farama.org/
- Stable-Baselines3 PPO docs: https://stable-baselines3.readthedocs.io/

## AI assistance
AI coding assistants (ChatGPT) was used as **support tool** while
building and documenting this project: to clarify RL / game concepts, discuss
design options, draft boilerplate, and help debug.

