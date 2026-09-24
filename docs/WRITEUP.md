# Writeup

Ontario Tech — Code & Sorcery Lab graduate coding challenge  
Applicant: **Khurshid Khan Ahmadzai**  
Project: Custom Mario-style platformer + PPO (Level 1)  

---

## 1. MDP formulation

### Observation
Hand-crafted **10-D `float32` feature vector** (not pixels):

| Index | Feature |
|------:|---------|
| 0–1 | player \(x, y\) normalized by map size |
| 2–3 | player \(v_x, v_y\) normalized by max speeds |
| 4 | `on_ground` |
| 5–6 | goal relative \(dx, dy\) |
| 7–8 | nearest enemy relative \(dx, dy\) (0 if none) |
| 9 | ground-ahead (right) |

**Rationale:** sample efficiency on CPU, interpretability, and a clean split
between engine and learning code.

### Actions
Discrete: `{NOOP, LEFT, RIGHT, JUMP, RIGHT+JUMP}`.  
Jump is only applied when the player is on the ground.

### Reward
Progress-shaped (`src/mario_rl/env/rewards.py`):

- \(+0.12 \times \Delta x\)
- \(-0.01\) per step
- \(+10\) on goal
- \(-1.5\) on death

### Episode end
Terminated on death or goal; truncated at `max_episode_steps` (2000).

### Determinism
Same seed + same actions ⇒ same trajectory (pytest-covered).

---

## 2. Method

**Game.** Custom headless engine (`src/mario_rl/game/`) with Level 1 tile map,
gravity, accelerated movement, AABB collisions, patrol enemy, on-path spikes,
pits, and goal. Optional pygame play is debug-only; training stays headless.

**Environment.** `MarioEnv` (Gymnasium) builds features and shaped rewards.

**Agents.** Random baseline vs Stable-Baselines3 `PPO` (`MlpPolicy`), trained by
`scripts/train.py` using `configs/default.yaml` (200k timesteps, seed 42, CPU).

**Evaluation.** `scripts/evaluate.py --compare` (30 episodes each); curves from
Monitor logs; GIF via `scripts/record_demo.py`.

---

## 3. Results (headline)

Level 1, 30 episodes, seed 42 
(`results/metrics/comparison_latest.md`):

| Metric | Random | PPO | PPO better? |
|--------|-------:|----:|:-----------:|
| Mean return | 15.19 | **52.77** | yes |
| Mean x progress | 138.53 | **367.49** | yes |
| Success rate | 0.10 | **1.00** | yes |
| Death rate | 0.90 | **0.00** | yes |

PPO clearly beats random and completes the level under this protocol.
Early Core runs (before Level 1 playability polish) only improved distance;
the final layout + retrain produced reliable goal reaches.

Artifacts: `results/plots/`, `results/demos/best_agent.gif`,
`models/ppo_mario_level1_latest.zip`.

---

## 4. Limitations and next steps

- Still one level; no held-out generalization study yet  
- Feature vector could add an explicit spike-distance cue  
- Results are for a fixed seed protocol; multi-seed reporting would strengthen claims  
- Stretch (official list): reward ablation; second / held-out level  

---

## 5. Disclosure

See `docs/DISCLOSURE.md` for libraries, references, and AI-assistance honesty.
