# Random vs PPO (Level 1)

- episodes: 30
- seed: 42
- model: `D:/Scholarships/OntarioTech_Mario_PPO_Challenge/models/ppo_mario_level1_latest.zip`

| Metric | Random | PPO | Delta (PPO-Random) | PPO better? |
|--------|-------:|----:|------------------:|:-----------:|
| mean_return | 15.1860 | 52.7688 | 37.5828 | yes |
| mean_x_progress | 138.5303 | 367.4900 | 228.9597 | yes |
| success_rate | 0.1000 | 1.0000 | 0.9000 | yes |
| death_rate | 0.9000 | 0.0000 | -0.9000 | yes |
| mean_steps | 108.7667 | 133.0000 | 24.2333 | yes |
