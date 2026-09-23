# Random vs PPO (Level 1)

- episodes: 30
- seed: 42
- model: `D:/Scholarships/OntarioTech_Mario_PPO_Challenge/models/ppo_mario_level1_latest.zip`

| Metric | Random | PPO | Delta (PPO-Random) | PPO better? |
|--------|-------:|----:|------------------:|:-----------:|
| mean_return | 9.2112 | 13.2860 | 4.0748 | yes |
| mean_x_progress | 93.4903 | 126.8000 | 33.3097 | yes |
| success_rate | 0.0000 | 0.0000 | 0.0000 | no |
| death_rate | 1.0000 | 1.0000 | 0.0000 | no |
| mean_steps | 50.7667 | 43.0000 | -7.7667 | no |
