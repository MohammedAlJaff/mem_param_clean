# Matched seed-1 and seed-2 training

This launcher trains the six missing trajectories for three-seed validation:

| Array task | Seed | Parameterization |
| ---: | ---: | --- |
| 0 | 1 | `eps/eps`, direct safe conversion, full Heun |
| 1 | 1 | `v/v` |
| 2 | 1 | `x/x` |
| 3 | 2 | `eps/eps`, direct safe conversion, full Heun |
| 4 | 2 | `v/v` |
| 5 | 2 | `x/x` |

All tasks reuse `configs/runners/celeba32_eps_recovery_1m.py`: one million
updates, N=256, batch cap 256, Adam at `1e-4`, unit weighting, the recovered
UNet, and the Bonnaire checkpoint schedule. The fixed post-hoc sampler is also
recorded in each config: 50-step full Heun, direct epsilon conversion, safe
floor 0.05, no final Euler, and `x_t_max` output.

Preview the exact unthrottled array command:

```bash
bash jobs/berzelius/eps_recovery_multiseed/submit.sh --dry-run
```

The researcher launches it with:

```bash
bash jobs/berzelius/eps_recovery_multiseed/submit.sh
```

The six tasks run concurrently. The assistant does not submit them. Every real
submission and later status check belongs in `RUN_JOURNAL.md`. After completion,
pass the resulting `runs/` directory to the feature evaluator's `--scope all`
command.
