# CelebA32 Small-DiT diagonal runs

This suite trains the three diagonal prediction/loss cells on the same saved
CelebA32-gray subset:

| Task | Prediction | Loss |
| ---: | --- | --- |
| 0 | `x` | `x` |
| 1 | `eps` | `eps` |
| 2 | `v` | `v` |

All tasks use seed 0, `N_train=256`, 100,000 optimizer updates, Small DiT-S/2
(depth 12, width 384, 6 heads, patch size 2), physical batch 64, and effective
batch 256 through four accumulated microbatches. Sampling uses safe division,
50-step full Heun, and `sampling.eps_to_v="via_x"`. The conversion setting is
shared by all tasks but is exercised only when an epsilon prediction must be
converted to velocity.

Submit all three tasks:

```bash
bash jobs/berzelius/celeba32_dit_diagonal_100k_viax/submit.sh
```

Preview without writing or submitting:

```bash
bash jobs/berzelius/celeba32_dit_diagonal_100k_viax/submit.sh --dry-run
```

Resume selected tasks from exact rotating recovery state, for example tasks 0
and 2:

```bash
bash jobs/berzelius/celeba32_dit_diagonal_100k_viax/resume.sh \
  outputs/job_sets/celeba32_dit_diagonal_100k_viax/BATCH_DIRECTORY 0,2
```

The submit wrapper freezes the config, source tree, variant table, and job
scripts under the batch directory before calling `sbatch`. Training monitoring
produces fixed-noise grids and projected-pixel memorization diagnostics at the
configured checkpoints; this suite does not pretend those diagnostics are FID
or feature-space memorization evaluation.
