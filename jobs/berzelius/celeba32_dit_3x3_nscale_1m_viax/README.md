# CelebA32 Small-DiT production matrix

This suite trains one seed of the complete prediction/loss 3x3 at four nested
CelebA32-gray training sizes. The 36 rows are the Cartesian product of
`N={512,1024,2048,4096}`, seed 0, prediction `{x,eps,v}`, and loss `{x,eps,v}`.

Every run uses Small DiT-S/2, unit loss weighting, Adam at 1e-4, physical batch
64, effective batch 256, and a scientific target of one million optimizer
updates. Training conversions use exact algebra. Checkpoint sampling uses safe
division with denominator floor 0.05, epsilon-to-x-to-velocity conversion,
50-step full Heun, no final Euler step, and no final pixel clipping.

The initial array is deliberately unthrottled under the repository rule for
arrays of at most 80 tasks. Each task requests one thin A100 for 24 hours. The
completed N=256 pilot suggests about four allocations per task and about 3,100
A100-hours for the complete matrix. This is an estimate, not a scheduler limit.

Submit all rows:

```bash
bash jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/submit.sh
```

Preview without writing or submitting:

```bash
bash jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/submit.sh --dry-run
```

Five minutes before wall-time, Slurm sends `SIGUSR1`. The trainer completes its
current optimizer update, saves exact recovery state, closes the attempt, and
exits with code 75. Slurm can therefore show an expected interrupted allocation
as failed. Inspect `run.json`, `recovery/index.json`, stdout, and `sacct` before
choosing tasks to resume.

Resume selected tasks manually to the one-million target:

```bash
bash jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/resume.sh \
  outputs/job_sets/celeba32_dit_3x3_nscale_1m_viax/BATCH_DIRECTORY \
  0,3,7-12
```

The resume worker uses the same frozen source, data, config, run directory,
recovery state, and W&B run ID. Each allocation gets unique Slurm and command
logs. W&B is strict so a tracking failure stops that allocation instead of
silently changing identity. Exact local recovery remains available for manual
resume. The local config, data bundle, data manifest, histories, checkpoints,
recovery states, generated tensors, metrics, grids, and Slurm logs are the
authoritative record.

The online 128-sample projected-pixel monitor is a trajectory diagnostic. It is
not fixed-10k FID, full-pixel f_mem, or learned-feature f_mem. Those evaluations
must read the saved artifacts after training.
