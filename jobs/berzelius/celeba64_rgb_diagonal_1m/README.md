# Fresh CelebA64 RGB diagonal production runs

This job set trains 18 new trajectories. It does not reuse any prototype
checkpoint, data bundle, fixed-noise file, recovery state, run directory, or
W&B identity.

| Images | Network | N | Seeds | Targets | Runs |
| --- | --- | --- | --- | --- | ---: |
| CelebA64 RGB | Wide/deep attention U-Net | 512, 2048 | 0, 1, 2 | `x/x`, `eps/eps`, `v/v` | 18 |

The U-Net uses base width 64, channel multipliers `(1,2,4,4)`, two residual
blocks, dropout 0.1, time-embedding dimension 128, attention at 16 and 8
pixels, and middle attention. It has 24,386,627 parameters. This is the
successful CelebA64 RGB profile from the 400k capacity gate.

The CelebA loader center-crops to 140 pixels, resizes and center-crops to
64 pixels, converts to RGB, and normalizes to `[-1,1]`. It uses the training
split for optimization and 1,000 images from the validation split as held-out
references. It does not augment or class-balance the data. Each seed pairs
source identities and fixed monitoring noise across the three target cells.
The N=512 training IDs are the first 512 IDs of the N=2048 subset.

Every optimizer update uses one physical batch of 256 images. The effective
batch is also 256, so gradient accumulation is one. Adam uses learning rate
`1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, and no weight decay. Runtime is
deterministic FP32 with TF32 enabled. Each run targets one million optimizer
updates and saves the 72-point `bonnaire-1m` checkpoint schedule.

Training conversions use exact division. Checkpoint monitoring uses one
sampler: 50-step full Heun on `[0.001,0.999]`, safe division with denominator
floor 0.05, epsilon-to-x-to-velocity conversion, `x_t_max`, no final Euler
step, and no final pixel clamp. This removes the three diagnostic sampler
profiles used in the prototype gate.

## Resources and staging

- All 18 tasks request fat A100-80GB nodes, eight CPUs, and 128 GB host memory.
- Five matching `aftercorr` arrays stop at 200k, 400k, 600k, 800k, and one
  million updates. Each allocation requests 30 hours.
- Five minutes before wall time, Slurm sends `SIGUSR1`. The runner saves exact
  recovery state at the next optimizer boundary.
- The observed prototype needed about 20 hours per 200k updates. The expected
  total is about 1,800 A100-hours. The maximum requested allocation is 2,700
  A100-hours.
- The 72 checkpoints require about 127 GB across the matrix. Allow about
  140 GB for checkpoints and online monitoring artifacts together.

All runs use W&B project `celeba64-rgb-unet-diagonal-1m`. The frozen batch has
one group, and each trajectory keeps one W&B run ID through every exact resume.

## Validate and submit

Preview validation and submission without writing a batch or calling Slurm:

```bash
bash jobs/berzelius/celeba64_rgb_diagonal_1m/submit.sh --dry-run
```

Expected final preview lines include:

```text
Fresh CelebA64 RGB diagonal suite: 18 trajectories, no reuse.
All tasks: fat A100-80GB, five 200k stages, 30 hours each.
W&B project: celeba64-rgb-unet-diagonal-1m
```

Submit all 18 fresh trajectories and the five-stage resume chain:

```bash
bash jobs/berzelius/celeba64_rgb_diagonal_1m/submit.sh
```

The submitter freezes source, config, job files, environment packages, and
SHA-256 hashes before calling Slurm. It records each accepted job in the
batch's `submission.tsv`. The dependent verifier checks all final checkpoints,
monitor artifacts, recovery histories, W&B finishes, paired noise, paired
source IDs, and fresh artifact files.

Use the manual wrapper only after checking Slurm state and the newest recovery
index. This example resumes selected tasks to the absolute one-million target:

```bash
bash jobs/berzelius/celeba64_rgb_diagonal_1m/resume.sh \
  outputs/job_sets/celeba64_rgb_diagonal_1m/BATCH_DIRECTORY \
  0,4,9-11 1000000
```

This workflow trains and monitors models only. It does not submit FID,
full-pixel, learned-feature, or reporting jobs.
