# CelebA32 native-v/v loss-weighting controls at N=2048

This extension trains nine fresh CelebA32 grayscale trajectories at N=2048.
Every network
predicts velocity and uses velocity-space MSE. The only scientific axis is the
raw scalar weight applied to that MSE.

| Variant | `framework.loss_weighting` | Weight | Seeds | Runs |
| --- | --- | --- | --- | ---: |
| Clean-data-equivalent | `x_equivalent` | `(1-t)^2` | 0, 1, 2 | 3 |
| Unweighted baseline | `unit` | `1` | 0, 1, 2 | 3 |
| Noise-equivalent | `eps_equivalent` | `t^2` | 0, 1, 2 | 3 |

The weights are not normalized. For the linear flow-matching path, the raw
weighted velocity losses equal the corresponding implied `x/x` or `eps/eps`
losses. These nine runs are dedicated controls. They do not reuse the `v/v`
trajectories from the full 3x3 block or the submitted N=512 weighting block.

All runs use N=2048 and U-Net-S with 3,917,409 parameters. Training uses Adam at
`1e-4`, zero weight decay, physical/effective batch 256/256, accumulation one,
deterministic FP32, TF32 disabled, exact training conversions, and one million
optimizer updates. The 72-point `bonnaire-1m` schedule controls checkpoints and
online monitoring.

Monitoring uses 50-step full Heun on `[0.001,0.999]`, safe division with floor
0.05, `eps_to_v=via_x`, `x_t_max`, no final Euler step, and no pixel clamp.
Within each seed, all three weights receive the same source IDs, initial model,
training order, random stream, and fixed monitoring noise. Files and W&B
identities remain separate and fresh. For every seed, the already-submitted
N=512 source selection is exactly the prefix of this N=2048 selection.

## Submission

Run the write-free validation first:

```bash
bash jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/submit.sh --dry-run
```

Expected summary:

```text
Validated nine fresh CelebA32 N=2048 weighted-v/v trajectories.
Validated raw weights: (1-t)^2, 1, and t^2; no normalization.
Validated each submitted N=512 source set is the N=2048 prefix.
```

Submit all nine rows:

```bash
bash jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/submit.sh
```

The submission freezes the source, config, job files, environment manifest,
and SHA-256 hashes. It then submits an unthrottled `0-8` array. Each task asks
for one thin A100-40GB, four CPUs, 32 GB RAM, and 24 hours. The direct target is
one million updates. Current CelebA32 throughput suggests about 35 GPU-hours
per trajectory and about 311 GPU-hours for the block.

W&B uses the dedicated project `celeba32-unet-vv-weighting-1m`. Tracking is
strict so an exact resume keeps one identity instead of silently creating a
replacement.

## Exact resume

Five minutes before wall time, Slurm sends `SIGUSR1`. The trainer saves exact
state at an optimizer boundary and exits 75. Audit the stopped rows before
submitting only those rows again:

```bash
bash jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/resume.sh \
  outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/BATCH_DIRECTORY \
  0-8
```

The resume worker loads the newest exact state, frozen source, original run
directory, and original W&B identity. It targets the absolute one-million-step
endpoint. Automatic requeue is disabled.

## Completion

After all nine trajectories finish, run:

```bash
bash jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/verify.sh \
  outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/BATCH_DIRECTORY
```

The verifier requires all checkpoints and monitoring artifacts, valid recovery
state at one million, successful unique W&B finishes, fresh files, and exact
within-seed pairing of source IDs, initial weights, and monitoring noise.
