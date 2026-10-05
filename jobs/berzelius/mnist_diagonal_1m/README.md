# Fresh MNIST diagonal production runs

This job set trains 36 new trajectories. It does not reuse any checkpoint,
data bundle, noise bank, recovery state, run directory, or W&B identity.

| Resolution | Network | N | Seeds | Targets | Runs |
| --- | --- | --- | --- | --- | ---: |
| MNIST32 | U-Net-S | 512, 2048 | 0, 1, 2 | `x/x`, `eps/eps`, `v/v` | 18 |
| MNIST64 | U-Net-L | 512, 2048 | 0, 1, 2 | `x/x`, `eps/eps`, `v/v` | 18 |

U-Net-S uses base width 32, channel multipliers `(1,2,4)`, attention at
16 and 8 pixels, and middle attention. U-Net-L uses base width 64, channel
multipliers `(1,2,4,4)`, and no attention. Both use two residual blocks,
dropout 0.1, and time-embedding dimension 128.

Every optimizer update uses one physical batch of 256 images. The effective
batch is also 256, so gradient accumulation is one. Adam uses learning rate
`1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, and no weight decay. Each run
targets one million updates.

The MNIST loader scales raw pixels to `[-1,1]` and resizes them with bilinear
interpolation. It does not augment or class-balance the data. The same seed
selects the same source IDs at both resolutions. The N=512 IDs are the first
512 IDs of the N=2048 subset.

Training conversions use exact division. Checkpoint monitoring uses 50-step
full Heun, safe division with denominator floor 0.05, epsilon-to-x-to-velocity
conversion, `x_t_max`, and no final pixel clamp. Both resolutions use the same
20 checkpoint steps so their saved trajectories align.

## Resources and staging

- MNIST32 uses 18 thin A100-40GB tasks with a 36-hour limit. Each task trains
  directly to one million updates.
- MNIST64 uses 18 fat A100-80GB tasks with a 30-hour limit per stage. Five
  matching `aftercorr` arrays stop at 200k, 400k, 600k, 800k, and one million.
- Five minutes before wall-time, Slurm sends `SIGUSR1`. The runner saves exact
  recovery state at the next optimizer boundary.

All runs use the dedicated W&B project `mnist-unet-diagonal-1m`. MNIST32 and
MNIST64 use separate batch-tagged groups. A trajectory keeps one W&B run ID
through every resume stage.

Preview validation and submission without writing:

```bash
bash jobs/berzelius/mnist_diagonal_1m/submit.sh --dry-run
```

Submit all 36 fresh trajectories:

```bash
bash jobs/berzelius/mnist_diagonal_1m/submit.sh
```

The submitter freezes source, config, job files, environment packages, and
SHA-256 hashes before calling Slurm. It records every accepted job in the
batch's `submission.tsv`. A dependent verifier checks all final checkpoints,
monitor artifacts, recovery states, W&B finishes, and paired source IDs.

Use the manual wrapper only after checking Slurm state and the newest recovery
index. This example resumes selected MNIST32 tasks to the absolute 1M target:

```bash
bash jobs/berzelius/mnist_diagonal_1m/resume.sh \
  outputs/job_sets/mnist_diagonal_1m/BATCH_DIRECTORY \
  0,3,7-9 1000000 thin
```

This workflow trains and monitors models only. It does not submit FID,
full-pixel, learned-feature, or reporting jobs.
