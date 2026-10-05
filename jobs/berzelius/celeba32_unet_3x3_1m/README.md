# Fresh CelebA32 U-Net-S full 3x3 production runs

This workflow trains the complete prediction-space by loss-space grid at four
training-set sizes and three seeds:

| Dataset | Model | N | Seeds | Cells | Runs |
| --- | --- | --- | --- | --- | ---: |
| CelebA 32x32 grayscale | U-Net-S | 512, 1024, 2048, 4096 | 0, 1, 2 | full `x`, `eps`, `v` 3x3 | 108 |

Every trajectory is fresh and targets one million optimizer updates. U-Net-S
has base width 32, multipliers `(1,2,4)`, two residual blocks, dropout 0.1,
time embedding 128, attention at resolutions 16 and 8, middle attention on,
and 3,917,409 trainable parameters.

Training uses Adam at `1e-4`, PyTorch defaults betas `(0.9,0.999)` and epsilon
`1e-8`, zero weight decay, physical/effective batch 256/256, accumulation one,
deterministic FP32, and TF32 disabled. Training target conversions use exact
division. Monitoring uses safe division with floor 0.05, `eps_to_v=via_x`,
50-step full Heun, `[0.001,0.999]`, `x_t_max`, and no final clamp.

The data adapter preserves the historical center-140 crop, resizes to 32,
converts to grayscale, and normalizes to `[-1,1]`. Within a seed, the nine
target cells share source IDs and each smaller N is a prefix of N=4096.

## Submission

Validate and print the exact command without writing a batch:

```bash
bash jobs/berzelius/celeba32_unet_3x3_1m/submit.sh --dry-run
```

Submit the complete unthrottled array:

```bash
bash jobs/berzelius/celeba32_unet_3x3_1m/submit.sh
```

The initial allocation is array `0-107` with no concurrency cap. Every task
requests one thin A100-40GB, four CPUs, 32 GB, and 24 hours. Slurm sends USR1
five minutes before walltime. The trainer then saves exact state at an optimizer
boundary and exits 75; this is an expected resumable walltime boundary, not a
scientific failure. Automatic requeue is disabled.

The launcher freezes source, configuration, job files, environment packages,
and SHA-256 hashes under `outputs/job_sets/celeba32_unet_3x3_1m/`. W&B uses the
dedicated project `celeba32-unet-3x3-1m` and strict identity preservation.

## Exact resume

Audit Slurm and `run.json` first, then select only interrupted or paused rows:

```bash
bash jobs/berzelius/celeba32_unet_3x3_1m/resume.sh \
  outputs/job_sets/celeba32_unet_3x3_1m/BATCH_DIRECTORY \
  0-4,9,12-17
```

The resume launcher refuses running, errored, completed, or hash-invalid rows.
It loads the frozen source, newest exact recovery state, and original W&B ID.
The stop step defaults to the absolute one-million target.

## Completion

After all 108 rows finish, submit the strict artifact verifier:

```bash
bash jobs/berzelius/celeba32_unet_3x3_1m/verify.sh \
  outputs/job_sets/celeba32_unet_3x3_1m/BATCH_DIRECTORY
```

The verifier requires all 72 checkpoint/model/sample/grid/f_mem sets, valid
recovery hashes at one million, successful unique W&B identities, distinct
fresh bundle/init files, paired target-cell source IDs, and nested training IDs.
It writes `TRAINING_COMPLETE` only after the full contract passes.

This workflow launches training and online monitoring only. It does not launch
FID, learned-feature, full-pixel, or paper-figure evaluation jobs.
