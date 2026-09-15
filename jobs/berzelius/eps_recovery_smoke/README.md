# CelebA32 epsilon recovery smoke comparisons

These scripts are prepared for the researcher to execute on Berzelius.
No jobs were submitted during preparation. No training, inference, W&B session,
or dataset loading was run. Validation was static only.

The current machinery is sufficient for a controlled smoke comparison.
Two thousand updates can expose numerical failure and early learning behavior.
They cannot establish recovered face convergence: the historical successful
run produced its best recorded FID around 25,000 updates.

## Comparison plan

Each array task runs the existing image training runner for 2,000 updates.
All settings are shared except the fields shown below.

| Task | Label | Prediction/loss | Conversion | Sampling floor | Final Euler | LR | Compare with |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | eps_old_sampling | eps/eps | via_x | 0.05 | no | 1e-4 | Historical sampling anchor |
| 1 | eps_direct_full_heun | eps/eps | direct | 0.05 | no | 1e-4 | 0: conversion only |
| 2 | eps_direct_final_euler | eps/eps | direct | 0.05 | yes | 1e-4 | 1: final step only |
| 3 | eps_exact_full_heun | eps/eps | direct | off | no | 1e-4 | 1: clamp only |
| 4 | eps_higher_lr | eps/eps | direct | 0.05 | yes | 1e-3 | 2: learning rate only |
| 5 | vv_reference | v/v | native velocity | unused for velocity | yes | 1e-4 | 2: reference without epsilon conversion |

The matrix is [variants.tsv](variants.tsv). The shared settings are in
[celeba32_eps_recovery_smoke.py](../../../configs/runners/celeba32_eps_recovery_smoke.py).
Task 5 is a representation/loss reference. Its loss magnitude is not directly
comparable with the epsilon losses.

Fixed settings:

- CelebA grayscale 32x32, aligned 140-pixel crop; train/valid source split.
- 256 training faces, batch size 256, 1,024 held-out reference images, seed 0.
- UNet channels 32/64/128, time embedding 128, two residual blocks, dropout 0.1,
  attention resolutions 16 and 8, as in the recovered historical config.
- Adam with zero weight decay. LR is 1e-4 except task 4.
- Training and sampling times from 0.001 to 0.999; 50 sampling intervals.
- Integrated-state output `x_t_max` for every task. There is no extra image
  readout after integration in this batch.
- FP32 tensors, CUDA TF32 enabled, no autocast.
- Checkpoints and sample panels at 0, 250, 500, ..., 2,000.
- 128 fixed-noise samples per checkpoint, all saved before display clipping.
  W&B panels show 32 fixed training faces and 64 generated faces.

This matches the historical architecture and principal training/sampling
settings. It does not reload the old training tensor, optimizer, or checkpoint.
The new jobs use the same current-loader subset selection with the same seed.
Verify the saved hashes before calling them matched.

Tasks 0 through 3 repeat the same training so each sampler gets its own W&B
image history through the existing runner. Their training curves should closely
agree. Sampling does not change the epsilon training loss. GPU bitwise equality
is not guaranteed. This costs more than replaying one checkpoint, but avoids
adding a new W&B evaluation runner for this short comparison.

## Files and execution

- [submit.sh](submit.sh) previews or submits an array. Default: all six tasks run simultaneously.
- [run_array.sbatch](run_array.sbatch) reads one matrix row, activates
  `mem_param_env`, and calls `mem_param_src.runners.image_train`.
- [inspect_results.py](inspect_results.py) reads saved results on CPU and prints
  raw numerical measurements and matching-input hashes. It writes no files.

From the current Berzelius workspace:

```bash
cd /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_clean

# Print the matrix and sbatch command. This submits nothing and creates no folders.
bash jobs/berzelius/eps_recovery_smoke/submit.sh --dry-run

# Submit all six jobs concurrently.
bash jobs/berzelius/eps_recovery_smoke/submit.sh
```

To submit only task 0:

```bash
bash jobs/berzelius/eps_recovery_smoke/submit.sh 0
```

Every submission gets a new batch directory and W&B group. Running the full
array after a single-task submission creates a fresh task 0 as well.

Each task requests one GPU, four CPUs, 32 GB RAM, and a two-hour limit.
The script uses account `berzelius-2026-78` from the existing project jobs.
Current allocation validity and job duration were not checked. Overrides:

```bash
MP_SMOKE_ACCOUNT=your_account MP_SMOKE_TIME=04:00:00 \
  bash jobs/berzelius/eps_recovery_smoke/submit.sh
```

The worker loads Mambaforge and activates `mem_param_env` automatically.
It uses the existing cache at
`/proj/rep-learning-robotics/users/x_moalj/project_mem_par/old_mem_par/data`.
To select another cache, set `MP_SMOKE_DATA_ROOT` to a directory containing
`celeba/img_align_celeba`. Downloads are disabled.

W&B uses the existing login, entity `WANDB_ENTITY` or the current config default
`maj90`, and project `celeba32-unet`. Every task name includes its matrix label.
Each submission shares one group named `eps-recovery-2k-<batch tag>`.
W&B strict mode is enabled: tracking failure stops the task rather than silently
continuing without the requested logs. Credentials were not inspected.

The submission copies the config and matrix into the batch folder. It records
the current Git commit and tracked source diff. Worker commands are saved under
`commands/`. Workers still import the live source tree; avoid editing that source
while this comparison batch runs.

## Where results appear

The submission prints the exact batch path:

```text
outputs/job_sets/eps_recovery_smoke/<batch tag>/
  smoke_config.py
  variants.tsv
  source_commit.txt
  source.diff
  commands/
  slurm/
  runs/
    0_eps_old_sampling_s0/
      config.json
      run.json
      data_bundle.pt
      data_manifest.json
      z_init.pt
      checkpoints/step_002000.pt
      train_history.json
      monitoring/comparison_grids/step_002000.png
      monitoring/generated_samples/step_002000.pt
```

The other tasks have the same layout. `run.json` records state and W&B URL.
W&B contains training loss, fixed image comparisons, and rough projected f_mem.
Raw ranges are not currently W&B metrics; inspect the saved tensors below.
No full FID evaluation or automatic post-training replay is launched.

After jobs finish, run this helper yourself with the printed batch path:

```bash
module load Mambaforge
mamba activate mem_param_env
python jobs/berzelius/eps_recovery_smoke/inspect_results.py \
  outputs/job_sets/eps_recovery_smoke/BATCH_TAG_FROM_SUBMISSION
```

It prints CSV rows with step, loss, finite status, min/max, RMS, fraction outside
[-1,1], and projected f_mem. Lines beginning with `#` report job state and hashes.
The helper recomputes the saved noise tensor hash and reports recorded training
and bundle hashes. It does not claim to verify every file's integrity.

## How to read the smoke results

1. Check which jobs reached step 2,000. Inspect errors for failed jobs.
2. Check data/noise hashes and the training curves for tasks 0 through 3.
   Different training curves require investigation before interpreting sampler effects.
3. Compare 0 with 1, then 1 with 2 and 3. Keep the checkpoint step fixed.
   Inspect both images and raw ranges. Early untrained samples can be poor.
4. Compare 2 with 4 to see the effect of the higher learning rate.
   Use task 5 as a separate velocity reference.
5. Look for sustained loss improvement, emerging image structure, and absence
   of explosive sample growth. A nonzero outside-range fraction is not by
   itself failure. Low loss, finite values, or f_mem=0 do not prove recovery.

If the new sampler looks better, confirm it by replaying the same saved weights
under both sampler settings. Then repeat the useful contrast with another seed.
Only after that should we extend a selected training recipe toward 25,000 updates.
Do not automatically launch a longer sweep based on these scripts.
The current training runner does not expose resume, so continuation is a separate
implementation decision; checkpoints here remain useful for sampling comparisons.
