# CelebA32 one-million-step epsilon recovery sweep

This launcher runs the requested broad epsilon comparison on Berzelius. It uses
the clean image runner and the Bonnaire-style checkpoint schedule.

## Task matrix

| Task | Label | Prediction/loss | Epsilon conversion | Sampling floor | Final Euler | LR |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | eps_old_sampling | eps/eps | via_x | 0.05 | no | 1e-4 |
| 1 | eps_direct_full_heun | eps/eps | direct | 0.05 | no | 1e-4 |
| 2 | eps_direct_final_euler | eps/eps | direct | 0.05 | yes | 1e-4 |
| 3 | eps_exact_full_heun | eps/eps | direct | off | no | 1e-4 |
| 4 | eps_exact_final_euler | eps/eps | direct | off | yes | 1e-4 |
| 5 | eps_higher_lr | eps/eps | direct | 0.05 | yes | 1e-3 |
| 6 | vv_reference | v/v | unused | unused | yes | 1e-4 |
| 7 | xx_reference | x/x | unused | 0.05 | yes | 1e-4 |

Tasks 0 through 4 intentionally repeat the same seed-0 epsilon training while
changing sampling behavior. Task 5 changes the learning rate. Tasks 6 and 7 are
velocity and clean-data references.

## Shared settings

- CelebA grayscale at 32x32.
- N=256, batch size 256, 1,024 held-out images, seed 0.
- Bonnaire-close UNet with channels 32/64/128 and dropout 0.1.
- Adam, one million optimizer updates.
- Training and sampling interval [0.001, 0.999].
- The 72-point bonnaire-1m checkpoint schedule, ending at step 1,000,000.
- Heun with 50 sampling intervals and integrated-state output.
- Saved data, fixed noise, raw generated samples, grids, and projected f_mem.
- Online strict W&B logging in project celeba32-unet.

The launcher requests one GPU, four CPUs, 32 GB memory, and 48 hours per task.
The default array runs all eight tasks concurrently. The longer limit avoids
the earlier 30-hour timeout, where comparable jobs stopped near one million
steps. The clean runner does not currently resume training, so retain every
checkpoint and inspect failures before relaunching.

## Commands

Preview without submission:

```bash
bash jobs/berzelius/eps_recovery_1m/submit.sh --dry-run
```

Submit all eight tasks:

```bash
bash jobs/berzelius/eps_recovery_1m/submit.sh
```

Submit a selected subset:

```bash
bash jobs/berzelius/eps_recovery_1m/submit.sh 0-2%2
```

Optional environment overrides are MP_EPS_1M_ACCOUNT, MP_EPS_1M_TIME, and
MP_EPS_1M_DATA_ROOT. The data root must contain celeba/img_align_celeba.

Each submission creates a timestamped directory under
outputs/job_sets/eps_recovery_1m/. It copies the resolved task matrix and config,
records the source commit and source diff, and saves Slurm logs and task commands.
Workers import the live mem_param_src tree. Do not edit that source while these
jobs run.

After completion, inspect the saved numerical results with:

```bash
python jobs/berzelius/eps_recovery_1m/inspect_results.py \
  outputs/job_sets/eps_recovery_1m/BATCH_TAG
```

## Checkpoint evaluation

The separate [`eval/` workflow](eval/README.md) compares all eight existing
runs at steps 100,000, 390,000, and 695,000. Each run uses its as-launched
sampler. FID uses 10,000 generated images and one shared frozen 10,000-image
CelebA validation reference.

Preview the six epsilon tasks without submitting them:

```bash
bash jobs/berzelius/eps_recovery_1m/eval/submit.sh --dry-run
```

Evaluation reads the training folders and writes to `outputs/evals/`. After the
epsilon evaluation, `eval/submit_references.sh` adds the `v/v` and `x/x`
references to the same saved result set.
