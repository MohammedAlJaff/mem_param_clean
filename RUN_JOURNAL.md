# Berzelius Run Journal

This file tracks every compute run launched from `mem_param_clean`, starting on
2026-09-14. Earlier launches remain in `../old_mem_par/CODING_JOURNAL.md`.

Use this journal for operational history. Use `MIGRATION_JOURNAL.md` for code
and research decisions. Use `TODO.txt` for work that has not started.

## Update rules

- Add a run immediately after the scheduler accepts it.
- Submit all tasks concurrently when an array has about 80 tasks or fewer. For
  larger arrays, agree on and record an explicit throttle before submission.
- Keep the active-runs table current.
- Append a dated check below. Do not erase earlier checks.
- Update after submission, material progress checks, state changes, cancellation,
  timeout, failure, resume, and completion.
- Check both Slurm and the application artifacts. Use `squeue`, `sacct`,
  `run.json`, the latest checkpoint, and stderr/stdout when available.
- Do not infer completion because a job disappeared from `squeue`. Confirm its
  final `sacct` state and saved artifacts.
- Record the scheduler reason for `CANCELLED`, `TIMEOUT`, `FAILED`, or preemption.
  Distinguish a Berzelius low-power cancellation from a wall-time termination.
- Record the job ID, array selection, resource limit, config or matrix, output
  directory, W&B group or links, latest step, and next action.

Status vocabulary: `PENDING`, `RUNNING`, `PARTIAL`, `COMPLETED`, `FAILED`,
`CANCELLED`, and `TIMEOUT`. An array is `PARTIAL` when its tasks have different
terminal states or only some tasks completed.

## Active runs

| Job | Study | Submitted | Tasks and limit | Current status | Last checked | Batch directory | Next check |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Closed runs

| Job | Study | Final status | Finished | Final artifacts | Batch directory |
| --- | --- | --- | --- | --- | --- |
| array `17523970`; combine `17523971` | CelebA32 v/v and x/x reference eval | COMPLETED, 2/2 tasks and combine | 2026-09-15 13:40 CEST | 6/6 reference rows; 24-row combined comparison | [`20260915T111553Z_3025872_three_point`](outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point) |
| `17523389`; array `17523390`; combine `17523391` | Corrected CelebA32 eps/eps FID-10k | COMPLETED: reference, 6/6 evaluator tasks, combine | 2026-09-15 13:23 CEST | 10k/10k protocol verified; 18/18 finite rows; CSV, plot, summary | [`20260915T111553Z_3025872_three_point`](outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point) |
| `17513629` | CelebA32 epsilon recovery, 1M steps | COMPLETED, 8/8 tasks | 2026-09-15 11:57 CEST | All runs state `trained`; all have step 1,000,000 | [`20260914T113123Z_2308577`](outputs/job_sets/eps_recovery_1m/20260914T113123Z_2308577) |
| `17522330`; combine `17522331` | CelebA32 eps/eps three-point 1k eval | FAILED, 6/6 evaluator tasks; combine CANCELLED | 2026-09-15 12:44 CEST | No metric rows; Mambaforge startup failure | [`20260915T104407Z_2899578_three_point`](outputs/evals/eps_recovery_1m/20260915T104407Z_2899578_three_point) |
| `17522337`; combine `17522338` | CelebA32 eps/eps three-point 1k eval | COMPLETED, 6/6 evaluator tasks and combine | 2026-09-15 12:47 CEST | 18/18 finite metric rows; combined CSV, plot, and summary | [`20260915T104512Z_2903270_three_point`](outputs/evals/eps_recovery_1m/20260915T104512Z_2903270_three_point) |
| `17522748`; combine `17522749` | CelebA32 eps/eps three-point 10k eval | COMPLETED, but FID protocol invalid | 2026-09-15 13:00 CEST | f_mem/loss valid; FID used only 1,024 references and is withdrawn | [`20260915T105557Z_2941333_three_point`](outputs/evals/eps_recovery_1m/20260915T105557Z_2941333_three_point) |

## Chronological checks

### 2026-09-14 13:31 CEST - Launch job 17513629

- Submitted the CelebA32 one-million-step epsilon recovery array under account
  `berzelius-2026-78`.
- Array: `0-7%4`. Each task requests one GPU, four CPUs, 32 GB, and 48 hours.
- Matrix: six `eps/eps` variants, one `v/v` reference, and one `x/x` reference.
- Training: N=256, batch=256, seed 0, one million optimizer updates, and 72
  `bonnaire-1m` checkpoints.
- Config: [`celeba32_eps_recovery_1m.py`](configs/runners/celeba32_eps_recovery_1m.py).
- Matrix: [`variants.tsv`](jobs/berzelius/eps_recovery_1m/variants.tsv).
- Batch snapshot:
  [`20260914T113123Z_2308577`](outputs/job_sets/eps_recovery_1m/20260914T113123Z_2308577).
- W&B project/group: `celeba32-unet` /
  `eps-recovery-1m-20260914T113123Z_2308577`.

### 2026-09-14 13:34 CEST - Startup check for job 17513629

- Slurm: tasks 0-3 `RUNNING`; tasks 4-7 `PENDING (JobArrayTaskLimit)`.
- Application: tasks 0-3 loaded 256 training and 1,024 held-out images, built
  the 3,917,409-parameter UNet, initialized W&B, and entered training.
- Latest logged steps: task 0 at 1,475; task 1 at 1,475; task 2 at 1,500; task 3
  at 1,200. No startup error appeared in stderr.
- Active W&B runs: [task 0](https://wandb.ai/maj90/celeba32-unet/runs/r8w6kfkj),
  [task 1](https://wandb.ai/maj90/celeba32-unet/runs/xrtcx766),
  [task 2](https://wandb.ai/maj90/celeba32-unet/runs/hynecbjf), and
  [task 3](https://wandb.ai/maj90/celeba32-unet/runs/v9ezlf83).
- Next check: confirm continued progress and inspect GPU power after the cluster
  low-power monitoring interval. Confirm tasks 4-7 start as slots become free.


### 2026-09-14 16:35 CEST - Release all tasks for job 17513629

- Changed the live array throttle from four to eight with `scontrol`; no task
  was cancelled or resubmitted.
- Slurm confirmed all eight tasks `RUNNING`. Tasks 4-7 started at 16:33:53 CEST;
  tasks 0-3 continued their existing runs.
- Future arrays of about 80 tasks or fewer now default to full parallelism.
  Larger arrays require an explicit recorded throttle decision.
- Next check: confirm tasks 4-7 pass startup and the Berzelius low-power
  monitoring window, then record application progress and GPU-power evidence.


### 2026-09-14 17:16 CEST - Logging-axis diagnosis and progress check

- Slurm and `sacct` confirmed all eight tasks `RUNNING`. Tasks 0-3 reached
  steps 190,625-191,700; tasks 4-7 reached steps 34,825-35,425.
- W&B history for task 0 confirmed that its internal `_step` counts logging
  calls rather than optimizer updates. For example, optimizer step 250 appeared
  at `_step` 11, followed by generation step 250 at `_step` 12.
- The active processes imported the old logger at startup. Their existing and
  future records therefore retain `train/step` and `generation/step` as their
  correct update axes; this source edit does not retrofit `optimizer_step`.
- Next check: confirm tasks 4-7 remain active after the one-hour low-power
  monitoring point and record their latest steps.

### 2026-09-15 12:04 CEST - Job 17513629 completed

- `sacct` reports all eight array tasks `COMPLETED` with exit code `0:0`.
- Tasks 0-3 ran for 19:19-19:26 and ended at 08:51-08:57 CEST.
- Tasks 4-7 ran for 19:17-19:24 and ended at 11:51-11:57 CEST.
- Every `run.json` records state `trained`. Every run contains the final
  `step_1000000.pt` checkpoint.
- No task hit the 48-hour limit. No low-power cancellation occurred.
- The separate checkpoint evaluator is prepared but was not submitted during
  this check.


### 2026-09-15 12:44 CEST - First 1k eval attempt failed at startup

- Submitted evaluation array `17522330` as `0-5`, unthrottled, plus dependent
  combine job `17522331`.
- Scope: six existing `eps/eps` configurations; checkpoints 100k, 390k, and
  695k; sample/FID-reference/loss counts all 1,000.
- All six tasks started on Berzelius GPU nodes and failed after four seconds
  with exit `1:0`. No checkpoint sampling or metric computation began.
- Shared cause in every stderr log: the Mambaforge wrapper referenced unset
  `CONDA_DEFAULT_ENV` while the Slurm wrapper had `set -u` enabled.
- Dependency handling cancelled combine job `17522331` before it ran.

### 2026-09-15 12:45 CEST - Corrected and relaunched 1k eval

- Initialized `CONDA_DEFAULT_ENV` before `module load Mambaforge` in both the
  evaluator and combine Slurm wrappers.
- Submitted replacement array `17522337` as `0-5`, unthrottled, plus dependent
  combine job `17522338`.
- Output: [`20260915T104512Z_2903270_three_point`](outputs/evals/eps_recovery_1m/20260915T104512Z_2903270_three_point).
- Each array task requests one GPU, eight CPUs, 64 GB, and two hours. The
  combine job requests one GPU, two CPUs, 16 GB, and 30 minutes.
- Startup check: all six evaluator tasks are `RUNNING` on six separate nodes
  with exit `0:0`; combine is `PENDING (Dependency)`. All six passed the former
  Mambaforge failure point and their stderr contains no application error.
- Next check: confirm the first checkpoint metric row for every task, then
  confirm all 18 rows and the combined CSV/plot.



### 2026-09-15 12:48 CEST - Replacement 1k eval completed

- `sacct` reports all six tasks in array `17522337` `COMPLETED` with exit
  `0:0`; runtimes were 1:15-1:37. Combine job `17522338` completed with exit
  `0:0` in 13 seconds.
- All six task summaries record `state=completed`, three loss rows, and three
  metric rows. Combined acceptance count is 18/18 `ok`, with zero nonfinite
  rows at steps 100k, 390k, and 695k.
- Final artifacts: `combined/eps_config_checkpoint_metrics.csv`,
  `combined/eps_config_comparison.png`, and `combined/summary.json` under the
  replacement output directory.
- Quick 1k-sample signal: exact/full-Heun FID was 449.8, 185.5, and 87.2;
  exact/final-Euler was 99.8, 80.7, and 68.0. The three safe-division
  standard-LR paths were much lower. Direct/full-Heun was best at 100k
  (33.0); old via-x/full-Heun was marginally best at 390k and 695k (35.3 and
  36.4 versus 35.5 and 36.7). Treat their small difference as unresolved at
  this smoke scale rather than as a final 10k-sample estimate.
- No timeout or low-power cancellation occurred. The active-runs table is now
  empty.



### 2026-09-15 12:56 CEST - Launch three-point 10k eps/eps evaluation

- Submitted evaluation array `17522748` as `0-5`, unthrottled, plus dependent
  combine job `17522749` under account `berzelius-2026-78`.
- Scope: the same six existing `eps/eps` configurations and checkpoints 100k,
  390k, and 695k, now with 10,000 generated samples per point. The exact saved
  1,024-image held-out artifact is used for both FID reference and loss.
- Output: [`20260915T105557Z_2941333_three_point`](outputs/evals/eps_recovery_1m/20260915T105557Z_2941333_three_point).
- Each evaluator task requests one GPU, eight CPUs, 64 GB, and two hours. The
  combine job requests one GPU, two CPUs, 16 GB, and 30 minutes.
- Startup check: all six evaluator tasks are `RUNNING` with exit `0:0` across
  the allocated Berzelius nodes. Stderr contains only the normal Mambaforge
  notice; combine is `PENDING (Dependency)`.
- Next check: confirm first 10k metric rows, then all 18 rows and combined
  outputs.



### 2026-09-15 13:01 CEST - Three-point 10k eval completed

- `sacct` reports all six tasks in array `17522748` `COMPLETED` with exit
  `0:0`; runtimes were 4:15-4:32. Combine job `17522749` completed with exit
  `0:0` in 14 seconds.
- All six summaries record `state=completed`, with three loss rows and three
  metric rows each. Combined acceptance count is 18/18 `ok` and zero
  nonfinite rows.
- Final artifacts: `combined/eps_config_checkpoint_metrics.csv`,
  `combined/eps_config_comparison.png`, and `combined/summary.json`.
- Direct safe/full-Heun FID was 28.19, 31.35, and 33.54 at 100k, 390k, and
  695k. Old via-x/full-Heun was close at 29.22, 31.69, and 32.99. Direct was
  better at the first two points and had slightly lower pixel f_mem at all
  three; old was 0.55 FID better at 695k.
- Safe final-Euler was consistently worse than safe full-Heun by roughly
  2.4-3.0 FID. Exact division remained decisively worse. The LR 1e-3 run had
  lower held-out loss but worse FID and higher f_mem than the two leading
  standard-LR/full-Heun runs.
- Operational decision signal: use direct epsilon conversion with safe
  division and full Heun for the next eps/eps training configuration; retain
  old via-x/full-Heun as the close reference.
- No timeout or low-power cancellation occurred. The active-runs table is
  empty.



### 2026-09-15 13:17 CEST - Correct FID-10k protocol and launch replacement

- The completed `17522748` FID values are invalid for the intended FID-10k
  comparison: they used 10,000 generated samples but only 1,024 reference
  images. The loss, pixel-space f_mem, and raw diagnostics remain valid. The
  prior operational recommendation based on those FID values is withdrawn.
- Submitted reference-preparation job `17523389`, dependent unthrottled
  evaluation array `17523390` (`0-5`), and dependent combine job `17523391`
  under account `berzelius-2026-78`.
- Corrected scope: 10,000 generated samples against one shared frozen 10,000
  CelebA `valid` reference; 1,024 saved held-out images remain in use for loss
  and pixel-space f_mem.
- The reference job must verify the exact saved 1,024-image prefix before the
  array is released. Each evaluator requests one GPU, eight CPUs, 64 GB, and
  two hours; all six run concurrently.
- Output: [`20260915T111553Z_3025872_three_point`](outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point).
- Initial status: reference job `RUNNING` on `node011`; array and combine are
  `PENDING (Dependency)`.


### 2026-09-15 13:23 CEST - Corrected FID-10k evaluation completed

- Reference job `17523389` completed with exit `0:0` in 1:55. Its manifest
  records 10,000 tensors from `torchvision_celeba:valid`, selection seed 1,
  tensor SHA-256 `3cdd00c0f000de7a8eabd3ee880491c53c8276f4011cbe18e99151e06021d40f`,
  and `saved_prefix_verified=true` for all 1,024 training-artifact references.
- All six tasks in array `17523390` completed with exit `0:0` in 4:31-4:35.
  Combine job `17523391` completed with exit `0:0` in 14 seconds.
- Acceptance: 18/18 rows are `ok`, with zero nonfinite rows. The combined
  summary records `sample_counts=[10000]` and `fid_reference_counts=[10000]`;
  every `fid_10k` value equals its FID value.
- Direct safe/full-Heun FID-10k is 20.57, 24.43, and 27.13 at 100k, 390k, and
  695k. Old via-x/full-Heun is 21.90, 25.15, and 26.65. Direct is better at the
  first two checkpoints and has slightly lower pixel f_mem at all three; old is
  0.49 FID better at 695k.
- Safe final-Euler is consistently worse than safe full-Heun. Exact division
  remains decisively worse. The corrected decision signal therefore remains
  direct epsilon conversion with safe division and full Heun, with old
  via-x/full-Heun retained as the close late-training reference.
- No timeout, low-power cancellation, or nonfinite evaluation occurred. The
  active-runs table is empty.


### 2026-09-15 13:35 CEST - Launch v/v and x/x reference evaluation

- Submitted reference array `17523970` as unthrottled tasks `0-1`, plus
  dependent combine job `17523971`, under account `berzelius-2026-78`.
- Scope: saved source 6 (`v/v`) and source 7 (`x/x`) at 100k, 390k, and 695k,
  each with its original saved 50-step Heun/final-Euler sampler.
- Protocol: 10,000 generated samples against the already verified frozen
  10,000-image CelebA `valid` reference. The two tasks reuse the corrected
  evaluation noise and FID reference-feature cache; loss and pixel-space f_mem
  continue to use the saved 1,024-image held-out artifact.
- Both tasks started together on `node051`; combine is `PENDING (Dependency)`.
- Output is appended to [`20260915T111553Z_3025872_three_point`](outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point), with a separate reference manifest and job-ID record.


### 2026-09-15 13:40 CEST - v/v and x/x reference evaluation completed

- Both tasks in array `17523970` completed with exit `0:0` in 4:32-4:33.
  Combine job `17523971` completed with exit `0:0` in 15 seconds.
- Acceptance: 6/6 reference rows and all 24 combined parameterization rows are
  `ok`; the summary records 10,000 generated and 10,000 reference images for
  every row.
- FID-10k at 100k/390k/695k is 25.86/28.72/30.19 for `v/v` and
  25.69/27.70/24.97 for `x/x`.
- Pixel f_mem at rho 0.3 is already 0.9995/1.0000/1.0000 for `v/v` and
  0.9996/1.0000/1.0000 for `x/x`. This is substantially higher than the
  leading epsilon configurations and should be inspected alongside the saved
  panels and later feature-space f_mem work.
- The combined plots intentionally omit cross-parameterization training loss:
  eps, v, and x losses use different targets and their raw magnitudes are not
  directly comparable. Per-run loss values remain in the CSV.
- Final artifacts include `reference_checkpoint_metrics.csv`,
  `reference_comparison.png`, `all_parameterization_checkpoint_metrics.csv`,
  `all_parameterization_comparison.png`, and `reference_summary.json` under
  the corrected evaluation's `combined/` directory. The active-runs table is
  empty.
