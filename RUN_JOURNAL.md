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

| Job | Study | Status | Resources | Latest evidence | Batch directory |
| --- | --- | --- | --- | --- | --- |
| `17662195` (initial); `17683429` (resume) | CelebA32-gray Small-DiT full 3x3 at N=512/2048; seed 1; target 1M; via-x safe sampling | RUNNING: 18/18 | Per task: one thin A100-40GB, four CPUs, 32 GB, 24 hours; physical/effective batch 64/256 | At 2026-10-01 14:39 CEST, all rows had exact recovery at step 295,000-300,000, `error=null`, finite losses, and clean current logs. | [`20260929T124425Z_2240178`](outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178) |
| `17645048` (MNIST32); `17645049`-`17645053` (MNIST64 stages); verifier `17645054` | Fresh MNIST32 U-Net-S and MNIST64 U-Net-L diagonals at N=512/2048; seeds 0-2; target 1M | PARTIAL: MNIST32 18/18 complete; MNIST64 18/18 running stage 5 | Thin A100-40GB for MNIST32; fat A100-80GB for MNIST64; physical/effective batch 256/256 | At 2026-10-01 14:39 CEST, MNIST64 was at steps 955,000-960,000 with exact recovery, `error=null`, finite losses, and clean stage-5 logs. | [`20260927T165106Z_1991997`](outputs/job_sets/mnist_diagonal_1m/20260927T165106Z_1991997) |
| `17654914`-`17654918` (stages); verifier `17654919` | Fresh CelebA64 RGB U-Net-L diagonals at N=512/2048; seeds 0-2; target 1M | RUNNING: 18/18 in stage 4; stage 5 and verifier dependency-pending | Per task: one fat A100-80GB, eight CPUs, 64 GB, 30 hours; physical/effective batch 256/256 | At 2026-10-01 14:39 CEST, all rows had exact recovery at step 620,000, `error=null`, finite losses, and clean stage-4 logs. | [`20260928T131627Z_3579761`](outputs/job_sets/celeba64_rgb_diagonal_1m/20260928T131627Z_3579761) |

## Closed runs

| Job | Study | Final status | Finished | Final artifacts | Batch directory |
| --- | --- | --- | --- | --- | --- |
| `17639669` | CelebA32-gray Small-DiT diagonal `x/x`, `eps/eps`, `v/v` pilot | COMPLETED, 3/3 at step 100,000; exit `0:0` | 2026-09-26 04:24 CEST | Per run: 28 checkpoints, generated tensors, projected-pixel records and grids; exact step-100k recovery; successful W&B finish | [`20260925T172747Z_3410286`](outputs/job_sets/celeba32_dit_diagonal_100k_viax/20260925T172747Z_3410286) |
| arrays `17556454`, `17556455`, `17556456`, `17556457`; verifier `17556458` | EPS/EPS 64x64 N=512 capacity gate | COMPLETED, 24/24 training tasks and final verifier; applications paused normally at step 400,000 | 2026-09-21 10:42 CEST | Exact two-process training, 24 checkpoints and recovery states, 1,344 valid sampler diagnostics, W&B finishes, and `GATE_400K_COMPLETE` | [`20260919T155341Z_3213634`](outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634) |
| arrays `17556193`, `17556194`; resume smoke `17556449`; finalizer `17556451` | Corrected W&B and exact-resume gate for the 64x64 study | COMPLETED, 6/6 W&B smokes plus strict process-resume comparison | 2026-09-19 18:21 CEST | Four-profile artifacts, successful online W&B finishes, exact model/Adam/RNG/batch/history comparison, and `SMOKE_COMPLETE` | [`20260919T155341Z_3213634`](outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634) |
| arrays `17556165`, `17556166`; finalizer `17556173` | EPS/EPS 64x64 N=512 capacity probes | COMPLETED, 6/6 profiles and finalizer | 2026-09-19 17:58 CEST | Fresh-process batch probes, selected physical batch 256 for every profile, and `PROBE_COMPLETE` | [`20260919T155341Z_3213634`](outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634) |
| `17549419` | CelebA32 prediction/loss 3x3, 200k gate | COMPLETED, 9/9; application paused normally at planned step 200,000 | 2026-09-18 23:41 CEST | Nine restart checkpoints, sampled tensors, comparison grids, projected-pixel f_mem records, and successful W&B syncs | [`20260918T164642Z_3540146`](outputs/job_sets/celeba32_prediction_loss_3x3/20260918T164642Z_3540146) |
| `17549408` | EPS/EPS data and image-format, 200k gate | COMPLETED, 7/7; application paused normally at planned step 200,000 | 2026-09-19 06:06 CEST | Seven restart checkpoints, sampled tensors, comparison grids, projected-pixel f_mem records, and successful W&B syncs | [`20260918T164620Z_3539380`](outputs/job_sets/eps_eps_data_image_format/20260918T164620Z_3539380) |
| `17549382` | Fully seeded exact CUDA resume smoke | COMPLETED, all exact comparisons passed | 2026-09-18 18:45 CEST | Bitwise-equal model, Adam, RNG, sampler, and history | [`20260918T164438Z_3530934`](outputs/job_sets/resume_cuda_smoke/20260918T164438Z_3530934) |
| `17549117` | Deterministic exact CUDA resume smoke | FAILED, RNG comparison rejected | 2026-09-18 18:42 CEST | Model and Adam exact; unseeded Python RNG differed | [`20260918T163642Z_3490500`](outputs/job_sets/resume_cuda_smoke/20260918T163642Z_3490500) |
| `17549012` | EPS/EPS data and image-format gate | CANCELLED, 0/7 tasks started; DependencyNeverSatisfied | 2026-09-18 18:30 CEST | Frozen source/config/variant snapshot only | [`20260918T163036Z_3465472`](outputs/job_sets/eps_eps_data_image_format/20260918T163036Z_3465472) |
| `17549010` | CelebA32 prediction/loss 3x3 gate | CANCELLED, 0/9 tasks started; DependencyNeverSatisfied | 2026-09-18 18:30 CEST | Frozen source/config/variant snapshot only | [`20260918T163036Z_3465363`](outputs/job_sets/celeba32_prediction_loss_3x3/20260918T163036Z_3465363) |
| `17548840` | Exact CUDA resume smoke | FAILED, comparison gate rejected | 2026-09-18 18:30 CEST | Baseline and resumed step-40 states plus comparison traceback | [`20260918T161941Z_3414143`](outputs/job_sets/resume_cuda_smoke/20260918T161941Z_3414143) |
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


### 2026-09-18 18:20 CEST - Launch exact CUDA resume smoke

- Submitted job `17548840` under account `berzelius-2026-78` on one thin A100
  with a 30-minute limit.
- The job runs one uninterrupted 40-update MNIST UNet trajectory and one
  20-update planned stop followed by exact resume to update 40. Both use
  physical batch 8, effective batch 16, Adam, dropout, and CUDA TF32.
- Acceptance requires bitwise-equal model, optimizer, RNG, sampler, and
  history state. The resumed run must record two process attempts.
- Batch snapshot:
  [`20260918T161941Z_3414143`](outputs/job_sets/resume_cuda_smoke/20260918T161941Z_3414143).
- Initial Slurm state: `PENDING (Resources)`. No long experiment was submitted
  before this check.


### 2026-09-18 18:31 CEST - CUDA comparison rejected; dependent arrays never ran

- Smoke job `17548840` ran on `node052` for 44 seconds and failed with exit
  `1:0` in the final exact-state comparison. Both 40-update paths completed.
- The separately started trajectories already differed at step 20, before the
  resume boundary: 91 model tensors differed, with maximum absolute difference
  `3.81e-5`. At step 40, 99 tensors differed by at most `1.08e-4`.
- Final CPU and CUDA RNG states, batch-sampler state, and logged loss history
  matched exactly. This localizes the rejection to cross-process CUDA kernel
  determinism rather than missing saved RNG or sampler state.
- Submitted unthrottled arrays `17549010` (`0-8`, thin) and `17549012` (`0-6`,
  fat) with `afterok:17548840`. Slurm immediately cancelled both with
  `DependencyNeverSatisfied` when the smoke failed. No array task started and
  no experiment run directory was created.
- Next action: enable an explicit deterministic CUDA runtime for recoverable
  runs, rerun the exact smoke, and submit fresh guarded arrays only after the
  replacement smoke passes.


### 2026-09-18 18:36 CEST - Launch deterministic replacement smoke

- Added an opt-in deterministic CUDA runtime: deterministic PyTorch algorithms,
  deterministic cuDNN, disabled cuDNN benchmarking, and the documented cuBLAS
  workspace configuration. Existing configs remain nondeterministic by default.
- Enabled the mode for the smoke and both planned long-run configs. The full
  local suite still passes, 86/86.
- Submitted replacement smoke `17549117` on one thin A100 for 30 minutes. Its
  initial state is `PENDING (Resources)`; Slurm estimates a 19:25 CEST start.
- Batch snapshot:
  [`20260918T163642Z_3490500`](outputs/job_sets/resume_cuda_smoke/20260918T163642Z_3490500).


### 2026-09-18 18:43 CEST - Deterministic smoke isolates process seeding gap

- Job `17549117` ran on `node038` for 49 seconds and failed only in the final
  RNG comparison. The final model checkpoint and recovery model and Adam state
  matched bitwise; batch-sampler state and loss history also passed.
- The mismatch was `rng.python`. Fresh runs seeded Torch but did not seed the
  process-global Python or NumPy generators, so the independent baseline and
  interrupted trajectory began from different non-training RNG states.
- No experiment array was submitted from this smoke. Next action: seed all
  three process RNGs for fresh runs, retain exact restore on resume, and rerun
  the complete gate.


### 2026-09-18 18:44 CEST - Launch fully seeded exact-recovery smoke

- Fresh training processes now seed Python, NumPy, and Torch from the recorded
  training seed. Resume still restores the captured states after reconstruction.
- Monitoring and W&B callbacks preserve all three RNG streams, preventing
  observability work after a recovery write from shifting future training.
- Focused exact recovery tests passed, and the complete local suite passed
  87/87 with only existing dependency warnings.
- Submitted smoke `17549382`; it started immediately on `node012`. Batch:
  [`20260918T164438Z_3530934`](outputs/job_sets/resume_cuda_smoke/20260918T164438Z_3530934).


### 2026-09-18 18:47 CEST - Exact recovery passed; launch both 200k gates

- Smoke `17549382` completed on `node012` in 53 seconds with exit `0:0`.
  Its final comparison passed bitwise for the model, Adam, Python/NumPy/Torch
  CPU and CUDA RNGs, stateful batch sampler, and training history.
- Submitted unthrottled thin array `17549419` as tasks `0-8` for the complete
  CelebA32 gray prediction/loss 3x3. Each task uses physical/effective batch
  256, seed 0, deterministic CUDA, and stops first at update 200,000.
- Submitted unthrottled fat array `17549408` as tasks `0-6` for the EPS/EPS
  data/image-format study. Each task uses physical batch 128, two accumulation
  steps, effective batch 256, seed 0, deterministic CUDA, and the same 200k gate.
- Initial Slurm state: data-format tasks 0-5 are `RUNNING` across `node068`,
  `node074`, and `node093`; task 6 is `PENDING (Resources)`. All nine 3x3 tasks
  are `PENDING (Priority)`. No array throttle is active.
- W&B groups are `celeba32-pred-loss-3x3-20260918T164642Z_3540146` and
  `eps-eps-data-image-format-20260918T164620Z_3539380`.


### 2026-09-18 18:48 CEST - Data/image-format startup verified

- All seven tasks in array `17549408` are now `RUNNING`. Every row loaded 256
  training and 1,000 held-out images with its intended channel count and 32 or
  64 pixel shape. Parameter counts are 3,917,409 for gray base, 3,918,563 for
  RGB base, and 6,465,379 for the deep Flowers capacity probe.
- All rows created an online W&B run and a complete step-1 recovery state.
  CelebA32 gray and RGB and MNIST have already reached at least step 250; no
  nonfinite loss, OOM, or application error is present.
- W&B run IDs by row 0-6: `3ikk1a04`, `l1wijncq`, `l6zvm6uc`, `ubu7pubt`,
  `fjm57d9s`, `aw95u14o`, and `0m7lrcpd`.
- All nine tasks in `17549419` remain `PENDING (Resources)`. Slurm currently
  estimates 19:25 CEST for all nine; the array remains unthrottled.


### 2026-09-18 18:58 CEST - Multi-minute live monitoring check

- Watched both arrays continuously at 45-second intervals. The error scan stayed
  empty for traceback, OOM, nonfinite loss, runtime errors, and failed states.
- All seven data/image-format tasks remain `RUNNING`. Latest logged optimizer
  steps by row 0-6 are 5,750, 6,250, 6,000, 3,000, 2,750, 2,750, and 2,500.
  Every row continues to save generated tensors and comparison grids and log
  them to its online W&B run.
- Thin GPUs released 3x3 rows 0 (`x/x`) and 1 (`x/eps`) on `node041` and
  `node038`. They reached logged steps 3,750 and 2,500, respectively, with
  online W&B runs `c4ydcsxs` and `0sdtea2o`, checkpoint sampling, comparison
  grids, projected f_mem, and valid rotating recovery states.
- The other seven 3x3 rows remain `PENDING (Resources)`. The array is still
  unthrottled; Slurm will release each row as a thin GPU becomes available.


### 2026-09-19 12:50 CEST - Both 200k gates completed and paused cleanly

- `squeue` is empty for both arrays. `sacct` reports all nine tasks in
  `17549419` and all seven tasks in `17549408` as `COMPLETED` with exit `0:0`.
  The 3x3 tasks ran for 4:22-4:31 and ended by 23:41 CEST. The data/image-format
  tasks ran for 5:13-11:19 and ended by 06:06 CEST.
- All 16 `run.json` files record `state=paused`, `latest_step=200000`, one
  attempt, `stop_reason=planned stop_after_step`, and successful W&B finish.
  Every recovery index reaches exactly step 200,000. Each run has a restart
  checkpoint, generated-sample tensor, comparison grid, and projected-pixel
  f_mem record at that step. The log scan found no traceback, OOM, nonfinite
  loss, runtime error, or failed application state.
- The 3x3 projected-pixel f_mem matrix at 200k, with rows `prediction=x,eps,v`
  and columns `loss=x,eps,v`, is `[[1.000, 0.469, 0.977], [0.961, 0.961,
  0.875], [1.000, 0.961, 1.000]]`. All nine visual grids contain recognizable
  faces. `eps/x` has the strongest visible noise; several other off-diagonal
  cells show milder artifacts. This does not yet establish a final ranking.
- Data/image-format projected-pixel f_mem at 200k is 0.977 for CelebA32 gray,
  0.977 for MNIST32 gray, 0.727 for CelebA32 RGB, 0.117 for CelebA64 gray,
  0.000 for CelebA64 RGB, 0.000 for Flowers64 RGB, and 0.031 for the deeper
  Flowers64 probe. Visuals show good CelebA32-gray and MNIST samples,
  recognizable but color-distorted CelebA32 RGB, face-like CelebA64 gray, and
  poor or collapsed CelebA64-RGB and Flowers64 outputs. The deeper Flowers
  network did not repair the 200k failure.
- These f_mem values are the online 128-sample
  `monitor/random_projected_pixels` diagnostic at rho 0.3. They are not the
  planned fixed-10k full-pixel or SSCD/LPIPS/DINOv2 evaluation and should not be
  compared as if they were the final scientific result.
- Next action: jointly choose which cells merit exact resume toward one million
  updates. Before that decision is evidence-complete, implement and run the
  generalized 200k evaluator described in both job READMEs. No continuation or
  evaluator job was submitted during this check.


### 2026-09-19 17:57 CEST - Launch EPS/EPS 64x64 N=512 capacity probes

- Froze and validated batch
  [`20260919T155341Z_3213634`](outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634).
  Its import assertion resolves `mem_param_src` from the frozen source tree.
- Submitted unthrottled thin array `17556165` as tasks `0-3` on full
  A100-SXM4-40GB GPUs and fat array `17556166` as tasks `4-5` on full
  A100-SXM4-80GB GPUs. Each profile tests physical batches 512, 256, 128, and
  64 in fresh processes with real CelebA64 RGB data.
- Berzelius accepted both GPU arrays. It rejected the first lightweight
  finalizer because every job on this system must request a GPU. Resubmitted
  the same frozen finalizer as `17556173` with one full thin GPU and an
  `afterok:17556165:17556166` dependency. Updated all later finalizer
  submitters with the same explicit GRES requirement.
- At 17:57 CEST, all four thin tasks had completed with exit `0:0` in
  2:42-2:44. Both fat tasks remained running on `node081`; the finalizer was
  pending on their completion. No smoke or main training task was submitted.


### 2026-09-19 17:59 CEST - EPS/EPS 64x64 capacity probes passed

- Thin array `17556165`, fat array `17556166`, and finalizer `17556173` all
  completed with exit `0:0`; `PROBE_COMPLETE` is present. All 24 fresh-process
  measurements (six profiles times four candidate batches) were recorded.
- Physical/effective batch 256 was selected for all six profiles, so the main
  runs use one optimizer update per batch with no gradient accumulation.
- Thin peak reserved memory is 14.56-15.30 GiB against the 32 GiB selection
  ceiling. Fat peak reserved memory is 30.70-31.69 GiB against the 64 GiB
  ceiling. Selected optimizer-step times range from 0.185 to 0.357 seconds.
- The 30%-margin estimates are 26.7-29.2 hours for small/deep runs and
  49.0-51.6 hours for wide-deep runs. The frozen planner therefore selected
  exact-resume split mode, 0-200k followed by 200k-400k.
- Next gate: six online-W&B sampling smokes plus the wide-attention interrupted
  versus resumed exact-state comparison. No main training task is submitted
  until that gate passes.


### 2026-09-19 18:01 CEST - Launch EPS/EPS 64x64 W&B smoke gate

- Submitted thin W&B smoke array `17556176` as tasks `0-3` on full 40GB
  A100s and fat W&B smoke array `17556177` as tasks `4-5` on full 80GB A100s.
  Each task trains for 250 updates and exercises fixed-noise sampling at steps
  0 and 250 under all four sampler profiles, namespaced artifacts, projected
  pixel f_mem, online W&B logging, and planned-stop recovery.
- Submitted the wide-deep attention interrupted/resumed exact-state comparison
  as fat-GPU job `17556178`. Finalizer `17556179` has an `afterok` dependency
  on both W&B arrays and the exact-resume job and will build the runtime plan.
- Initial scheduler state: all four thin tasks, both fat tasks, and the exact-
  resume task are `RUNNING`; finalizer `17556179` is `PENDING (Dependency)`.
- The main 24-run arrays remain unsubmitted until `SMOKE_COMPLETE` exists.


### 2026-09-19 18:02 CEST - Smoke gate stopped on dataset-root launch error

- Thin array `17556176`, fat array `17556177`, and exact-resume job `17556178`
  failed before training in 21-24 seconds. Dependent finalizer `17556179` was
  cancelled. No W&B run or sample artifact was created, and no main task was
  submitted.
- Root cause: the frozen copy of `submit_smokes.sh` was invoked directly. Its
  live-tree relative-path calculation resolved the repository root as
  `mem_param_clean/outputs/job_sets`, so workers looked for CelebA under a
  nonexistent `outputs/old_mem_par/data` path. The successful probes had been
  submitted through the live wrapper, which passed the correct repository root.
- Preserve this failed attempt, repair frozen-wrapper root discovery, validate
  the real data path before submission, and rerun the complete smoke gate.


### 2026-09-19 18:06 CEST - Corrected W&B smoke gate resubmitted

- Preserved the complete failed attempt under
  `validation/failed_attempts/20260919T1802_bad_data_root`; nothing from the
  failed attempt was deleted.
- Changed all live and frozen operational wrappers to discover the Git
  repository root from either layout. Added submit-time cache checks for
  CelebA smokes and for all CelebA, MNIST, and Flowers main-study inputs.
  Shell syntax, corrected frozen-wrapper dry run, and `git diff --check` pass.
- Submitted corrected thin smoke array `17556193`, fat smoke array `17556194`,
  exact-resume job `17556195`, and dependent finalizer `17556196`. All seven
  GPU workers entered `RUNNING`; the finalizer is `PENDING (Dependency)`.
- The corrected commands pass
  `/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_clean`
  as the repository root, resolving data under `../old_mem_par/data`.


### 2026-09-19 18:12 CEST - Exact-resume smoke repaired in place

- The six W&B smoke tasks successfully loaded 512 RGB 64x64 CelebA images,
  created online W&B runs, generated named step-0 sampler artifacts, and began
  training. They remain healthy and running.
- Exact-resume job `17556195` stopped before training because its smoke override
  reduced `evaluation.sample_bank_size` to 8 without also reducing the base
  `evaluation.n_samples` from 128. Its dependent finalizer `17556196` was
  cancelled; this did not affect the six running smoke tasks.
- Preserved that failed exact-resume directory under
  `validation/failed_attempts/20260919T1808_resume_eval_mismatch`, added the
  matching `evaluation.n_samples=8` override, and passed shell/whitespace checks.
- Submitted corrected exact-resume job `17556329` on a full 80GB A100 and
  replacement finalizer `17556330` with dependency
  `afterok:17556193:17556194:17556329`. Both replacement IDs are appended to
  `validation/smoke_jobs.tsv`.


### 2026-09-19 18:18 CEST - Strengthen exact-resume history comparison

- Job `17556329` completed both four-step training paths with identical final
  checkpoint, model, Adam state, Python/NumPy/Torch CPU and CUDA RNGs, shuffled
  batch cursor, and losses at common steps. Its comparator then rejected the
  history length: the planned stop records step 2, while the uninterrupted path
  used the production 250-step logging cadence and therefore did not.
- Preserved that complete attempt under
  `validation/failed_attempts/20260919T1816_resume_history_cadence`. Set the
  four-step smoke's logging cadence to one so both paths record steps 1-4 and
  the existing strict history and monitoring-hash comparisons remain intact.
- Submitted strengthened exact-resume job `17556449` and dependent finalizer
  `17556451`; both IDs are appended to `validation/smoke_jobs.tsv`. Job
  `17556449` is running on a full 80GB A100.


### 2026-09-19 18:23 CEST - Smoke gate passed; launch 24 main runs

- Exact-resume job `17556449` and finalizer `17556451` completed with exit
  `0:0`. The strict comparison passed for checkpoint, model, Adam state, all
  RNGs, shuffled-batch state, every-step loss history, four-profile monitoring
  artifacts, and reused W&B identity. `SMOKE_COMPLETE` is present.
- All six W&B smokes completed at step 250. Every profile has raw 128-sample
  tensors, grids, diagnostics, projected-pixel f_mem at steps 0 and 250, zero
  nonfinite samples, a step-250 checkpoint/recovery state, and a successful
  online W&B finish. Run IDs by profile 0-5 are `hwpmajxw`, `apm0stls`,
  `7axvgjqq`, `afdb4y4l`, `gw8ainp8`, and `m8jujqvg`.
- The measured runtime plan selected exact-resume split mode. Its worst 30%-
  margin estimates are 26.01 hours for stage 1 and 25.85 hours for stage 2,
  each below the 40-hour threshold and 48-hour Slurm limit.
- Submitted unthrottled stage-1 thin array `17556454` as tasks `0-15` and fat
  array `17556455` as tasks `16-23`, both stopping at 200,000. Submitted
  task-matched `aftercorr` stage-2 arrays `17556456` and `17556457`, resuming
  exactly to 400,000. Final verifier `17556458` waits `afterok` on both stage-2
  arrays.
- Initial state: 14 thin and all 8 fat stage-1 tasks are `RUNNING`; thin tasks
  13 and 14 are `CONFIGURING`. Stage 2 and verifier are dependency-pending.
  Main W&B group is `eps-eps-64-n512-20260919T155341Z_3213634`.


### 2026-09-19 18:25 CEST - All 24 main startups verified

- All 16 thin and 8 fat stage-1 tasks are `RUNNING`. All 24 run records are in
  CUDA training state with unique online W&B IDs, physical/effective batch 256,
  finite step-1 loss, checkpoint zero, and rotating recovery state.
- Every row loaded exactly 512 training and 1,000 held-out examples at 64x64.
  CelebA gray and MNIST rows have one channel; CelebA RGB and Flowers rows have
  three channels. Parameter counts match capacity and attention assignments.
- All 24 runs completed their fixed-noise step-0 monitor under direct-safe,
  via-x-safe, direct-exact, and via-x-exact: 96/96 diagnostics are `ok`, with
  zero nonfinite samples, and all raw tensors, grids, and projected-pixel f_mem
  records are present.
- A full stage-1 Slurm-log scan found no traceback, OOM, runtime/configuration
  error, or nonfinite failure. Early training remains under active watch.


### 2026-09-19 18:27 CEST - Multi-minute main-run health check

- All 24 stage-1 tasks remain `RUNNING`. Thin runs have reached logged steps
  500-1,000 and fat runs have reached steps 250-500; every reported loss is
  finite. The slowest wide-deep row has advanced through step 250.
- Repeated full-log scans remain empty for traceback, OOM, nonfinite loss,
  floating-point failure, runtime/configuration error, or assertion failure.
- Stage-2 arrays `17556456` and `17556457` remain correctly dependency-pending
  for task-matched exact resume after 200k. Gate verifier `17556458` remains
  dependency-pending after stage 2. `git diff --check` passes.


### 2026-09-21 14:28 CEST - Complete and review EPS/EPS 64x64 400k gate

- Stage-1 arrays `17556454` and `17556455`, stage-2 exact-resume arrays
  `17556456` and `17556457`, and final verifier `17556458` all completed with
  exit `0:0`. `squeue -u x_moalj` is empty. All 24 runs are paused at the
  planned step 400,000 boundary after two recorded attempts and successful W&B
  finishes.
- The final verifier found 24 step-400k checkpoints, 24 current recovery
  states, and all 96 final sampling-profile diagnostics and f_mem records. It
  also verified every profile at all 14 requested monitor steps.
- All 1,344 sampler diagnostic records are `ok` and contain zero nonfinite
  values. Small/deep stages took 10.3-11.5 hours each; wide-deep stages took
  18.9-20.3 hours each.
- Direct-safe projected-pixel f_mem shows a strong capacity effect at 400k.
  Attention-averaged wide-deep values are 0.930 for CelebA gray, 0.551 for
  CelebA RGB, 0.992 for MNIST gray, and 0.156 for Flowers RGB. Small/deep RGB
  Flowers remain at zero, and small/deep CelebA RGB remain at or near zero.
- Visual review agrees: wide-deep is the first profile with recognizable RGB
  faces and flowers. Attention has no consistent main effect. Capacity is the
  dominant factor in this seed-zero matrix.
- Direct-safe and via-x-safe are stable and visually similar. Exact division
  produces severe saturation for most small/deep face and flower rows; 56-95%
  of pixels fall outside `[-1,1]` in those groups. Exact division remains a
  diagnostic rather than a viable default sampler.
- Full tables, sampler comparisons, visual conclusions, and limitations are in
  [`RESULTS_400K.md`](outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634/RESULTS_400K.md).
  These values are the 128-sample projected-pixel monitor, not fixed-10k FID,
  full-pixel f_mem, or learned-feature f_mem. No run was continued past 400k.


### 2026-09-21 14:40 CEST - Detailed review of CelebA32 prediction/loss 3x3

- Rechecked Slurm array `17549419`: all nine task indices completed with exit
  `0:0` in 4:22:52-4:31:46. Every application is recoverably paused at the
  planned 200,000-update gate with a successful W&B finish. None continued to
  the configured one-million-update ceiling.
- The final projected-pixel f_mem matrix, with prediction rows `x, eps, v` and
  loss columns `x, eps, v`, is `[[1.000, 0.469, 0.977], [0.961, 0.961,
  0.875], [1.000, 0.961, 1.000]]`.
- The trajectory matters: `x/x`, `v/x`, and `v/v` are near saturation by 25k;
  `x/v` rises from 0.000 at 25k to 0.977 at 200k; `x/eps` remains at 0.469.
  `eps/x` and `eps/v` are visibly non-monotonic at the sampled checkpoints.
- Visual inspection finds recognizable faces in all nine final grids and no
  collapse. `eps/x` is the noisiest and has the largest raw out-of-range pixel
  fraction, while `x/eps` is coherent despite its lower f_mem. This reinforces
  that the monitor measures training-set proximity, not perceptual quality.
- All final tensors are finite; only 0.24-3.54 percent of raw pixels are outside
  `[-1,1]`. The complete table, grids, interpretation, and limitations are in
  [`RESULTS_200K.md`](outputs/job_sets/celeba32_prediction_loss_3x3/20260918T164642Z_3540146/RESULTS_200K.md).
- This remains seed-zero online triage with 128 projected-pixel samples. No
  fixed-10k, FID, full-pixel, or learned-feature evaluation was inferred from
  it, and raw losses were not compared across differently scaled targets.


### 2026-09-21 14:55 CEST - Consolidate experiment record for the paper session

- Confirmed that `squeue -u x_moalj` is empty. Rechecked the successful probe,
  smoke, exact-resume, main-array, and verifier job IDs for the 64x64 study;
  every successful stage reports `COMPLETED` with exit `0:0`. Corrected the
  stale active-runs row and added the successful stages to the closed index.
- Added the missing full report for the seven-row EPS/EPS data/image-format
  gate at
  [`RESULTS_200K.md`](outputs/job_sets/eps_eps_data_image_format/20260918T164620Z_3539380/RESULTS_200K.md).
  It records all trajectories, raw-range checks, controlled comparisons,
  visual conclusions, confounding, and the relation to the later capacity
  sweep.
- Consolidated the one-million CelebA32 anchors, the 3x3 gate, the seven-row
  format gate, and the 24-row capacity gate in `PAPER_NOTES.txt`. The note
  separates supported findings from claims that still require fixed-10k,
  learned-feature, and multi-seed evidence.
- Recorded the next-session decision in `TODO.txt` as T019 and T020. The next
  session must freeze the full-run matrix, common sampler, checkpoint rule,
  evaluation artifact contract, compute budget, and AISTATS main/appendix
  figure list before any new long job is submitted.
- No training, evaluation, continuation, or paper-figure job was submitted in
  this documentation pass. All 3x3 rows remain paused at 200k, all seven
  data/image rows remain paused at 200k, and all 24 capacity rows remain paused
  at 400k.


### 2026-09-25 19:28 CEST - Launch Small-DiT CelebA32 diagonal array 17639669

- Submitted the unthrottled three-task array under account
  `berzelius-2026-78`: task 0 is `x/x`, task 1 is `eps/eps`, and task 2 is
  `v/v`. All use seed 0, the identical saved CelebA32-gray N=256 subset, and
  100,000 optimizer updates.
- Network: Small DiT-S/2 with depth 12, hidden size 384, six heads, MLP ratio
  4, patch size 2, one image channel, and 32,563,972 trainable parameters.
- Optimization: Adam at 1e-4, physical batch 64, effective batch 256 through
  four accumulated microbatches, and the 28-step `bonnaire-1m` checkpoint
  schedule through 100k. Exact recovery state is saved at least every 5,000
  updates and on `SIGUSR1` before the 48-hour limit.
- Sampling monitor: 50-step full Heun, safe division with denominator floor
  0.05, `sampling.eps_to_v=via_x`, and `x_t_max` output. The via-x conversion
  is exercised by the epsilon-prediction task; it is inert for native x and v
  predictions.
- Frozen batch snapshot:
  [`20260925T172747Z_3410286`](outputs/job_sets/celeba32_dit_diagonal_100k_viax/20260925T172747Z_3410286).
  W&B project/group: `celeba32-dit-diagonal` /
  `celeba32-dit-diagonal-100k-viax-20260925T172747Z_3410286`.
- Initial scheduler state: task 0 `RUNNING` on `node012`; tasks 1-2
  `PENDING (Resources)`. Next action: confirm all three application configs,
  saved data identities, W&B starts, finite initial losses, and step-0
  monitoring artifacts.


### 2026-09-25 19:32 CEST - Task-0 Small-DiT startup verified

- Task 0 (`x/x`) is running on `node012`. The application loaded exactly 256
  training and 1,024 held-out CelebA-gray tensors with shape `[1,32,32]`,
  constructed the 32,563,972-parameter DiT, and persisted the exact data
  bundle, manifest, fixed noise bank, resolved config, and run record.
- The persisted config confirms unit weighting, x prediction/x loss, 100k
  target updates, physical/effective batch 64/256 with accumulation 4,
  full-Heun safe sampling, and `eps_to_v=via_x`. W&B run `4cy4axyl` is online.
- Step-0, step-250, and step-500 monitoring completed with 128 finite saved
  samples per checkpoint and projected-pixel diagnostics. Training losses at
  steps 1, 250, and 500 are finite: 0.280562, 0.079990, and 0.068929.
- No traceback, OOM, nonfinite warning, or runtime error appears in the live
  stdout/stderr scan. Tasks 1-2 remain `PENDING (Resources)` and have not yet
  created application artifacts; this is scheduler waiting, not a run error.


### 2026-09-25 19:36 CEST - EPS/EPS startup verified

- Task 1 (`eps/eps`) started on `node008`, reached step 250, and produced
  finite losses (0.999696 at step 1 and 0.140919 at step 250). Its step-0 and
  step-250 monitors saved 128 finite samples and projected-pixel diagnostics.
- The live sampler message explicitly reports `eps to v: via_x`, confirming
  that the requested epsilon-to-x-to-velocity route is active in the task
  that exercises it.
- Task 0 reached step 1,250 with finite loss and monitor artifacts. A complete
  live log scan still finds no traceback, OOM, nonfinite warning, runtime
  error, or assertion failure. Task 2 remains `PENDING (Resources)`.
- Tasks 0 and 1 persist the identical exact data-bundle SHA-256
  `5cedc52c672920917445fe8de831335f71c31d8f0c210e2f8abad03dc29f85b5`.
  All 107 files listed in the frozen source/job snapshot passed their SHA-256
  checks; shell syntax and `git diff --check` also passed.


### 2026-09-25 19:57 CEST - All three Small-DiT tasks running

- Slurm reports all array tasks `RUNNING`: task 0 on `node012`, task 1 on
  `node008`, and task 2 on `node029`, each within its 48-hour allocation.
- Latest verified recoverable steps are 5,000 for `x/x`, 4,250 for
  `eps/eps`, and 1,750 for `v/v`. Live stdout has advanced `x/x` through
  step 5,250. The corresponding latest finite losses are 0.059085 at step
  5,000, 0.067521 at step 4,250, and 0.224955 at step 1,750.
- All three run records remain in `training` state with `error: null`, online
  W&B identities, checkpoint artifacts, and both rotating recovery slots.
  All persist the same exact data-bundle SHA-256.
- The full Slurm-log scan finds no traceback, OOM, nonfinite warning, runtime
  error, or assertion failure. One concurrent read of the EPS/EPS history
  returned a transient filesystem error; an immediate retry succeeded and
  matched its step-4,250 stdout and recovery state.


### 2026-09-26 11:03 CEST - Complete Small-DiT diagonal array 17639669

- Slurm confirms all three tasks `COMPLETED` with exit `0:0`. Runtime was
  8:38:46 for `x/x`, 8:37:42 for `eps/eps`, and 8:38:50 for `v/v`; the last
  task finished at 04:24:11 CEST. The user queue is empty.
- Every application record is `trained` with `error: null`, one uninterrupted
  attempt, the identical saved data-bundle hash, and a step-100,000 model
  checkpoint plus exact rotating recovery state.
- Each run contains all 28 planned checkpoints, 28 raw generated-sample
  tensors, 28 comparison grids, and 28 projected-pixel f_mem records. W&B
  finished and synced 28 media files for every task.
- Final training losses are 0.037338 for `x/x`, 0.007908 for `eps/eps`, and
  0.056867 for `v/v`. These target-specific raw losses are not directly
  comparable across parameterizations.
- Projected-pixel f_mem at 10k/25k/50k/75k/100k is
  `0.5625/0.9688/0.9766/1.0000/1.0000` for `x/x`,
  `0.0156/0.4688/0.8672/0.9531/0.9766` for `eps/eps`, and
  `0.4375/0.9688/1.0000/1.0000/1.0000` for `v/v`.
- Within this seed-zero online monitor, epsilon prediction reaches the same
  high-memorization regime later than x or v. This is trajectory evidence,
  not a perceptual-quality ordering.
- The log scan is clean: no traceback, OOM, nonfinite warning, runtime error,
  or assertion failure. This pilot still lacks multi-seed aggregation,
  fixed-10k FID, full-pixel or learned-feature memorization evaluation, and
  a cross-run comparison report; none of those is inferred from the 128-sample
  projected-pixel monitor.


### 2026-09-26 12:44 CEST - Launch Small-DiT 3x3 training-size matrix 17641282

- Submitted one unthrottled 36-task array under `berzelius-2026-78`. The
  matrix is seed 0 and the complete prediction/loss 3x3 at
  `N={512,1024,2048,4096}`. Every row has a one-million-update scientific
  target.
- Network and optimization are fixed across rows: Small DiT-S/2 with
  32,563,972 parameters, Adam at 1e-4, unit loss weighting, physical batch
  64, effective batch 256, deterministic fp32/TF32 runtime, and 72 Bonnaire
  checkpoints.
- Training conversions use exact algebra. Monitoring uses 50-step full Heun,
  safe sampling division with floor 0.05, `eps_to_v=via_x`, `x_t_max`, no
  final Euler step, and no final pixel clipping.
- Each task requests one thin A100-SXM4-40GB, four CPUs, 32 GB host memory,
  and 24 hours. `SIGUSR1` is scheduled five minutes before wall-time so each
  task can save exact state for manual resume. Automatic requeue is disabled.
- Frozen batch snapshot:
  [`20260926T104429Z_1787399`](outputs/job_sets/celeba32_dit_3x3_nscale_1m_viax/20260926T104429Z_1787399).
  W&B project/group: `celeba32-dit-3x3-nscale` /
  `celeba32-dit-3x3-nscale-1m-viax-20260926T104429Z_1787399`. W&B is strict
  to preserve one identity across resumes; exact local artifacts remain
  authoritative.
- Initial scheduler state: all 36 tasks `PENDING (Resources)`. Next action:
  verify frozen-source imports, exact factor configs and data hashes, online
  W&B starts, finite first losses, step-0 monitoring, and recovery state.


### 2026-09-27 14:59 CEST - First Small-DiT wall-time interruptions verified

- Slurm reports tasks 0-15 as `FAILED` with exit `75:0` after 23:54-23:55,
  and tasks 16-35 as `RUNNING`. The frozen job contract defines exit 75 as the
  expected result after the five-minute `SIGUSR1` warning, exact-state save,
  and clean attempt close. All 16 terminal stdout logs record interruption by
  `SIGUSR1`; this is a recoverable wall-time boundary, not a training error.
- All nine N=512 tasks have exact terminal recovery states at steps
  275,882-278,249. Seven N=1024 tasks have exact terminal states at steps
  276,928-277,745; the two running N=1024 tasks have durable step-250,000
  states. The running N=2048 tasks have durable states at steps 240,000-250,000,
  and the running N=4096 tasks have durable states at steps 240,000-245,000.
  Every one of the 36 recovery indexes references a present, non-empty state
  file.
- The 16 stopped `run.json` files record `state=interrupted`, `error=null`, the
  exact terminal step, and their preserved W&B run IDs. Their W&B runs closed
  and synced. The other 20 records remain in `state=training` with `error=null`
  and active W&B identities.
- Each training size still has one shared data-bundle hash across its nine
  prediction/loss rows, with a distinct hash for each of the four sizes. A
  complete stdout/stderr scan remains empty for traceback, OOM, CUDA error,
  nonfinite failure, `RuntimeError`, `AssertionError`, or exception.
- The user queue contains only the 20 surviving tasks from array `17641282`.
  No `dit3x3_resume` allocation appears in Slurm accounting, and the batch has
  no `recovery_jobs.tsv`. No resume was submitted. The remaining tasks should
  receive their first warning between about 17:03 and 17:53 CEST; recheck all
  terminal states and exact recovery steps after that window.


### 2026-09-27 15:06 CEST - Resume interrupted Small-DiT tasks as array 17644604

- Rechecked the initial array before submission. Tasks 0-15 were the only
  terminal rows: each had Slurm exit `75:0`, `run.json` state `interrupted`
  with `error=null`, and a present exact recovery state. Tasks 16-35 remained
  `RUNNING` and were not selected.
- Submitted unthrottled resume array `17644604` as tasks `0-15` under
  `berzelius-2026-78`, with the original one-million-step target. Each task
  reuses its frozen source, config, data bundle, run directory, newest exact
  state, and W&B identity. Each allocation again requests one thin A100-40GB,
  four CPUs, 32 GB, 24 hours, no automatic requeue, and a five-minute
  `SIGUSR1` warning.
- Slurm accepted the array at 15:05:51 CEST and started all 16 tasks at
  15:05:53. The 20 surviving initial tasks also remained running, so all 36
  matrix rows were active. The submission is recorded in
  `recovery_jobs.tsv` with target step 1,000,000.
- Next check: verify attempt count two, exact restored steps, reused W&B run
  IDs, finite continued losses, and refreshed recovery states for every resumed
  task. The still-running initial tasks should reach their first USR1 boundary
  between about 17:03 and 17:53 CEST.


### 2026-09-27 15:09 CEST - Exact resume startup verified

- All 16 tasks in resume array `17644604` remain `RUNNING`. Every `run.json`
  records attempt count two, `state=training`, `error=null`, and an active W&B
  identity. Each second attempt names resume array `17644604`.
- Every task loaded the exact step recorded by its first interrupted attempt:
  275,882-278,249 across tasks 0-15. All 16 stderr logs report that W&B is
  resuming the existing named run, and every run retains its original run ID.
- Every task advanced beyond its restored step and logged a finite continued
  loss. The resumed stdout/stderr scan is empty for traceback, OOM, CUDA error,
  nonfinite failure, `RuntimeError`, `AssertionError`, or exception.
- Together with the 20 surviving tasks in initial array `17641282`, all 36
  matrix rows are running. The resumed tasks should receive their next
  five-minute warning at about 15:01 CEST on 2026-09-28 if they do not reach
  one million steps first.


### 2026-09-27 18:43 CEST - All first allocations reached exact recovery boundaries

- Initial array `17641282` has no live tasks. Its remaining tasks 16-35 stopped
  between 17:02:57 and 17:52:59 CEST after 23:54-23:55. Like tasks 0-15,
  Slurm records `FAILED` with exit `75:0`; this is the frozen wrapper's expected
  result after the five-minute `SIGUSR1` recovery path.
- All 20 new terminal stdout logs record interruption by `SIGUSR1`. Their
  `run.json` files record `state=interrupted` and `error=null`, and all 20 W&B
  runs closed and synced. Exact terminal recovery steps are 276,170-277,420
  for the two N=1024 rows, 272,854-277,792 for the nine N=2048 rows, and
  276,694-278,297 for the nine N=4096 rows.
- All 16 tasks in resume array `17644604` remain `RUNNING` in attempt two.
  Their latest logged steps are 317,750-320,250, and their newest durable
  recovery states are at steps 315,000-320,000. Their run records remain in
  `state=training` with `error=null` and active original W&B identities.
- The complete initial-plus-resume log scan is empty for traceback, OOM, CUDA
  error, nonfinite failure, `RuntimeError`, `AssertionError`, or exception.
  No resume allocation has been submitted for tasks 16-35; the only recorded
  resume remains array `17644604` for tasks 0-15.


### 2026-09-27 18:46 CEST - Resume remaining Small-DiT tasks as array 17645021

- Rechecked tasks 16-35 immediately before submission. All 20 remained in
  `state=interrupted` with `error=null`, present exact recovery states, and
  preserved W&B run IDs. Resume array `17644604` for tasks 0-15 remained
  healthy and was not modified.
- Submitted unthrottled resume array `17645021` as tasks `16-35` under
  `berzelius-2026-78`, with the original one-million-step target. Each task
  reuses its frozen source, config, data bundle, run directory, newest exact
  state, and W&B identity. Each allocation requests one thin A100-40GB, four
  CPUs, 32 GB, 24 hours, no automatic requeue, and a five-minute `SIGUSR1`
  warning.
- Slurm accepted the array at 18:46:01 CEST and started all 20 tasks at
  18:46:02. Together with the 16 tasks in array `17644604`, all 36 matrix rows
  were running in their second allocations. The new submission is recorded in
  `recovery_jobs.tsv` with target step 1,000,000.
- Next check: verify attempt count two, exact restored steps, reused W&B run
  IDs, finite continued losses, and clean logs for all tasks 16-35.


### 2026-09-27 18:48 CEST - Remaining exact-resume startups verified

- All 20 tasks in resume array `17645021` remain `RUNNING`. Every `run.json`
  records attempt count two, `state=training`, `error=null`, and an active W&B
  identity. Each second attempt names resume array `17645021`.
- Every task loaded the exact step recorded by its first interrupted attempt:
  272,854-278,297 across tasks 16-35. All 20 stderr logs report that W&B is
  resuming the existing named run, and every run retains its original run ID.
- Every task advanced beyond its restored step and logged a finite continued
  loss. The new resume stdout/stderr scan is empty for traceback, OOM, CUDA
  error, nonfinite failure, `RuntimeError`, `AssertionError`, or exception.
- Resume arrays `17644604` and `17645021` now cover the complete matrix, with
  all 36 rows running in attempt two. Tasks 0-15 should receive their next
  five-minute warning at about 15:01 CEST on 2026-09-28; tasks 16-35 should
  receive theirs at about 18:41 CEST if they do not reach one million first.


### 2026-09-27 18:51 CEST - Launch 36 fresh MNIST diagonal trajectories

- Submitted 18 fresh MNIST32 U-Net-S tasks as thin array `17645048`.
  Each task requests one A100-40GB, four CPUs, 32 GB, and 36 hours and
  trains directly toward one million updates.
- Submitted 18 fresh MNIST64 U-Net-L tasks as fat stage-1 array `17645049`.
  Each task requests one A100-80GB, eight CPUs, 128 GB, and 30 hours.
  Dependent matching arrays `17645050`, `17645051`, `17645052`, and
  `17645053` resume to 400k, 600k, 800k, and one million updates.
- Completion verifier `17645054` waits for the thin array and final fat
  stage. All 36 trajectories use new run directories, data bundles,
  initializations, noise banks, recovery states, and W&B run IDs.
- Physical/effective batch is 256/256 with accumulation one. All runs use
  W&B project `mnist-unet-diagonal-1m`.
- Frozen batch:
  `outputs/job_sets/mnist_diagonal_1m/20260927T165106Z_1991997`.
  Every source, job, config, manifest, and environment snapshot hash passed.
- Initial state: thin tasks 0-11 are running; thin tasks 12-17 are pending
  for resources; fat stage 1 is pending for priority; later stages and the
  verifier are dependency-pending.
- Next check: verify all 36 resolved configs, parameter counts, paired data
  identities, W&B starts, finite first losses, step-0 monitoring, and
  recovery state after the fat tasks and remaining thin tasks start.


### 2026-09-27 18:57 CEST - Fresh MNIST startup audit clean for 35 tasks

- All 18 MNIST64 tasks in fat stage-1 array `17645049` are running. Thin
  tasks 0-16 in array `17645048` are running; task 17 remains pending only
  for resources, with Slurm estimating a 23:05 CEST start.
- Audited every persisted artifact for the 35 started trajectories. All 35
  have distinct data-bundle and initialization files, unique active W&B run
  IDs, matching bundle and recovery hashes, `error=null`, finite first losses,
  and the full checkpoint/generated-grid/f_mem artifact set at step zero.
- All 35 resolved configs match physical/effective batch 256/256,
  accumulation one, deterministic FP32 with TF32 enabled, the exact 50-step
  Heun sampling contract, and W&B project `mnist-unet-diagonal-1m`.
  Parameter counts are 3,917,409 for MNIST32 U-Net-S and 23,773,889 for
  MNIST64 U-Net-L.
- The 11 complete resolution/N/seed cells have identical source IDs and
  bundle contents across `x/x`, `eps/eps`, and `v/v`. Cross-resolution source
  IDs match, and every available N=512 set is the prefix of its N=2048 set.
  The twelfth cell awaits only thin task 17.
- The complete current Slurm stdout/stderr scan is empty for traceback, OOM,
  CUDA error, nonfinite failure, `RuntimeError`, `AssertionError`, exception,
  or W&B error. Latest durable recovery steps span 500-2,500.


### 2026-09-27 18:59 CEST - Fresh MNIST live status checkpoint

- Thirty-five of 36 trajectories are running: all 18 MNIST64 tasks and
  MNIST32 tasks 0-16. MNIST32 task 17 remains pending for resources; Slurm's
  current estimated start is 23:05 CEST.
- All 35 started runs record `state=training`, `error=null`, and active unique
  W&B identities. MNIST32 has logged and durably recovered through steps
  2,500-5,000; all MNIST64 tasks have logged and recovered through step 1,000.
- The current stdout/stderr scan remains empty for traceback, OOM, CUDA error,
  nonfinite failure, `RuntimeError`, `AssertionError`, exception, or W&B error.
- MNIST64 continuation arrays `17645050`-`17645053` and completion verifier
  `17645054` remain correctly dependency-pending.


### 2026-09-27 21:12 CEST - Launch fresh CelebA32 U-Net-S full 3x3 array 17645398

- Submitted all 108 fresh trajectories as one unthrottled array `0-107` under
  account `berzelius-2026-78`. The matrix is N={512,1024,2048,4096}, seeds
  0-2, and every prediction/loss cell in `{x,eps,v}` squared.
- Every task targets one million optimizer updates and requests one thin
  A100-40GB, four CPUs, 32 GB, and 24 hours. Physical/effective batch is
  256/256 with accumulation one. A five-minute `SIGUSR1` warning saves exact
  state for a manually audited resume; automatic requeue is disabled.
- U-Net-S has 3,917,409 parameters. Training uses Adam at 1e-4, unit weighting,
  deterministic FP32 with TF32 disabled, and exact target conversions.
  Monitoring uses safe division with floor 0.05, `eps_to_v=via_x`, 50-step
  full Heun, `x_t_max`, no final Euler, and no final clamp.
- All rows are new. Within each N/seed cell the nine targets have paired source
  IDs and random-stream conventions; smaller N selections are prefixes of
  N=4096. W&B project is `celeba32-unet-3x3-1m` with strict resume identity.
- Frozen batch:
  `outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324`.
  The snapshot includes source, config, job files, environment, and SHA-256
  hashes. Matrix/config/model validation, shell/Python syntax, dry run, and
  56 focused model/runner tests passed before submission.
- Initial scheduler state: all 108 tasks are `PENDING (Resources)`. Next check:
  verify frozen imports, resolved configs, unique W&B starts, finite first
  losses, paired data identities, step-zero artifacts, and recovery state as
  tasks start.


### 2026-09-28 14:15 CEST - Overnight production status audit

- Slurm has 180 production trajectories running: all 36 resumed Small-DiT
  rows, all 18 MNIST32 rows, all 18 MNIST64 stage-2 rows, and all 108
  CelebA32 U-Net-S rows. No production trajectory is currently resource-
  pending. MNIST64 stages 3-5 and the completion verifier remain correctly
  dependency-pending.
- The only 36 `FAILED` records are the first Small-DiT allocations from array
  `17641282`. Every one ran for 23:54-23:55 and exited `75:0` through the
  documented `SIGUSR1` exact-recovery path. There is no short Slurm failure in
  any submitted production array.
- Small-DiT arrays `17644604` and `17645021` are healthy in attempt two. All
  36 records have `state=training`, `error=null`, finite current losses, unique
  active W&B identities, and durable recovery states at steps 495,000-545,000.
  Tasks 0-15 approach their next planned wall boundary at about 15:01 CEST;
  tasks 16-35 approach it at about 18:41 CEST.
- All 18 MNIST64 stage-1 tasks completed normally at exactly step 200,000 with
  Slurm exit `0:0` after 18:55-19:18. Stage-2 array `17645050` then started all
  18 correlated rows from exact step 200,000 using attempt two and the original
  W&B identities. This is the planned five-stage chain, not a crash.
- All 18 MNIST32 tasks remain healthy in their first allocations. Durable
  recovery states span steps 885,000-890,000, and current finite histories
  reach step 895,000. All 108 CelebA32 tasks also remain healthy in their first
  allocations; their staggered starts explain the wider durable range of
  70,000-490,000 steps.
- Across the three production batches, every run record has `error=null`; all
  180 active loss histories are finite. The log scan found no traceback, CUDA
  OOM, nonfinite failure, runtime/assertion error, disk/quota failure, killed
  process, or W&B error. W&B identities are unique and active locally: 36/36
  Small-DiT, 36/36 MNIST, and 108/108 CelebA32, with no recorded warnings.
- The CelebA64 RGB diagonal package remains prepared but unsubmitted. No jobs
  were submitted, cancelled, resumed, or otherwise modified during this
  read-only audit.


### 2026-09-28 14:23 CEST - Resume readiness check

- Nothing needs resuming yet. Every active production task remains `RUNNING`.
- Small-DiT tasks 0-15 are expected to reach their planned stop at about
  15:01 CEST; tasks 16-35 at about 18:41 CEST. These will need manually
  audited resume submissions after their exact recovery states are verified.
- MNIST64 continuation is already automatic: stage 2 is running, and stages
  3-5 will start in order. MNIST32 needs no resume while it continues toward
  one million steps.
- CelebA32 remains fully running. Its first planned stops should begin around
  21:12 CEST tonight, followed by staggered stops through about 11:39 CEST
  tomorrow. No jobs were changed during this check.


### 2026-09-28 14:30 CEST - Rigorous early-crash audit passed

- Cross-checked Slurm accounting, all 180 active `run.json` records, all
  training histories and recovery indexes, W&B identities, step-zero
  monitoring artifacts, and every current Slurm stdout/stderr file.
- No task stopped early. The only 36 failed production records are the first
  Small-DiT attempts: all ran 23:54-23:55, exited `75:0`, and have matching
  `SIGUSR1` interruption records. The only 18 completed production records are
  MNIST64 stage 1: all exited `0:0` at exactly step 200,000 and all 18 stage-2
  attempts resumed from step 200,000 with their original W&B identities.
- All 180 active runs remain `state=training` with `error=null`. Every history
  has strictly increasing steps and finite losses; every recovery index and
  referenced state file is present and non-empty. Current history ranges are
  Small-DiT 495k-550k, MNIST 200k-905k, and CelebA32 75k-500k.
- Artifact counts are exact: 36/36 Small-DiT, 36/36 MNIST, and 108/108
  CelebA32 for run records, recovery indexes, and step-zero monitoring.
  The full log scan found zero traceback, OOM, nonfinite, runtime/assertion,
  disk/quota, killed-process, W&B-error, or node/preemption signatures.
- Conclusion: nothing crashed before its declared stop. Nothing needs resuming
  at this timestamp. No job was submitted, cancelled, or modified.


### 2026-09-28 15:17 CEST - Launch CelebA64 RGB diagonal production block

- Repeated the write-free launch validation immediately before submission. It
  passed the complete 18-row matrix, real-data contract, paired and nested
  source identities, 24,386,627-parameter model, batch 256/256, exact training,
  72 checkpoints, and safe via-x monitoring checks. No earlier submitted batch
  existed for this job set.
- Submitted all 18 fresh trajectories as fat A100-80GB array `17654914`, with
  planned stop step 200,000 and a 30-hour limit. Submitted the automatic
  task-matched continuation arrays `17654915`, `17654916`, `17654917`, and
  `17654918`, ending at 400k, 600k, 800k, and 1M respectively. Completion
  verifier `17654919` waits for successful completion of stage 5.
- Frozen batch:
  `outputs/job_sets/celeba64_rgb_diagonal_1m/20260928T131627Z_3579761`.
  Every frozen source, job, config, manifest, and environment hash passed.
- Initial scheduler state: stage 1 is `PENDING (Priority)` with an estimated
  start at 15:41:44 CEST. Stages 2-5 and the verifier are waiting for their
  required earlier jobs. No existing DiT, MNIST, or CelebA32 job was modified.
- Next check: verify all 18 frozen imports, RGB `[3,64,64]` data, resolved
  configs, unique W&B starts, finite first losses, step-zero monitoring, and
  recovery states after stage 1 starts.


### 2026-09-28 15:20 CEST - Production program status after RGB launch

- Small-DiT tasks 0-15 reached their second planned wall boundary at
  15:00:42-15:00:46. All 16 ran for 23:54:49-23:54:53, exited `75:0`, recorded
  `received SIGUSR1`, and saved exact recovery states at steps 551,933-556,209.
  Their run records have `error=null`; the logs contain no training error.
  They are ready for a user-approved third allocation but were not resumed.
- Small-DiT tasks 16-35 remain running in attempt two with current histories
  at steps 505,000-515,000. Their next planned boundary is about 18:41 CEST.
- All 18 MNIST32 runs remain active at steps 935,000-945,000. All 18 MNIST64
  runs are active in stage 2 at steps 210,000-215,000; stages 3-5 remain queued
  to follow automatically.
- All 108 CelebA32 runs remain active, with staggered progress from 100,000 to
  525,000 updates. Their first planned wall boundaries begin around 21:12 CEST.
- CelebA64 RGB stage 1 array `17654914` remains queued for priority, with the
  current estimated start at 15:41:44 CEST. Its four continuation arrays and
  final verifier remain correctly waiting behind it. No RGB run directory or
  W&B identity exists yet because no stage-1 task has started.
- The scientific and operating design remains documented in
  `jobs/berzelius/celeba64_rgb_diagonal_1m/README.md`; the frozen launch record
  is `outputs/job_sets/celeba64_rgb_diagonal_1m/20260928T131627Z_3579761`.
  Apart from the approved RGB submission, no job was launched, resumed,
  cancelled, or modified during this check.


### 2026-09-28 16:06 CEST - Resume 16 stopped Small-DiT trajectories

- Rechecked tasks 0-15 after their second planned wall boundary. All 16 still
  had `state=interrupted`, `error=null`, matching `SIGUSR1` attempt records,
  present exact recovery states at steps 551,933-556,209, and clean logs.
- Submitted third-allocation array `17655136` for exactly tasks 0-15, targeting
  the unchanged absolute one-million-step endpoint. It reuses each frozen run
  directory, newest exact state, and W&B identity on a thin A100-40GB with a
  24-hour limit and five-minute recovery signal.
- Initial state is `PENDING (Priority)` with Slurm currently estimating a
  2026-09-29 02:01:23 CEST start. The other 20 Small-DiT tasks remain running
  in array `17645021` and were not included or modified.
- CelebA64 RGB stage 1 has meanwhile started tasks 0-10; tasks 11-17 remain
  queued for priority. No MNIST or CelebA32 job was changed.


### 2026-09-28 16:10 CEST - CelebA64 RGB startup audit

- Audited all 11 stage-1 tasks that had started from array `17654914`
  (tasks 0-10). Every run is training with `error=null`; tasks 11-17 remain
  queued for priority and have not failed.
- All started runs use actual float32 tensors shaped `[N,3,64,64]` and held-out
  references shaped `[1000,3,64,64]`. The resolved configuration's
  `celeba32_grey` value is the legacy loader key; `image_channels=3` and
  `image_size=64`, and the tensor manifests prove the launched data are RGB64.
- All 11 runs have the exact 24,386,627-parameter model, batch 256/256 with
  accumulation 1, unit weighting, the intended diagonal prediction/loss
  spaces, all 72 checkpoint steps, and the frozen Heun-50 via-x sampler.
- First and current losses are finite. Each run has the step-zero checkpoint,
  generated samples, comparison grid, current exact-recovery files, and an
  active unique W&B identity in `celeba64-rgb-unet-diagonal-1m`.
- The complete Slurm stdout/stderr scan found no traceback, CUDA OOM, disk or
  quota fault, nonfinite value, exception, failed assertion, or W&B error.
  Current logs have reached 6,750-7,250 updates. No job was changed by this
  read-only audit.


### 2026-09-28 16:16 CEST - Live status after approved resume

- New Small-DiT resume array `17655136` remains queued for priority for exactly
  tasks 0-15; Slurm estimates a 02:01 CEST start on 2026-09-29.
- Small-DiT tasks 16-35, all 18 MNIST32 tasks, all 18 MNIST64 stage-2 tasks,
  all 108 CelebA32 tasks, and CelebA64 RGB tasks 0-10 remain running.
- CelebA64 RGB tasks 11-17 are queued for priority with a current 02:34 CEST
  estimate. RGB stages 2-5 and the MNIST64 later stages are correctly waiting
  for their earlier stages to finish successfully.
- No other stopped production task currently needs a resume submission.


### 2026-09-28 16:46 CEST - Launch native-v/v weighting controls

- Built the dedicated nine-run CelebA32 U-Net-S block at N=512. It crosses
  seeds 0-2 with raw `x_equivalent`, `unit`, and `eps_equivalent` weights while
  keeping prediction and loss spaces fixed at native `v/v`.
- Validation passed for all nine unique rows, the exact raw formulas, the
  3,917,409-parameter model, batch 256/256, exact training, 72 checkpoints,
  safe via-x full-Heun monitoring, shell/Python syntax, and dry-run submission.
- All 56 focused model and runner tests passed in `mem_param_env`. Ruff was not
  installed in that environment; the optional Ruff check did not run and the
  first launch command stopped before creating a batch or submitting a job.
- Submitted unthrottled thin-A100 array `17655496`, tasks 0-8, with a 24-hour
  limit, five-minute recovery signal, and the absolute one-million-step target.
- Frozen batch:
  `outputs/job_sets/celeba32_unet_vv_weighting_1m/20260928T144550Z_222375`.
- W&B project is `celeba32-unet-vv-weighting-1m`. Continuation after a planned
- Every frozen source, job, config, manifest, environment, and top-level hash
  passed after submission.
- Initial scheduler state is `PENDING (Priority)` for all nine tasks. Slurm's
  current estimated start is 2026-10-01 10:23 CEST; this estimate can move.
- No existing DiT, MNIST, CelebA32 3x3, or CelebA64 job was modified.


### 2026-09-28 16:52 CEST - False crash display for CelebA32 task 97

- Investigated `celeba32-unetS-n2048_v_eps_s2-B256-Beff256-1m`, task 97 in
  array `17645398`, after the W&B dashboard appeared to show a crash.
- Slurm reports the task `RUNNING` on node001 with no exit or stop reason. The
  worker continued through step 147,750 during the audit.
- Local state is healthy: `run.json` has `state=training` and `error=null`, all
  inspected losses are finite, and both exact-recovery slots are valid, with
  the newest durable state at step 145,000.
- The W&B API reports the exact run ID `9rjq1v1u` as `running`. No other run in
  the project has the same display name.
- W&B's internal log shows repeated project-level file-stream rate limits
  (`HTTP 429`) from 16:32 through 16:46 while 108 runs shared the project.
  Uploads recovered with `HTTP 200` at 16:47 and continued normally afterward.
- Conclusion: training did not crash. The visible state was stale during the
  W&B upload backlog. No resume, cancellation, or job change was made.


### 2026-09-28 17:13 CEST - Launch N=2048 native-v/v weighting extension

- Built a separate nine-run CelebA32 U-Net-S weighting package at N=2048. The
  already-submitted N=512 frozen batch was not modified.
- The matrix crosses seeds 0-2 with native `v/v` and raw `x_equivalent`,
  `unit`, and `eps_equivalent` weights. Every trajectory is fresh and targets
  one million optimizer updates.
- The validator proved that each submitted N=512 source selection is exactly
  the prefix of its matching N=2048 selection. Within N=2048 and seed, the
  three weighting rows share initialization, data order, random streams, and
  monitoring noise by construction while keeping independent files.
- Shell/Python syntax, all nine rows, the raw formulas, the 3,917,409-parameter
  model, batch 256/256, 72 checkpoints, exact training, safe via-x full-Heun
  monitoring, three focused weighting tests, and dry-run submission passed.
- Submitted unthrottled thin-A100 array `17655777`, tasks 0-8, with a 24-hour
  limit, five-minute recovery signal, and the absolute one-million-step target.
- Frozen batch:
  `outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/20260928T151300Z_373224`.
- W&B remains in `celeba32-unet-vv-weighting-1m` with a separate N=2048 group.
- Every frozen N=2048 source, job, config, manifest, environment, and top-level
  hash passed after submission.
- Both weighted arrays, N=512 `17655496` and N=2048 `17655777`, are currently
  `PENDING (Priority)`. Slurm estimates 2026-10-01 10:23 CEST for both.
- Neither weighted array has started or failed. No existing job was changed.


### 2026-09-29 12:11 CEST - Full production status audit

- Cross-checked live `squeue`/`sacct` state against every `run.json`, complete
  loss history, step-zero sample, and exact-recovery index in all six active
  production batches. The complete Slurm log scan found no traceback, CUDA
  OOM, disk/quota fault, nonfinite value, assertion error, or runtime error.
- There are 70 tasks currently running and no real training crash.
- Small-DiT: resume array `17655136` is running tasks 0-15 in attempt three at
  steps 755,000-765,000. Tasks 16-35 ended attempt two normally at their
  planned wall boundary and are safely stopped at steps 548,257-557,023; all
  20 now need a user-approved manual resume.
- MNIST: all 18 MNIST32 trajectories finished successfully at 1,000,000.
  All 18 MNIST64 trajectories are running stage 3 at steps 425,000-435,000;
  stages 4-5 and the final verifier remain submitted and will follow
  automatically after successful earlier stages.
- CelebA32 U-Net 3x3: all 108 trajectories reached their first wall boundary
  and are safely stopped with exact states at steps 690,721-699,716. Slurm
  recorded 103 controlled `75:0` exits and five `TIMEOUT` rows, but all 108
  local records say `interrupted` by `SIGUSR1`, have `error=null`, and have
  present recovery files. All 108 now need a user-approved manual resume.
- CelebA64 RGB: 11 trajectories completed the planned 200,000-step stage 1 and
  are running stage 2 at step 205,000. The other seven are still running stage
  1 at step 180,000. Later per-task stages are correctly waiting for their own
  prior stage; no task failed.
- Native-v/v weighting controls: all nine N=512 tasks are running at step
  510,000, and all nine N=2048 tasks are running at steps 490,000-510,000.
  Both arrays are healthy and have finite histories and current recovery files.
- Five CelebA32 3x3 records retain a stale W&B `active=true` flag because Slurm
  killed them at the hard boundary immediately after their signal recovery;
  this is not a training failure and their existing identities can be resumed.
- No job was submitted, resumed, cancelled, or otherwise changed during this
  read-only audit.


### 2026-09-29 12:24 CEST - Resume all currently stopped production runs

- Rechecked the live queue immediately before submission. The selected 20
  Small-DiT tasks (16-35) and all 108 CelebA32 U-Net 3x3 tasks were stopped;
  none was already queued or running.
- Hash-validated the newest exact recovery payload for every selected run.
  All 128 files match their recorded SHA-256 values; all run records are
  interrupted, have `error=null`, and remain below one million updates.
- Submitted Small-DiT resume array `17661660` for tasks 16-35 from exact steps
  548,257-557,023 to the unchanged 1,000,000-step target. It uses the frozen
  batch, original W&B identities, thin A100-40GB tasks, 24-hour limit, and
  five-minute recovery signal.
- Submitted CelebA32 U-Net 3x3 resume array `17661664` for tasks 0-107 from
  exact steps 690,721-699,716 to the unchanged 1,000,000-step target, with the
  same frozen batch, original W&B identities, thin A100-40GB task shape,
  24-hour limit, and five-minute recovery signal.
- At 12:24 CEST, DiT tasks 16-23 were running and tasks 24-35 were waiting for
  GPUs. Every started DiT record has attempt count three, `state=training`,
  `error=null`, the exact restored step, an active original W&B identity, and
  finite loss. The new logs contain no traceback, OOM, disk/quota error,
  nonfinite failure, assertion error, or runtime error.
- All 108 CelebA32 tasks were waiting for GPUs. Slurm's current estimated start
  is 18:43 CEST; this is an estimate and may move.
- Existing running DiT tasks 0-15, MNIST, CelebA64 RGB, and both weighted-v/v
  arrays were not modified.


### 2026-09-29 12:27 CEST - Post-resume status check

- There are 78 production tasks running and no real training failure.
- Small-DiT: the older resume wave has all 16 tasks running at steps
  755,000-770,000. New array `17661660` has tasks 16-23 running from their
  exact states and tasks 24-35 waiting for GPUs; Slurm currently estimates
  13:54 CEST for the waiting tasks.
- CelebA32 U-Net 3x3 resume array `17661664` has all 108 tasks accepted and
  waiting for GPUs. Slurm's current estimated start is 18:17 CEST.
- MNIST32 remains complete at one million. All 18 MNIST64 tasks are running
  stage 3 at steps 430,000-435,000, with later stages waiting automatically.
- All 18 CelebA64 RGB tasks are running: 11 in stage 2 at step 205,000 and
  seven in stage 1 at step 180,000.
- Both native-v/v weighting arrays are healthy: all nine N=512 tasks are at
  step 515,000 and all nine N=2048 tasks are at steps 500,000-515,000.
- Across all production runs, recorded errors, nonfinite histories, and
  missing recovery files are all zero. The new resume-log scan is also clean.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-29 14:16 CEST - Production status check

- There are 96 production tasks running and no real training failure.
- All 36 Small-DiT trajectories are now running. Tasks 0-15 are at steps
  775,000-790,000 and newly resumed tasks 16-35 are at steps 560,000-575,000.
- CelebA32 U-Net 3x3 resume array `17661664` has tasks 0-5 running cleanly at
  steps 696,250-710,000. Tasks 6-107 are waiting for GPUs; Slurm currently
  estimates 15:02 CEST, which may move.
- MNIST32 remains complete at one million. All 18 MNIST64 trajectories are
  running stage 3 at steps 450,000-455,000.
- All 18 CelebA64 RGB trajectories are now running stage 2, at steps
  200,000-225,000; later stages remain queued to follow automatically.
- Both weighted-v/v arrays remain healthy. N=512 is at steps 565,000-570,000
  and N=2048 is at steps 550,000-565,000.
- Across all production runs, recorded errors, nonfinite histories, and
  missing recovery files remain zero. The new DiT and CelebA32 resume logs
  contain no traceback, OOM, disk/quota, assertion, or runtime error.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-29 14:44 CEST - Launch seed-1 Small-DiT robustness matrix

- Built and validated 18 fresh CelebA32-gray Small-DiT trajectories: seed 1,
  N={512,2048}, and the complete prediction/loss 3x3.
- The scientific config exactly matches the running seed-0 DiT block except
  for approved seed and tracking-identity fields. All rows retain the
  32,563,972-parameter model, batch 64/256, one million updates, 72 checkpoints,
  exact training, and safe via-x full-Heun monitoring.
- Real-data validation proved prefix-nested N=512/N=2048 source identities and
  fresh selection relative to seed 0. All 18 W&B names are unique. Shell and
  Python syntax, dry-run submission, 68 focused tests, and every frozen hash
  passed.
- Submitted unthrottled thin-A100 array `17662195`, tasks 0-17, with a 24-hour
  limit and five-minute exact-recovery signal. Frozen batch:
  `outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178`.
- Initial scheduler state is waiting for priority with no start estimate. No
  existing DiT, MNIST, CelebA32, CelebA64, or weighted-v/v job was modified.


### 2026-09-30 10:31 CEST - Full production status audit

- Cross-checked live Slurm state against every frozen batch's `run.json`, loss
  history, step-zero monitoring output, exact-recovery index, and recovery
  files. The production total is 234 trajectories: 137 running, 56 complete,
  34 safely stopped at the planned wall-time boundary and ready for manual
  resume, and seven CelebA64 RGB trajectories automatically waiting for GPUs.
- Small-DiT seed 0: tasks 0-15 stopped cleanly at steps 829,500-834,250 with
  exact recovery at 829,718-834,307 and now need manual resume. Tasks 16-35
  are running at steps 795,000-810,000; their planned stop signals are due
  approximately 12:16-13:26 CEST today if they have not finished first.
- Small-DiT seed 1: all 18 tasks are running cleanly at steps 55,000-60,000.
- MNIST: all 18 MNIST32 tasks remain complete at one million; all 18 MNIST64
  tasks are running automatic stage 4 at steps 660,000-670,000.
- CelebA32 U-Net 3x3: 38 tasks are complete at one million and 70 are running
  at steps 855,000-915,000.
- CelebA64 RGB: stage 2 completed all 18 at 400,000. Eleven are running stage 3
  at steps 420,000-425,000 and seven are safely paused at 400,000 waiting for
  GPUs; their already-submitted later stages will follow automatically.
- Both native-v/v weighting blocks stopped cleanly at the planned boundary and
  need manual resume: all nine N=512 tasks have recovery at steps
  695,871-699,137, and all nine N=2048 tasks at 695,598-698,330.
- There are zero real crashes. Every recorded run error is null; all checked
  losses are finite; recovery and step-zero artifacts are present; and a scan
  of all seven production batches found no traceback, OOM, disk/quota,
  non-finite, assertion, runtime, or segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this audit.


### 2026-09-30 10:47 CEST - Resume all 34 stopped production trajectories

- Rechecked the live queue and selected exactly the stopped, sub-million runs:
  Small-DiT seed-0 tasks 0-15 and all nine tasks in each native-v/v weighting
  block. None of the 34 was already running or queued.
- Verified every selected `run.json` is interrupted with `error=null` and
  SHA-256 checked all 34 newest exact-recovery payloads before submission.
- Submitted Small-DiT resume array `17674243`, tasks 0-15, from exact steps
  829,718-834,307 to the unchanged one-million-step target.
- Submitted weighted-v/v N=512 resume array `17674244`, tasks 0-8, from exact
  steps 695,871-699,137 to the unchanged one-million-step target.
- Submitted weighted-v/v N=2048 resume array `17674245`, tasks 0-8, from exact
  steps 695,598-698,330 to the unchanged one-million-step target.
- All three unthrottled thin-A100 arrays were accepted and are waiting for GPUs.
  They retain their frozen source, run directories, data, and W&B identities.
- No running or already-queued DiT, MNIST, CelebA32, or CelebA64 job was changed.


### 2026-09-30 13:05 CEST - Production status check

- Across 234 production trajectories, 124 are running, 56 are complete at one
  million, 41 are stopped or paused with continuation jobs already waiting for
  GPUs, and 13 newly stopped Small-DiT runs now need a resume submission.
- Small-DiT seed 0 tasks 16-28 reached their planned 24-hour boundary with
  exit 75, `received SIGUSR1`, `error=null`, and exact recovery at steps
  825,737-835,235. All 13 recovery payload hashes passed. Tasks 29-35 remain
  running at steps 825,000-830,000 and will stop shortly if unfinished.
- Previously submitted arrays `17674243`, `17674244`, and `17674245` remain
  accepted and waiting for GPUs: 16 older seed-0 DiTs and both nine-run
  weighted-v/v blocks.
- Small-DiT seed 1 has all 18 running at steps 85,000-90,000. MNIST32 remains
  18/18 complete; MNIST64 has all 18 running at steps 690,000-700,000.
- CelebA32 U-Net 3x3 remains 38 complete and 70 running at steps
  925,000-990,000. CelebA64 RGB has 11 running stage 3 at steps
  445,000-450,000 and seven waiting for GPUs at 400,000.
- Every recorded error remains null and every checked loss is finite. The full
  production-log scan found no traceback, OOM, disk/quota, non-finite,
  assertion, runtime, or segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-30 13:15 CEST - Resume newly stopped seed-0 DiTs

- Rechecked the live queue and found seed-0 Small-DiT tasks 16-29 stopped and
  not yet represented by a continuation job. Tasks 30-35 were still running;
  all previously stopped DiTs and weighted-v/v runs already had queued jobs.
- All 14 selected records have `state=interrupted`, `error=null`, and
  `stop_reason=received SIGUSR1`. Their newest exact recovery payloads at steps
  825,737-835,235 passed SHA-256 verification.
- Submitted unthrottled thin-A100 resume array `17675549`, tasks 16-29, to the
  unchanged absolute one-million-step target with the frozen source, original
  run directories, and original W&B identities.
- Slurm accepted the complete array; all 14 tasks are waiting for GPUs. At this
  timestamp every stopped production trajectory has a continuation job.
- No running task or existing queued job was changed or duplicated.


### 2026-09-30 13:47 CEST - Production status check

- Across 234 trajectories, 82 are running, 91 are complete at one million, 55
  are stopped or paused with continuation jobs waiting for GPUs, and six newly
  stopped seed-0 Small-DiTs still need a resume submission.
- All 36 seed-0 DiTs have now reached the planned boundary. Tasks 0-15 are in
  queued array `17674243`, tasks 16-29 are in queued array `17675549`, and
  tasks 30-35 stopped cleanly at exact steps 829,604-835,088 without a new job.
  The final six have `error=null`, the expected SIGUSR1 reason, and verified
  recovery hashes.
- Small-DiT seed 1 has all 18 running at steps 95,000-100,000. MNIST32 remains
  18/18 complete and MNIST64 has all 18 running at steps 695,000-705,000.
- CelebA32 U-Net 3x3 is now 73 complete and 35 running at steps
  945,000-990,000. CelebA64 RGB has 11 running at steps 455,000-460,000 and
  seven waiting for GPUs at 400,000.
- Both nine-run weighted-v/v resume arrays remain accepted and waiting for
  GPUs. Every recorded error is null, every checked loss is finite, and the
  complete log scan found no traceback, OOM, disk/quota, non-finite, assertion,
  runtime, or segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-30 14:34 CEST - Resume final six seed-0 DiTs

- Rechecked the queue and confirmed Small-DiT tasks 30-35 were interrupted,
  below one million, and not already represented by a continuation job.
- All six records have `error=null` and the expected SIGUSR1 stop reason. Their
  newest exact recovery payloads at steps 829,604-835,088 passed SHA-256 checks.
- The write-free submission preview passed, then resume array `17676367` was
  submitted for exactly tasks 30-35 to the unchanged one-million-step target.
- Slurm accepted all six tasks; they are waiting for GPUs. Task 0 from earlier
  array `17674243` has started, while the other seed-0 continuations remain
  queued. All 36 seed-0 DiTs now either run or have a resume job waiting.
- No other running or queued production job was changed or duplicated.


### 2026-09-30 14:45 CEST - Production status check

- Across 234 trajectories, 76 are running, 106 are complete at one million,
  and 52 have accepted continuation jobs waiting for GPUs. No stopped run is
  currently missing a resume job.
- Seed-0 Small-DiT tasks 0 and 1 have started from their exact states at about
  834,000-835,000; tasks 2-35 remain queued across arrays `17674243`,
  `17675549`, and `17676367`. All 18 seed-1 DiTs are running at
  105,000-110,000.
- MNIST32 remains 18/18 complete. All 18 MNIST64 runs are active at
  705,000-715,000.
- CelebA32 U-Net 3x3 is now 88 complete and 20 running at
  975,000-995,000.
- All 18 CelebA64 RGB trajectories are now running stage 3, spanning
  405,000-470,000. Both nine-run weighted-v/v continuation arrays remain
  accepted and waiting for GPUs.
- Every recorded error is null and every checked loss is finite. The full log
  scan found no traceback, OOM, disk/quota, non-finite, assertion, runtime, or
  segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-30 16:15 CEST - Production status check

- Across 234 trajectories, 57 are running, 126 have reached one million, and
  51 have accepted continuation jobs waiting for GPUs. No stopped trajectory
  is missing a resume job.
- CelebA32 U-Net 3x3 reached 108/108 trained trajectories. All 108 have
  `error=null`, successful W&B finishes, exact recovery at one million, a final
  one-million checkpoint, and exactly 72 checkpoints. The separate strict
  verifier has not yet been submitted, so `TRAINING_COMPLETE` is not present.
- Seed-0 Small-DiT tasks 0-2 are running at steps 840,000-850,000 and tasks
  3-35 are waiting for GPUs. All 18 seed-1 DiTs are running at step 125,000.
- MNIST32 remains 18/18 complete. All 18 MNIST64 trajectories are running at
  steps 720,000-730,000.
- All 18 CelebA64 RGB trajectories are running stage 3 at steps
  420,000-485,000. Both nine-run weighted-v/v continuation arrays remain
  accepted and waiting for GPUs.
- Every recorded error is null and every checked loss is finite. The complete
  production-log scan found no traceback, OOM, disk/quota, non-finite,
  assertion, runtime, or segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this check.


### 2026-09-30 17:00 CEST - Production status and resume audit

- Across 234 trajectories, 57 are running, 126 are complete at one million,
  and 51 have accepted continuation jobs waiting for GPUs.
- Checked every interrupted trajectory against the live queue. None lacks a
  resume job, so no duplicate or additional submission was made.
- Seed-0 Small-DiT tasks 0-2 are running at steps 845,000-860,000 and tasks
  3-35 are waiting for GPUs. All 18 seed-1 DiTs are running at
  130,000-135,000.
- CelebA32 U-Net 3x3 remains 108/108 trained. MNIST32 remains 18/18 trained;
  all 18 MNIST64 trajectories are running at steps 730,000-740,000.
- All 18 CelebA64 RGB trajectories are running at steps 425,000-490,000. Both
  nine-run weighted-v/v continuation arrays remain waiting for GPUs.
- Every recorded error is null and every checked loss is finite. The complete
  log scan found no traceback, OOM, disk/quota, non-finite, assertion, runtime,
  or segmentation-fault signature.
- No job was submitted, resumed, cancelled, or modified during this audit.


### 2026-10-01 12:29 CEST - Full audit after Berzelius project rollover

- Slurm confirms the old project ID `berzelius-2026-78` now has association limits
  `GrpJobs=0` and `GrpSubmitJobs=0`. The replacement project ID
  `berzelius-2026-269` is active with zero prior user usage.
- The rollover did not abruptly kill running work. Jobs continued for hours
  after midnight: all 36 seed-0 DiTs and both nine-run weighted-v/v blocks
  completed at one million with exit 0; all 18 RGB stage-3 tasks completed at
  600,000 with exit 0; and all 18 seed-1 DiTs reached their normal 24-hour
  SIGUSR1 boundary with exit 75. All 18 MNIST64 stage-5 tasks are still running.
- Across 234 production trajectories, 180 are trained at one million, 18
  MNIST64 trajectories are running at steps 935,000-940,000, 18 RGB64
  trajectories are safely paused at 600,000, and 18 seed-1 DiTs are safely
  interrupted at exact steps 275,903-278,296.
- Every one of the 54 unfinished trajectories has a present newest recovery
  payload whose SHA-256 matches its recorded index. All run errors are null,
  all inspected losses are finite, and the complete log scan found no crash,
  cancellation, OOM, disk/quota, or numerical-failure signature.
- Completed blocks are 108/108 CelebA32 U-Net 3x3, 36/36 seed-0 Small-DiT,
  18/18 MNIST32, 9/9 weighted-v/v N=512, and 9/9 weighted-v/v N=2048. Every
  finished row has a final checkpoint, exact recovery at one million, and a
  successful W&B finish.
- Jobs under the old project ID that have not started cannot start while the zero limits
  remain. This affects RGB64 stage 4 (`17654917`), stage 5 (`17654918`), and
  verifier (`17654919`), plus the MNIST verifier (`17645054`). The RGB stage-4
  tasks explicitly report `AssocGrpJobsLimit`.
- Remaining training is: seed-1 DiT from about 276k to 1M; RGB64 from 600k to
  1M; and the final 60k-65k of MNIST64, which is currently still progressing
  under the old project ID. No scheduler job was changed during this audit.


### 2026-10-01 12:38 CEST - Move pending work and resume under the new project ID

- Updated all 37 live Berzelius shell launchers from project ID
  `berzelius-2026-78` to `berzelius-2026-269`; `bash -n` passed for every
  updated launcher. Frozen batch snapshots and historical run artifacts were
  not modified.
- Moved the already-submitted RGB64 stage-4 array `17654917`, stage-5 array
  `17654918`, final verifier `17654919`, and MNIST64 verifier `17645054` to
  project ID `berzelius-2026-269` in place. Their job IDs and dependencies were
  preserved; no duplicate jobs were submitted.
- All 18 RGB64 stage-4 tasks started. Logs confirm exact recovery from step
  600,000, RGB shape `[3,64,64]`, the 24,386,627-parameter U-Net, physical and
  effective batch 256/256, and continuation of the original W&B identities.
  Stage 5 is waiting for stage 4; the final verifier is waiting for stage 5.
- Verified the 18 seed-1 DiT recovery payloads, completed the write-free resume
  preview, and submitted exact-resume array `17683429` to target one million.
  At this check, tasks 0-13 are running and tasks 14-17 are waiting for GPUs.
  A live log confirms exact recovery, the 32,563,972-parameter DiT, and physical
  and effective batches 64/256.
- Submitted strict completed-block verifiers `17683435` (CelebA32 U-Net 3x3),
  `17683436` (weighted v/v N=512), and `17683437` (weighted v/v N=2048). They
  are accepted under the new project ID and waiting for GPUs.
- The 18 MNIST64 stage-5 tasks remain running under the old project ID and were
  deliberately left untouched. Their verifier is now under the new project ID
  and waits for stage 5 to finish.
- A scan of every new RGB64 and seed-1 DiT log found no traceback, OOM,
  disk/quota, non-finite-loss, assertion, runtime, cancellation, or kill
  signature.


### 2026-10-01 12:41 CEST - Post-submission live check

- RGB64 stage 4 remains 18/18 running under project ID
  `berzelius-2026-269`.
- Seed-1 DiT resume array `17683429` has tasks 0-14 running; tasks 15-17 are
  accepted and waiting for GPUs.
- The three completed-block verifiers remain accepted and waiting for GPUs.
  RGB64 stage 5 waits for stage 4, and both final chained verifiers wait for
  their training stages.


### 2026-10-01 14:00 CEST - Production status check

- Across 234 production trajectories, 180 are trained at one million and all
  54 unfinished trajectories are running. No training trajectory is pending,
  stopped, or missing a continuation job.
- All 18 seed-1 Small-DiT tasks in resume array `17683429` are running under
  `berzelius-2026-269`. Their histories and exact-recovery indexes reached step
  290,000.
- All 18 CelebA64 RGB tasks in stage-4 array `17654917` are running under
  `berzelius-2026-269`. Their histories and exact-recovery indexes reached step
  610,000. Stage 5 (`17654918`) and verifier `17654919` remain dependency-
  pending.
- All 18 MNIST64 stage-5 tasks remain running under `berzelius-2026-78` at
  steps 950,000-955,000. Verifier `17645054` is dependency-pending under the
  replacement project ID.
- Completed-block verifiers `17683435`, `17683436`, and `17683437` finished
  with exit `0:0`. Their `TRAINING_COMPLETE` markers confirm the 108-row
  CelebA32 U-Net 3x3 block and both nine-row native-v/v weighting blocks.
- Every active `run.json` has `state=training` and `error=null`. All recorded
  losses are finite. The current DiT, MNIST64, and RGB64 Slurm logs contain no
  traceback, OOM, disk/quota, non-finite, assertion, runtime, cancellation, or
  kill signature.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  check.


### 2026-10-01 14:39 CEST - Full block and FID-readiness audit

- The production matrix contains eight blocks and 234 trajectories. Five
  completed blocks contain 180 trained trajectories; the other 54 trajectories
  are running cleanly.
- The completed CelebA32 U-Net 3x3, seed-0 Small-DiT 3x3, MNIST32 diagonal,
  and both native-v/v weighting blocks have all expected final checkpoints,
  final monitoring samples, saved data bundles, manifests, noise banks,
  one-million-step recovery states, and finite loss histories.
- Strict `TRAINING_COMPLETE` markers cover the 108-row U-Net block and both
  nine-row weighting blocks. The 36-row seed-0 DiT block has the same checked
  artifact presence but no dedicated strict verifier marker.
- Active seed-1 DiT rows reached steps 295,000-300,000, MNIST64 reached
  955,000-960,000, and RGB64 reached 620,000. Their run errors remain null,
  losses remain finite, recovery indexes are current, and current logs are clean.
- No production run has a post-hoc evaluation directory yet. The validated
  fixed-10k FID workflow is tied to the older epsilon-recovery matrix, although
  its frozen CelebA32 10k reference matches the new CelebA32 source catalog
  and preprocessing.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  check.


### 2026-10-02 10:33 CEST - Production status check

- Across 234 production trajectories, 198 are trained at one million and 36
  are running. No trajectory is stopped or waiting for a missing continuation.
- All 18 MNIST64 trajectories completed at one million with `error=null`,
  finite losses, five contiguous exact-resume attempts, and successful W&B
  finishes. Verifier `17645054` completed with exit `0:0` and wrote the shared
  MNIST `TRAINING_COMPLETE` marker for all 36 MNIST32/MNIST64 rows.
- All 18 CelebA64 RGB stage-4 tasks in `17654917` completed with exit `0:0` at
  the planned 800,000-step boundary. Stage-5 array `17654918` is 18/18 running
  under project ID `berzelius-2026-269`. Every row resumed from exactly 800,000,
  has active W&B, and has a current recovery state at 815,000; live stdout has
  progressed through about 818,000. Final verifier `17654919` waits for stage 5.
- All 18 seed-1 Small-DiT tasks in exact-resume array `17683429` are running
  under project ID `berzelius-2026-269`. Every row resumed from its matching
  SIGUSR1 recovery step and has active W&B. Recovery and history records span
  525,000-530,000; live stdout has progressed through about 533,000.
- The DiT allocation is near its planned 24-hour signal boundary around 12:35
  CEST. Inspect and resume it only after the tasks stop and their newest
  recovery hashes are verified.
- All latest recorded losses are numeric. The current DiT, RGB stage-4/stage-5,
  MNIST64 stage-5, and MNIST verifier logs contain no traceback, OOM,
  disk/quota, non-finite, assertion, runtime, cancellation, or kill signature.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  check.


### 2026-10-02 12:42 CEST - Production status and DiT recovery audit

- Across 234 production trajectories, 198 are trained at one million, 18
  CelebA64 RGB trajectories are running, and all 18 seed-1 Small-DiTs are
  safely interrupted and ready for another exact resume.
- Every task in DiT array `17683429` reached the planned signal boundary after
  about 23 hours 55 minutes and exited `75:0`. All 18 run records say
  `state=interrupted`, `reason=received SIGUSR1`, and `error=null`. Exact
  recovery steps span 552,194-556,621 and match the final attempt end steps.
- All DiT histories have numeric latest losses. The write-free resume
  validation passed for tasks 0-17, target one million, project ID
  `berzelius-2026-269`, thin A100, 24 hours, and the existing run identities.
- All 18 RGB stage-5 tasks in `17654918` remain running under the new project
  ID. Their histories span 835,000-840,000 with numeric latest losses. Final
  verifier `17654919` still waits for stage 5 to finish.
- The current DiT and RGB logs contain no traceback, OOM, disk/quota,
  non-finite, assertion, runtime, cancellation, or kill signature. Separate
  interactive job `17689102` is running and was not touched.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  check.


### 2026-10-02 12:57 CEST - Validate completed interactive MNIST encoder

- The researcher-run interactive output is present at
  `outputs/mnist_jointae32_interactive_v1/` with `TRAINING_COMPLETE`, epoch
  history, manifest, selected checkpoint, test features, and split indices.
- All artifact hashes match the manifest. The checkpoint selected epoch 10
  after ten epochs. Validation accuracy is 0.9930 and official-test accuracy
  is 0.9935.
- The exact 55,000 training and 5,000 validation indices are unique and
  disjoint. The 10,000 official-test indices cover the full test split.
- The local raw MNIST train/test image and label hashes match the training
  manifest. Saved test features are finite with shape `[10000,128]`.
  Recomputed mean and covariance match the saved float64 statistics exactly.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  validation and packaging work.


### 2026-10-02 13:55 CEST - Recompute fixed-weight MNIST metrics on CPU

- Loaded the selected epoch-10 checkpoint and the exact saved split indices.
  Recomputed all 55,000 training, 5,000 validation, and 10,000 official-test
  examples with fixed weights using the standalone trainer implementation.
- Recomputed accuracy is 0.998818 on training, 0.9930 on validation, and 0.9935
  on test. Validation and test accuracies exactly match the saved checkpoint.
- Recomputed validation loss is 0.008962179 and test loss is 0.008644693. Their
  differences from the GPU-saved values are below `2e-7`.
- Recomputed all 10,000 raw test features on CPU. Shape is `[10000,128]`, every
  value is finite, labels match exactly, and the maximum CPU-versus-GPU feature
  difference is `1.0009e-3`. The maximum logit difference is `3.6440e-3`.
- This was a read-only local CPU validation. No scheduler job was submitted,
  resumed, cancelled, or modified.
- The packaged public loader also passed frozen-weight controls. Official-test
  half-versus-half FD is 6.8939 and a seed-2003 same-distribution bootstrap FD
  is 0.0657. On 512 fixed test images, FD rose monotonically for Gaussian noise
  (`0.157, 0.741, 4.244` at sigma `0.05, 0.1, 0.2`), translation (`10.722,
  36.918, 90.782` at `1,2,4` pixels), and rotation (`1.043, 3.823, 13.699` at
  `5,10,20` degrees).
- Progressive class support also raised FD monotonically: retaining 8, 5, 2,
  and 1 digit classes gave `9.784, 32.600, 93.904, 134.747`, respectively.


### 2026-10-02 21:29 CEST - Production status and DiT recovery audit

- Across 234 production trajectories, 198 are trained at one million, 18
  CelebA64 RGB trajectories are running, and 18 seed-1 Small-DiT trajectories
  are cleanly interrupted and ready for exact resume.
- All 18 seed-1 DiT tasks in array `17683429` reached the intended 24-hour
  SIGUSR1 boundary between 12:30 and 12:38 CEST. Slurm records exit `75:0` for
  every task; no task stopped early.
- All 18 DiT run records have `state=interrupted`, `error=null`, the expected
  `received SIGUSR1` reason, inactive cleanly closed W&B sessions, and finite
  latest losses. Exact recovery steps span 552,194-556,621.
- Recomputed SHA-256 for each newest DiT recovery payload. All 18 match their
  recovery indexes, so all 18 are safe to resume to the unchanged one-million
  target. No DiT continuation job is currently submitted.
- CelebA64 RGB stage-5 array `17654918` remains 18/18 running under project ID
  `berzelius-2026-269`. All recovery indexes and histories have reached 925,000
  with finite losses. Final verifier `17654919` waits for stage 5 to finish.
- The complete current DiT and RGB log scan found no traceback, OOM,
  disk/quota, non-finite, assertion, runtime, cancellation, or kill signature.
- No scheduler job was submitted, resumed, cancelled, or modified during this
  check.


### 2026-10-02 21:37 CEST - Resume all seed-1 Small-DiT trajectories

- Reused the frozen batch
  `outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178`.
  All 18 selected rows had the expected SIGUSR1 stop, `error=null`, finite
  losses, and newest recovery payloads whose SHA-256 matched their indexes.
- The write-free preview selected exactly tasks 0-17, the unchanged one-million
  target, one thin A100 per task for 24 hours, a five-minute recovery signal,
  no automatic requeue, and project ID `berzelius-2026-269`.
- Submitted exact-resume array `17691417`. Slurm accepted all 18 tasks under
  the new project ID; they are waiting for the scheduler to assign GPUs.
- No RGB, MNIST, seed-0 DiT, U-Net, or weighting job was changed or duplicated.


### 2026-11-19 - Rebuttal access check

- Attempted a read-only SSH check of the checkout and job queue.
- Berzelius rejected authentication with publickey,keyboard-interactive.
  No remote command ran. Current job status is unknown.
- No scheduler job was submitted, resumed, cancelled, or changed.
- The researcher must establish SSH login before cluster preparation can
  continue. Local preservation and GitHub work can proceed independently.
