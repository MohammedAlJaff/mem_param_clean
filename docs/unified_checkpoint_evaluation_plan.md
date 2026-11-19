# Unified checkpoint evaluation plan

## Goal

For each saved checkpoint, generate one deterministic bank of 10,000 images.
Compute all requested metrics from that bank. Save enough intermediate results
to add or change one metric without repeating unrelated work.

This plan covers MNIST and natural images. The current implementation already
contains the metric mathematics and restartable per-checkpoint artifacts. The
remaining work is the production-scale orchestration and its frozen reference
banks.

## Work unit

| Level | Unit | Reason |
| --- | --- | --- |
| Scheduler task | One run and one evaluation stage | One GPU stays useful across several checkpoints. |
| Saved progress | One complete checkpoint | A restart never repeats completed checkpoints. |
| Parallel work | Many runs | Runs do not share model state. |
| Metric restart | One checkpoint and feature space | A changed encoder does not invalidate other spaces. |

One task loads a run once. It then evaluates as many whole checkpoints as the
wall time permits. It writes a completion marker after each checkpoint. A later
task resumes at the first incomplete checkpoint.

Do not submit one task per checkpoint by default. That design repeats setup,
increases queue overhead, and creates too many short jobs. Use a checkpoint task
only for a focused diagnosis or timing test.

## Data flow

```text
saved run + frozen noise + frozen held-out bank
                    |
                    v
          10,000 generated images
                    |
          +---------+----------+
          |                    |
          v                    v
   quality features      memorization features
          |                    |
          v                    v
  Fréchet statistics     top-two train neighbors
          |                    |
          +---------+----------+
                    |
                    v
        scalar tables + compact evidence
```

Generate each image once per checkpoint. All metrics for that checkpoint use
the same generated bank.

## Frozen inputs

Each evaluation records and hashes these inputs:

- `config.json`, `run.json`, and the selected checkpoint;
- the exact saved training tensor and source identifiers;
- a deterministic 10,000-row noise bank;
- one fixed 10,000-image held-out bank for the dataset and preprocessing;
- every encoder weight file and preprocessing rule;
- the sampler settings and software environment.

CelebA32 evaluation uses one separately frozen set of 10,000 official test
images. The seed-1 builder maps saved split-local train indices through the
official train catalog. It rejects training JPEG overlap, internal JPEG
duplicates, training tensor overlap, and internal transformed-tensor
duplicates. Its tensor, filenames, test indices, identity labels, source-run
manifest, preprocessing, partition hash, and integrity hashes are saved as
`outputs/reference_banks/celeba_test_seed1_clean_n10000.{pt,json}`.
Post-hoc evaluation reads this bank. It never rebuilds or reselects it.

Other datasets and resolutions still need one equivalent frozen bank before
paper-scale evaluation.

## Metrics

### Natural images

| Question | Metric | Representation |
| --- | --- | --- |
| Standard image quality | Inception FID | Inception-v3 pool features |
| Semantic image quality | DINOv2 Fréchet distance | DINOv2 class-token features |
| Exact visual memorization | `f_mem` | Raw pixels |
| Copy-like semantic memorization | `f_mem` | SSCD features |
| Perceptual memorization | `f_mem` | LPIPS features |
| Semantic memorization | `f_mem` | DINOv2 features |

Call the second quality value DINOv2 Fréchet distance or FD-DINOv2. Do not call
it FID because it does not use Inception features.

### MNIST

| Question | Metric | Representation |
| --- | --- | --- |
| Global quality | JointAE Fréchet distance | Raw JointAE-32 features |
| Class support | Class-aware JointAE FD | Per-class JointAE-32 features |
| Exact memorization | `f_mem` | Raw pixels |
| Learned-feature memorization | `f_mem` | Standardized JointAE-32 features |

### Diagnostics

Every checkpoint also records finite fraction, raw minimum, raw maximum, raw
root-mean-square value, and the fraction outside `[-1,1]`. These checks use the
unclamped generated output.

## Memorization evidence

For every generated image and feature space, retain:

- `d1`: distance to the nearest training image;
- `d2`: distance to the second-nearest training image;
- `r = d1 / d2`;
- `nn_idx`: nearest training-image index;
- `second_nn_idx`: second-nearest training-image index.

Do not save the full generated-by-training distance matrix. It is large and can
be recomputed in chunks from cached features. The top-two evidence is enough for
threshold sweeps, nearest-neighbor examples, and `f_mem`.

## Threshold calibration

Use one fixed held-out bank to determine what neighbor ratios occur without
training-set membership.

| Held-out rows | Use |
| --- | --- |
| First 1,024 | Calibrate one `rho` per feature space and alpha. |
| Remaining 8,976 | Validate the resulting false-positive rates. |

Freeze alpha values `{0.005, 0.01, 0.02, 0.05}`. Use alpha `0.01` as the
headline result. Report the complete curve. For raw-pixel continuity with older
work, also report fixed `rho` values `{0.3, 0.5, 0.7}`.

Never reuse a raw-pixel threshold in SSCD, LPIPS, DINOv2, or JointAE space.

## Held-out denoising loss

Evaluate loss on the fixed held-out bank with two views:

- seeded random training times for the primary held-out loss;
- fixed times `t={0.5,0.75}` for interpretable cross-run checks.

Record the noise seed and batch size. Loss evaluation is separate from image
sampling and feature metrics.

## Saved artifacts

Retain permanently:

- request, manifest, hashes, and completion markers;
- scalar metrics and combined checkpoint tables;
- top-two neighbor distances, ratios, and identities;
- reference and training features shared across checkpoints;
- small preview grids and selected nearest-neighbor examples.

Treat these as reusable but evictable:

- the 10,000 generated images for each checkpoint;
- generated feature matrices for each encoder.

Keep generated banks for the timing gate, final checkpoint, figure checkpoints,
and any failed or disputed result. Other banks can be removed only after their
features, neighbor evidence, statistics, hashes, and completion markers pass
validation.

## Figure-bundle contract

Freeze figure inputs before a paper-scale evaluation starts. Each completed
checkpoint must retain scalar metrics, full top-two neighbor arrays, source
hashes, checkpoint hashes, generated tensor hashes, and run identifiers. Keep
the generated bank for the final checkpoint, each selected figure checkpoint,
and each failed or disputed result. Do not create transfer archives in a GPU
task.

After all evaluations finish, run
`python -m mem_param_src.runners.figure_bundle` once on a CPU. Give it one
`--evaluation-root` for each completed checkpoint. It combines all inputs into
CSV, JSON, NPZ, and PNG files and creates one `figure_bundle.tar.gz`-style
archive. It verifies test-reference use, tensor and file hashes, neighbor-array
lengths, run-step uniqueness, and table row counts. It refuses to replace an
existing bundle.

The qualitative rule is part of schema version 1. In SSCD space at alpha
`0.01`, select 16 smallest-r examples, 8 unused boundary-nearest examples, and
8 deterministic unused examples above the boundary. Save each generated image
with its nearest and second-nearest training image as exact float arrays. The
PNG is a preview, not the numeric source.

The final archive must support every planned figure without model loading,
sampling, encoding, nearest-neighbor computation, or FID computation.

## Restart rules

| Change | Repeat |
| --- | --- |
| Plot style or reported alpha | Summary only |
| `rho` threshold in an existing space | Threshold summary only |
| New memorization encoder | That feature and neighbor branch |
| New quality encoder | That feature and Fréchet branch |
| New sample count, noise, sampler, or checkpoint | Sampling and all downstream work |
| Corrupt generated bank | That checkpoint and all downstream work |

Every cache key includes the source tensor hash, checkpoint hash, sampler,
noise, encoder hash, preprocessing, and sample count.

## Production sequence

1. Freeze the shared 10,000-image held-out bank.
2. Run a three-checkpoint gate on CelebA32 N=2048, seed 0: x/x, eps/eps, v/v.
3. Generate 10,000 images once for each gate checkpoint.
4. Compute every planned quality and memorization space.
5. Review runtime, peak memory, storage, ranges, and neighbor examples.
6. Freeze batch sizes, wall time, retention, and array concurrency.
7. Evaluate complete runs. Process runs in parallel and checkpoints serially.

For arrays above 80 tasks, select and record an explicit concurrency throttle
before submission.

## Sampling timing gate

The first timing gate measures sampling only. It does not compute metrics or
save the 10,000-image bank.

The default run is CelebA32 N=2048, seed 0, x/x at step 1,000,000. It uses the
saved 50-step full-Heun sampler and one thin A100-40GB GPU. The run has 72 saved
checkpoints.

The benchmark reports:

- measured batch throughput;
- measured or projected time for 10,000 samples at one checkpoint;
- projected sampling hours for all 72 checkpoints;
- separate setup, checkpoint-load, warm-up, and peak-memory values;
- raw-range and finite-value diagnostics.

The all-checkpoint number is a sampling-only projection. It excludes metric
encoders, nearest-neighbor searches, artifact writes, and scheduler wait time.

Use `notebooks/sanity/checkpoint_sampling_runtime.ipynb` for interactive changes.
Use `jobs/berzelius/checkpoint_sampling_runtime/submit.sh` for the frozen 30-minute
Berzelius measurement.

## Full interactive metric gate

`mem_param_src/runners/checkpoint_metrics_benchmark.py` evaluates one
CelebA32 checkpoint. It can generate a complete 10,000-image bank or accept a
verified `--generated-bank`. It computes raw-pixel `f_mem`, SSCD `f_mem`,
and Inception FID-10k against the clean test bank described above. Reference
contamination fails before sampling or metric computation.

Run it inside an existing GPU allocation:

```bash
bash jobs/berzelius/checkpoint_sampling_runtime/run_interactive_metrics.sh
```

The default wrapper reuses
`outputs/interactive_checkpoint_metrics/20261003T000727Z/generated_samples.pt`
and writes clean-test results to a new output directory. The evaluator saves
both encoder feature matrices, float64 FID statistics, and top-two neighbor
evidence. It prints progress and separate sampling, shared-reference, and
checkpoint-metric times. It also projects the saved sampling cost across all
72 checkpoints.

### Measured CelebA32 gate

The completed clean-test gate is
`outputs/interactive_checkpoint_metrics/20261003T084819Z/`. It evaluates the
N=2048, seed-0, x/x U-Net-S checkpoint at step 1,000,000. It reuses the exact
10,000 generated images from the preceding sampling measurement and compares
them with the frozen, contamination-checked official CelebA test bank.

| Metric | Result |
| --- | ---: |
| Inception FID-10k | 34.8159 |
| Raw-pixel f_mem, fixed rho=0.3 | 0.5197 |
| Raw-pixel f_mem, held-out alpha=0.01 | 0.9450 |
| SSCD f_mem, held-out alpha=0.01 | 0.6235 |

Fresh sampling took 154.401 seconds on one A100-80GB. The clean-test metric
rerun took 31.992 seconds because it reused that generated bank. Combining the
measured sampling kernel with the metric stages projects 173.191 seconds per
checkpoint and 3.469 GPU-hours for all 72 checkpoints. This is one run and one
checkpoint. It qualifies the workflow and budget; it does not establish trends
across parameterizations.

The strict CPU export of the replay gate is
`outputs/figure_bundles/celeba32_checkpoint_gate_test_v1.tar.gz`. Its manifest
marks it ready for publication. It contains 12 scalar rows, 8 calibration
rows, 306 distribution rows, four complete neighbor-array files, and 32 exact
qualitative example triplets. All 14 bundle files passed their SHA-256 checks.

## Historical full-evaluation runtime evidence

The old repository contains complete runtime evidence for two main CelebA32
full-trajectory evaluations and one 11-run follow-up array from May 2026. Each
scheduler task evaluated one run at 72 checkpoints. Each checkpoint generated 10,000 images with 50-step
Heun sampling and used 10,000 held-out images for FID. Each task used one A100
GPU and eight CPU cores.

The old combined CSV files record separate timers for sampling, f_mem, FID,
held-out loss, and the complete checkpoint. Slurm accounting still retains the
matching task wall times.

| Old evaluation | Tasks | Median checkpoint | Median task | Task range | GPU-hours | Array span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original sweep, mixed A100 | 84 | 78.9 s | 01:34:58 | 01:31:59-01:44:23 | 135.1 | 35:23:20 |
| Capacity sweep, mixed A100 | 97 | 143.7 s | 02:53:37 | 01:32:43-03:14:15 | 243.1 | 03:14:15 |
| Capacity delta, mixed A100 | 11 | not combined | 03:09:48 | 02:54:59-03:13:26 | 34.4 | 03:13:26 |

Slurm assigned both available A100 classes. The original suite used 35 A100-40GB
and 49 A100-80GB tasks. The capacity suite used 67 A100-40GB and 30 A100-80GB
tasks. The delta suite used nine A100-40GB and two A100-80GB tasks.

The original array used a four-task concurrency limit. Its 35-hour array span
does not mean that one run took 35 hours. The capacity arrays ran without that
limit, so their array spans approximately equal their slowest task.

The per-checkpoint timer distributions contain 6,048 original-sweep records
and 6,984 capacity-sweep records:

| Timed section | Original median | Capacity median |
| --- | ---: | ---: |
| Generate 10,000 images | 65.224 s | 129.319 s |
| Compute f_mem | 0.221 s | 0.220 s |
| Compute FID | 9.438 s | 9.714 s |
| Compute held-out losses | 2.038 s | 3.943 s |
| Complete checkpoint | 78.897 s | 143.707 s |

Sampling dominated the old checkpoint cost. Model capacity explains most of
the difference inside the capacity suite. The median sums of all 72 checkpoint
timers were 01:39:26 for `base32_m124`, 02:58:41 for `base64_m124`, and
03:08:46 for `base64_m1244`.

The timer sums closely match Slurm wall time. The original median differs by
about 31 seconds. The capacity median differs by about 72 seconds. This small
gap covers imports, model loading, and task cleanup.

The historical measurements support one run per staged scheduler task. They
do not establish the final budget for the new protocol. The old evaluator did
not compute SSCD, LPIPS, DINOv2, or JointAE feature branches, and it did not
retain the new restartable feature artifacts.

Historical evidence paths in the sibling old repository:

- `outputs/evals/mem_boundary_20260427_all_done_quality_20260501/`
  `all_checkpoint_metrics.csv`;
- `outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_20260503/`
  `all_checkpoint_metrics.csv`;
- `outputs/evals/mem_boundary_capacity_20260427_full_trajectory_quality_delta_20260503/`
  for the 11-run follow-up manifest;
- `jobs/evals/logs/` for the matching Slurm output and error files;
- Slurm jobs `16473994`, `16479328`, and `16484437`.

Use Slurm accounting to recover task wall time:

```bash
sacct -X -j JOB_ID \
  --format=JobID,State,Elapsed,Start,End,AllocTRES
```

Use the CSV timing columns for compute-stage attribution:
`sample_runtime_sec`, `f_mem_runtime_sec`, `fid_runtime_sec`,
`test_loss_runtime_sec`, and `checkpoint_runtime_sec`.

## Current implementation boundary

| Component | State |
| --- | --- |
| Dataset-aware metric mathematics | Implemented and tested |
| Restartable per-checkpoint metric artifacts | Implemented and tested |
| Minimal MNIST and CelebA metric notebooks | Implemented |
| Sampling-only timing library and notebook | Implemented |
| Short frozen Berzelius timing job | Implemented |
| Shared fixed 10,000-image reference banks | CelebA32 implemented; other datasets pending |
| Full one-checkpoint CelebA32 metric gate | Implemented and measured on 10,000 generated/test images |
| Portable multi-root figure exporter | Implemented and tested on the clean gate |
| Run-level staged production scheduler | Not implemented |
| Full natural-image metric matrix at 10,000 samples | Not launched |

Do not interpret the sampling timing job as validation of the complete metric
pipeline. Add metric stages one at a time after this timing result is reviewed.
