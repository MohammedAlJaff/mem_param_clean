# EPS/EPS 64x64 N=512 capacity study

This study holds the native EPS/EPS objective, N=512, seed 0, optimizer, and
effective batch 256 fixed. It crosses four image formats with three UNet
capacities and complete attention on/off controls. The result is 24 runs:

| Factor | Values |
| --- | --- |
| Data | CelebA gray, CelebA RGB, MNIST gray, Flowers RGB |
| Resolution and N | 64x64, N=512 |
| Capacity | small, deep, wide-deep |
| Attention | off everywhere, or middle plus 16/8-pixel attention |
| GPU class | 16 small/deep tasks on thin; 8 wide-deep tasks on fat |

The config targets one million optimizer updates, but every launcher is capped
at the 400k review gate. No launcher in this directory continues past 400k.
The late 500k-1M checkpoints remain in the config only for a separately
authorized continuation.

## Validation sequence

No script submits work when this directory is created. First validate the
matrix locally:

```bash
PYTHONPATH=. conda run -n memcompgen python \
  jobs/berzelius/eps_eps_64_n512/validate_suite.py \
  jobs/berzelius/eps_eps_64_n512 \
  configs/runners/eps_eps_64_n512_400k.py
```

Create one immutable batch snapshot:

```bash
bash jobs/berzelius/eps_eps_64_n512/prepare.sh
```

Copy the printed absolute directory into `BATCH_DIR`. Every later stage uses
the frozen config, matrix, source tree, and validation scripts in that folder.
Workers change into the batch directory and assert that `mem_param_src` came
from `BATCH_DIR/source`, never from the live checkout.

Submit the capacity probes:

```bash
bash jobs/berzelius/eps_eps_64_n512/submit_probes.sh --dry-run BATCH_DIR
bash jobs/berzelius/eps_eps_64_n512/submit_probes.sh BATCH_DIR
```

The four small/deep profiles run on full A100-SXM4-40GB GPUs. The two
wide-deep profiles run on full A100-SXM4-80GB GPUs. Explicit GRES requests
exclude the 10GB MIG slices exposed by fat nodes. Each profile measures
physical batches 512, 256, 128, and 64 in fresh CUDA processes on real
CelebA64 RGB data. Batch 512 is diagnostic only. Selection chooses 256, then
128, then 64, subject to finite loss and peak-reserved limits of 32 GiB on
thin or 64 GiB on fat. Smaller physical batches accumulate to effective batch
256. The finalizer writes:

- `validation/probes/results/*.json`
- `validation/selected_batches.tsv`
- `validation/probe_summary.json`
- `PROBE_COMPLETE`

After the probe finalizer succeeds, run six real online-W&B integration tests
and the widest attention exact-resume comparison:

```bash
bash jobs/berzelius/eps_eps_64_n512/submit_smokes.sh --dry-run BATCH_DIR
bash jobs/berzelius/eps_eps_64_n512/submit_smokes.sh BATCH_DIR
```

Each profile smoke trains CelebA64 RGB for 250 optimizer updates using its
selected batch. It must finish W&B and save all four sampler diagnostics and
raw tensors. Finite profiles also require grids and projected-pixel f_mem.
The wide-deep
attention-on check compares an uninterrupted run with a process-boundary
resume. It requires exact model, Adam, RNG, shuffled-batch, history, and
monitoring artifacts. It also requires the resumed process to reuse the same
online W&B run ID.

The smoke finalizer combines measured optimizer speed with complete monitor
walltime. It includes all 14 monitor points through 400k and adds 30 percent.
It writes `validation/runtime_plan.json` and `validation/launch_mode.txt`.
An estimate above 40 hours selects two exact-resume segments only when each
segment remains below 40 hours after its actual monitor count and margin. An
unsafe segment blocks launch for manual redesign. A shorter total estimate
selects one direct allocation.

## Main launch and recovery

The main submitter refuses to run without `SMOKE_COMPLETE`:

```bash
bash jobs/berzelius/eps_eps_64_n512/submit.sh --dry-run BATCH_DIR
bash jobs/berzelius/eps_eps_64_n512/submit.sh BATCH_DIR
```

Both arrays are unthrottled. Every task requests a full named A100, 48 hours,
no automatic requeue, and USR1 five minutes before timeout. Direct mode runs
to 400k. Split mode submits 0-200k followed by 200k-400k arrays with `aftercorr`
within the identical thin and fat task ranges. Accepted Slurm IDs are written
immediately to `main/submission.tsv`. If submission becomes partial, the
submitter refuses a blind retry.

A final `afterok` job checks all 24 rows. It requires state `paused` at 400k,
all 14 checkpoints through 400k, the newest recovery state, successful W&B
completion, and diagnostics plus raw samples for all four sampling profiles at
all 14 steps. Direct-safe must also have every grid and f_mem record. A finite
alternate must have both; a nonfinite alternate remains a valid recorded
diagnostic. Only that verifier writes `GATE_400K_COMPLETE`.

Recovery states rotate every 5,000 updates and retain the model, Adam, all
RNGs, shuffled-batch cursor, history, elapsed time, and W&B identity. Inspect
Slurm first, then manually recover failed tasks only within the 400k gate:

```bash
bash jobs/berzelius/eps_eps_64_n512/resume.sh --dry-run \
  BATCH_DIR thin 3,7 400000
bash jobs/berzelius/eps_eps_64_n512/resume.sh \
  BATCH_DIR thin 3,7 400000
```

Thin IDs must stay in 0-15. Fat IDs must stay in 16-23. The run lock rejects a
concurrent resume, and the resume path uses the saved config and W&B run.
Each accepted recovery job is appended immediately to
`main/recovery_jobs.tsv`.

The original dependent gate cannot run when an original array task fails, even
if a later recovery succeeds. After every repaired task is complete, submit an
independent replacement gate. Depend only on the recovery jobs, not on the
stale original verifier:

```bash
bash jobs/berzelius/eps_eps_64_n512/submit_gate.sh --dry-run \
  BATCH_DIR afterok:RECOVERY_JOB_ID
bash jobs/berzelius/eps_eps_64_n512/submit_gate.sh \
  BATCH_DIR afterok:RECOVERY_JOB_ID
```

Omit the dependency only after manually confirming that all 24 tasks are
inactive. Accepted replacement gate IDs append to
`main/replacement_gate_jobs.tsv`.

## Monitoring contract

At steps 0, 250, 500, 1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 200k, 300k,
390k, and 400k, one fixed 128-noise bank is replayed with full Heun-50 over
`[0.001, 0.999]`, `x_t_max`, and no final clamp:

1. `direct_safe`: direct epsilon conversion, denominator floor 0.05
2. `via_x_safe`: legacy via-x conversion, denominator floor 0.05
3. `direct_exact`: direct epsilon conversion, literal division
4. `via_x_exact`: legacy via-x conversion, literal division

Artifacts are namespaced under `monitoring/<artifact>/<profile>/`. W&B logs
raw range, standard deviation, nonfinite counts, out-of-range fraction, the
64-sample comparison grid, and the 128-sample projected-pixel f_mem result.
The direct-safe profile is mandatory. During full training, a nonfinite
alternate profile is saved as a failed diagnostic without terminating a
healthy direct-safe trajectory.

At 20 occupied hours per run, the 16 thin runs use 320 hours. The eight fat
runs use 160 occupied hours and are billed at twice the thin rate. The total is
640 thin-equivalent GPU-hours, excluding probes and smokes. This stage does not
run FID, learned-feature evaluation, extra seeds, or training past 400k.
