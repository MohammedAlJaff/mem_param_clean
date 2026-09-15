# mem_param_clean

STANDING COLLABORATION RULE
Use plain English, short sentences, active verbs, precise terms, and a natural
voice. Help the researcher understand and own the work. For substantial changes,
show the relevant files, how the parts connect, and a small usage example with
expected outputs. Use real code or clearly labelled pseudocode. State what was
tested. Link to the implementation, settings, and results. Keep reusable examples
in the relevant docs and the layout in docs/repo_map.md. Involve the researcher
in meaningful choices without routine approval requests. Apply this rule to
future notes and documentation. Full guidance: root AGENTS.md, Collaboration
and Code Understanding.

Clean working repository for the target-parameterization memorization study.

Research question:

```text
Under fixed architecture, data, optimizer, sampler, and evaluation protocol,
how do target parameterizations (`x`, `eps`, `v`) change memorization dynamics?
```

Main claim:

```text
Target parameterization changes the route from generation to memorization.
```

This repository is built one file class at a time from the old `mem_par`
checkout. The old checkout stays intact. Generated outputs are not part of the
live source tree unless a small result file is selected for a paper figure.

See:

- `docs/objective.md`
- `docs/config_reference.md`
- `docs/repo_map.md`
- `docs/preservation.md`
- `TODO.txt`
- [Critical eps/eps audit and Berzelius handoff](CRITICAL_EPS_EPS_SAMPLING_AUDIT_2026-09-09.txt)
- `PAPER_NOTES.txt` — draft paper wording, interpretation, and research ideas.

## Environment

Create the complete local research environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mem_param_clean
```

The environment installs this repository in editable mode with the image,
evaluation, and development profiles. The dependency groups in
`pyproject.toml` are authoritative:

- the base installation provides the current PyTorch package and configuration;
- `image` provides Torchvision and Pillow for image datasets;
- `evaluation` provides torch-fidelity for Inception features and FID;
- `dev` provides tests, linting, plotting, and notebook execution.

For a core-only installation:

```bash
python -m pip install -e .
```

PyTorch and Torchvision are declared without machine-specific CPU, CUDA, or MPS
build pins. On managed GPU systems, install the site-supported compatible pair
before installing this package.

## Training and evaluation commands

Run one short synthetic experiment and evaluate it after training:

```bash
mem-param-synthetic-train \
  --config configs/runners/gmm_smoke.py \
  --run-dir outputs/gmm_smoke \
  --evaluate-after-training
```

Run evaluation later from the saved artifacts:

```bash
mem-param-synthetic-eval --run-dir outputs/gmm_smoke
```

Image runs use `mem-param-image-train` and `mem-param-image-eval`. Evaluation
requires the saved run directory. An optional evaluation config and `--set`
flags can change only sampling and evaluation settings.

Each run saves `data_bundle.pt` and `data_manifest.json`. The tensor bundle
contains the exact training and reference values. The manifest records their
source pools, stable sample identifiers, source indices, tensor hashes, and
selection seed. Unused source examples are not copied into the run.

## Berzelius: prepared 2,000-step epsilon comparison jobs

The [smoke-job plan](jobs/berzelius/eps_recovery_smoke/README.md) prepares six
CelebA32 comparisons with W&B image histories and saved raw samples. It uses
historical training settings and varies conversion, final Euler, clamp, LR,
and a native velocity reference. Two thousand steps test early behavior;
they do not establish recovered face convergence. No jobs were submitted.

```bash
bash jobs/berzelius/eps_recovery_smoke/submit.sh --dry-run
bash jobs/berzelius/eps_recovery_smoke/submit.sh
```

The first command prints only. The second submits all six one-GPU tasks
concurrently. Use `submit.sh 0` for just the historical sampling anchor.
The job README explains W&B, data paths, saved outputs, and raw-range inspection.

The [one-million-step recovery workflow](jobs/berzelius/eps_recovery_1m/README.md)
contains the eight completed seed-0 training configurations and their three-point
evaluation. FID uses 10,000 generated images against one frozen 10,000-image
CelebA validation reference. `RUN_JOURNAL.md` records job IDs, status checks,
and the corrected eps, v/v, and x/x results. Generated artifacts remain outside
Git.

## CelebA32 eps/eps controls

From the repository root, activate the environment and run:

```bash
conda activate memcompgen
python -m mem_param_src.runners.image_train \
  --config configs/runners/celeba32_eps_eps_monitor.py \
  --run-dir outputs/session_runs/celeba32_eps_eps_n32_exact_01 \
  --device mps --steps 1000 --checkpoint-every 250 \
  --sampler heun --sampling-steps 50 --no-sampling-safe-div
```

This uses 32 grayscale faces at 32x32, the historical 140-pixel aligned-face
crop, a separate validation reference set, and direct unit-MSE eps/eps.
The current UNet has widths 64/128/192 with middle attention. This is a diagnostic
configuration; it is not an exact reconstruction of the old CelebA study.
Local grids, loss/f_mem curves, and all 500 raw generated samples are enabled.
W&B is off. The comparison grid shows 32 training and 64 generated images.

Change these flags when needed:

```text
--sampling-safe-div --sampling-safe-div-denominator-floor 0.05
--no-sampling-safe-div
--sampling-steps 200
--sampler euler
--heun-final-euler
--no-heun-final-euler
--set sampling.eps_to_v=direct
--set sampling.eps_to_v=via_x
--sampling-start-time 0.01 --sampling-end-time 0.99
--sampling-output x_1
--steps 5000 --lr 0.0001 --batch-size 32 --seed 7
```

The safe-division flags affect sampling only. Changing the denominator value
alone does not enable clamping. A floor of 0.01 matches the smallest ideal
coefficient on [0.01,0.99]; 0.05 engages the clamp over part of that interval.
Keep exact and clamped results distinct. Use a new run directory for each
training launch. For RGB faces, add `--set data.image_channels=3`.

To isolate a sampling change after training, replay the same saved checkpoints
and noise with a new evaluation name:

```bash
python -m mem_param_src.runners.image_eval \
  --run-dir outputs/session_runs/celeba32_eps_eps_n32_exact_01 \
  --eval-name clamp005_heun50 --device mps \
  --set sampling.use_safe_div=true \
  --set sampling.safe_div_denominator_floor=0.05 \
  --set sampling.n_steps=50
```

Replay reads saved data and checkpoints. Full generated batches go to
`evaluations/clamp005_heun50/generated_samples/`; full-pixel evaluation records
go to `checkpoint_metrics.json` in that evaluation directory. This evaluator
does not create the training-monitor PNG panels or projected f_mem curve.
The eps/eps discrepancy remains open; low training loss alone is not success.

### Direct epsilon conversion and final Euler

New configurations use `sampling.eps_to_v="direct"` and
`sampling.heun_final_euler=True`. Epsilon sampling computes
`(z_t - eps_hat) / t`, with the existing optional denominator floor.
Training losses keep their current behavior. Heun uses its usual two-stage
step until the final interval, where it takes only the Euler step.

Older saved runs retain `via_x` and full Heun during standalone evaluation.
To test the new conversion on an older saved run, add these overrides to a
named evaluation command:

```text
--set sampling.eps_to_v=direct
--set sampling.heun_final_euler=false
```

Then repeat with `sampling.heun_final_euler=true` and a different evaluation
name to isolate the final-step change. Keep the checkpoint, noise, floor,
step count, and output choice fixed. To reproduce the previous conversion,
use `sampling.eps_to_v=via_x`. These choices appear in `settings.json`.
With exact division, direct and historical conversions agree mathematically.
A floor of 0.05 on [.01,.99] exposes their different behavior near the start.

Example Python call using an already loaded model and saved noise:

```python
samples = model.sample(
    z_init, n_steps=50, method="heun", start_time=0.01, end_time=0.99,
    use_safe_div=True, safe_div_denominator_floor=0.05,
    eps_to_v="direct", heun_final_euler=True, output="x_t_max",
)
```

The result has the same shape as `z_init`. This API was tested with analytical
CPU models; trained-checkpoint generation remains to be compared.
`x_t_max` returns the integrated state. `x_1` still makes an additional model
call at the final time, even when the last solver interval uses Euler.

## Online monitoring demonstration

For the local MNIST16 v/v run with 32 training images and 1,000 updates, use only
the image training runner:

```bash
conda activate memcompgen
python -m mem_param_src.runners.image_train \
  --config configs/runners/mnist_vv_monitor.py \
  --run-dir outputs/session_runs/mnist16_vv_n32_1k_20260908_02 \
  --device mps
```

Images are resized to 16 x 16. The config generates 500 fixed-noise samples at
step zero and every 250 updates.
Each grid shows 32 fixed training images above 64 generated images, eight per
row. Find the grids in the run's `monitoring/comparison_grids/` and the live
loss/f_mem figures in `monitoring/plots/`. W&B is off for this local run.
The f_mem curve uses projected pixels at rho=0.3 against all 32 training images.
Interpret it alongside sample quality. No separate evaluation command is needed.
The terminal briefly announces data loading, model preparation, training,
checkpoint generation, metric computation, and completion. It prints loss every
25 updates and projected f_mem, mean_r, rho, and sample counts at checkpoints.
One saved-folder line replaces repeated figure paths after each checkpoint.
The 32x32 run was stopped at the user's request. The first MNIST16 run completed;
use a new output folder for each launch.

For another image config, enable these features with:

```bash
--generation-monitor --training-grid-samples 32 \
--f-mem-monitor --f-mem-samples 500
```

Change `generation_monitor.n_samples` to choose how many generated images the
grid displays. `--training-grid-samples 0` retains generated-only grids.

Use `--save-generated-samples` to keep the entire generated batch at every
monitoring checkpoint. This is enabled for future runs using the MNIST16 config.
The first completed run predates this option. The files are
`monitoring/generated_samples/step_<step>.pt`, with raw CPU tensors of shape
`[500, 1, 16, 16]` for this run. The first 64 samples form the generated grid.
Use `--no-save-generated-samples` to keep only grids and metric records.
If post-training evaluation is enabled, the same option saves every evaluated
batch under `evaluations/<name>/generated_samples/`.

After a run with tensor storage enabled, read a saved batch directly:

```python
from pathlib import Path

import torch

run_dir = Path("outputs/session_runs/mnist16_vv_n32_1k_20260908_02")
samples = torch.load(run_dir / "monitoring/generated_samples/step_000250.pt",
                     map_location="cpu", weights_only=True)
# samples.shape is [500, 1, 16, 16] for the MNIST16 config.
```

To redraw the curves from saved records without loading a model:

```python
from mem_param_src.runners.image_plotting import save_monitor_curves

paths = save_monitor_curves("outputs/session_runs/mnist16_vv_n32_1k_20260908_01")
# paths maps "loss" and "f_mem" to PNG files once both records exist.
```

The local results index is `outputs/session_runs/INDEX.md`. It links this run
and the earlier MNIST runs without moving their saved artifacts.

The runner script is `mem_param_src/runners/image_train.py`. The single-run
settings are in `configs/runners/mnist_wandb_demo.py`.

Run from the repository root after confirming the W&B destination:

```bash
WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true \
conda run --no-capture-output -n memcompgen \
  python -m mem_param_src.runners.image_train \
  --config configs/runners/mnist_wandb_demo.py \
  --run-dir outputs/mnist_wandb_eps_eps_large_10k_01 \
  --device mps --evaluate-after-training
```

The current config selects eight grayscale MNIST images at 32 x 32, eps/eps,
exact division, and 10,000 updates. The UNet uses 64 base channels, multipliers
(1, 2, 3), and two residual blocks per encoder level: 9,393,985 parameters. Optional
encoder attention is disabled; middle-block attention remains active.

W&B logs loss every 25 updates and 16-image previews every 250 updates. Each
preview uses 50 Heun steps and the same saved noise. Use a new run directory.
The larger model completed 10,000 updates and checkpoint evaluation on MPS:
[W&B run](https://wandb.ai/maj90/mnist-unet/runs/en9xtd1x). Final loss was
0.009429, but most final samples remained noisy or saturated. Final pixel
f_mem was zero at rho=0.3; poor generation prevents a generalization claim.
This run did not enable the later training-time f_mem option. See TODO.txt
for the next-session starting point and attention comparison (T017).

Earlier completed runs used the smaller UNet and 1,000 updates:
- v/v: https://wandb.ai/maj90/mnist-unet/runs/xg3q5n39
- eps/eps, exact division: https://wandb.ai/maj90/mnist-unet/runs/xfc8d7un

Their saved configs record the actual historical settings.

For a rough f_mem curve alongside previews, add these flags to a new image run:

```bash
--generation-monitor --f-mem-monitor --f-mem-samples 512
```

The grid can still show 16 images while the metric uses 512. Both use prefixes
of one shared generated batch. The saved noise bank grows as needed. W&B shows
`generation/random_projected_pixels/f_mem`; local JSON records are saved in
`monitoring/f_mem_step_<step>.json`. This optional approximation uses resized,
random-projected pixels and rho=0.3. It is off by default and is not FID.
See `docs/config_reference.md` for preprocessing and interpretation limits.
