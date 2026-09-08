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

## Online monitoring demonstration

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
