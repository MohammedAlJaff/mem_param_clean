# mem_param_clean

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
