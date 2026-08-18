# Migration Journal

This journal tracks the clean repository build in `mem_param_clean`.

## 2026-08-10T15:28:28Z - Create clean repository shell

### Context
The old repository remains at:

```text
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par
```

A full pre-overhaul backup exists at:

```text
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par_preservation/pre_overhaul_20260810T145249Z
```

### Decision
Create a fresh local repository at:

```text
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_param_clean
```

Use fresh Git history. Insert files one class at a time. Do not bulk-copy the old tree.

### Current Skeleton
Keep only folders that are likely to survive:

```text
docs/
mem_param_src/
external/
notebooks/
jobs/berzelius/
```

Removed early placeholder folders:

```text
experiments/
scripts/
tests/
archive/
```

These can be created later only when a reviewed file class needs them.

### Files Added

```text
.gitignore
README.md
pyproject.toml
environment.yml
docs/objective.md
docs/repo_map.md
docs/preservation.md
docs/todo.md
mem_param_src/__init__.py
```

### Checks

```text
new repo size: 112K
mem_param_src/__init__.py syntax: ok
old repo git diff --check: ok
```

### Next Action
Review one file class from the old repo. Then copy only files with a clear role into the clean repo.


## 2026-08-12T14:17:44Z - Rename source root and record file classes

### Decision
Use `mem_param_src/` as the source root instead of `src/`.

The package root is now:

```text
mem_param_src/
```

### Known File Classes
The old repo contains these useful code and workflow classes:

- model and sampler code
- losses and metrics
- trainers
- data loaders, including synthetic data
- checkpoint replay and evaluation scripts
- artifact, logging, and tracking code
- visualization and figure-generation scripts
- Berzelius job files
- experiment sweep definitions
- tests
- notebooks
- baselines and third-party code

### Cleanup Rule
Keep the clean repo small. Move one class at a time after review. Do not create
final-looking folders before their contents are selected.


## 2026-08-12T14:20:00Z - Flatten package root

### Decision
Remove the nested `memparam/` folder. The source package now lives directly at:

```text
mem_param_src/
```

This keeps the clean repo shallow while the surviving code classes are still
being selected.


## 2026-08-12T14:44:40Z - Move first source core

### Decision
Move only the first source core into `mem_param_src/`:

```text
mem_param_src/
  model.py
  model_setup.py
  sampling.py
  networks/
    common.py
    mlp.py
    small_unet.py
    unet.py
```

### Cleanup
- Renamed old `gdfm/core.py` to `model.py`.
- Renamed old `gdfm/samplers.py` to `sampling.py`.
- Updated imports from `gdfm.*` to `mem_param_src.*`.
- Kept MLP and UNet network paths.
- Removed the JiT factory path for now because the clean source does not yet
  include the vendored JiT dependency.
- Removed the VP reverse-SDE sampler from the clean sampler surface.
- Removed runnable demo blocks from source modules.
- Made `GDFM.sample(...)` call the standalone sampler module so Euler and Heun
  live in one place.

### Checks
```text
AST parse: 9 source files ok
import smoke: ok
MLP GDFM loss/sample smoke: ok
UNet forward smoke: ok
```

### Next Action
Move the config/data/training layer only after the source core is accepted.


## 2026-08-12T14:57:16Z - Check conversions and equal weighting

### Decision
Make equal loss weighting explicit in `GDFM`:

```text
loss_weighting = "equal"
```

This binds `w(t) = 1`. It is still the default if the config does not provide
`loss_weighting`.

### Checks
```text
FM and VP x/eps/v conversion round trips: max error 1.669e-06
Endpoint clamp finite check: ok
Equal weighting check: w(t) shape (B, 1), all values 1.0
py_compile with cache redirected to /tmp: ok
```

### Note
Exact conversion checks use times away from the endpoints. Near `t = 0` or
`t = 1`, `denom_eps` clamps small denominators to keep outputs finite.


## 2026-08-12T15:01:38Z - Correct common-loss terminology

### Decision
Do not use `equal` to mean scalar loss weighting.

In the paper-control sense, "equal" means a common loss space. For the main
control this is the fixed velocity-loss row:

```text
x/v
eps/v
v/v
```

The scalar time weight option is now named:

```text
loss_weighting = "unit"
```

This means `w(t) = 1`.

### Checks
```text
Common velocity row x/v eps/v v/v: ok
Conversion round trips: max error 1.669e-06
loss_weighting="unit": ok
loss_weighting="equal": rejected as ambiguous
py_compile with cache redirected to /tmp: ok
```


## 2026-08-13T14:14:29Z - Add base config to clean source

### Decision
Add a minimal `mem_param_src.config` module using `ml_collections.ConfigDict`.
Do not add a local config replacement class.

### Cleanup
- Added `get_base_config`, `config_to_dict`, `clone_config`, and
  `apply_overrides`.
- Exported the config helpers from `mem_param_src`.
- Added `ml-collections` to `environment.yml` and `pyproject.toml`.
- Removed remaining VP hooks from `model.py`; the clean model is FM-only.
- Kept only MLP and UNet config fields. Removed JiT and VP config fields from
  the clean base config.

### Checks
```text
AST parse: 10 source files ok
py_compile with cache redirected to /tmp: ok
memcompgen import get_base_config + GDFM: ok
clone_config + apply_overrides + config_to_dict: ok
small MLP loss/sample smoke: ok
family="vp" direct check: raises ValueError
stale VP/JiT source scan: clean
```


## 2026-08-13T14:31:41Z - Audit clean metric and data split

### Decision
Keep metrics in two layers:

- Dataset-agnostic metric math.
- Dataset-specific metric code for synthetic geometries and image inputs.

### Findings
- `mem_param_src/metrics/metrics_images.py` mixes generic nearest-neighbor
  math, image preprocessing, feature encoders, random projections, FID, and
  checkpoint record formatting.
- `mem_param_src/metrics/metrics_synthetic_data/metrics_synthetic.py` mixes
  generic nearest-neighbor math with GMM-specific and torus-specific geometry
  metrics.
- The clean `data/` package exists, but several imports still point to old
  `mp_data.*` and `mp_config` names.
- `pyproject.toml` does not yet package `mem_param_src.metrics` or
  `mem_param_src.data`.

### Proposed Split
```text
mem_param_src/metrics/
  nearest_neighbors.py
  sliced_wasserstein.py
  image.py
  synthetic/
    gmm_subspace.py
    torus3d.py
```

### Checks
```text
py_compile for current metric and data files: ok
```


## 2026-08-13T14:49:31Z - Split synthetic metrics

### Decision
Split synthetic metrics into generic vector metrics and dataset-specific
synthetic metrics.

Sliced Wasserstein stays dataset-agnostic. It accepts two vector batches and
fixed projection directions. Image metrics must first convert images to pixel
vectors or encoder features before they call it.

### Cleanup
- Added `mem_param_src.metrics.nearest_neighbors` for `d1`, `d2`, `r`, and
  `f_mem`.
- Added `mem_param_src.metrics.sliced_wasserstein` for vector sliced-W2.
- Added `mem_param_src.metrics.synthetic.gmm_subspace` for GMM subspace
  projection, intrinsic GMM NLL, and intrinsic sliced-W2.
- Added `mem_param_src.metrics.synthetic.torus3d` for torus angle features and
  torus sliced-W2.
- Added `mem_param_src.metrics.synthetic.controlled` for the controlled
  synthetic metric entrypoints.
- Removed the old `metrics_synthetic_data` package.
- Changed package discovery in `pyproject.toml` so nested packages are included.
- Fixed `mem_param_src.data.__init__` so it imports from the clean package
  path.

### Checks
```text
compileall for mem_param_src: ok
setuptools find_packages includes metrics, metrics.synthetic, data: ok
system python runtime check: blocked by missing ml_collections
memcompgen runtime smoke for NN, sliced-W2, GMM metrics, torus metrics: ok
stale metrics_synthetic_data scan: clean
```


## 2026-08-13T15:00:19Z - Audit clean data and old runner inputs

### Findings
- The clean source has data loaders in `mem_param_src/data/` for MNIST,
  CelebA32 grayscale, Flowers102, AFHQ-v2, FFHQ128, GMM subspace, and torus3d.
- The clean data package also has shared helpers for seeded subset selection,
  optional flattening, bundle validation, and dataset dispatch.
- Direct `load_gmm_subspace_bundle(...)` works in the clean package.
- Public `build_data_bundle(...)` still fails because `common.py` imports old
  `mp_data.*` paths.
- MNIST, CelebA32, and image-collection modules also still import
  `mp_data.common`.

### Old Runner Requirements
- Training must save `train_tensor.pt`, `reference_tensor.pt`, and `z_init.pt`.
- Optional data artifacts are `train_labels.pt`, `reference_labels.pt`,
  `train_indices.pt`, `reference_indices.pt`, and `geometry.pt`.
- `data_meta.json` records dataset name, root, splits, shapes, source ids,
  class counts, leakage validation, batching, runtime, and seed policy.
- Synthetic evaluation requires saved `geometry.pt` and then recomputes
  intrinsic coordinates for train, reference, and generated samples.
- Image evaluation requires saved train/reference tensors and `z_init.pt`.

### Config And Experiment Sources
- Standard config shape lives in `src/mp_config/base.py`.
- Image runner defaults and dataset profiles live in
  `src/mp_runners/image_config.py`.
- Synthetic runner defaults live in `src/mp_runners/synthetic_train_runner.py`.
- Sweep definitions are mostly shell scripts plus TSV manifests under
  `jobs/job_sets/`.
- Repo YAML is mainly environment YAML, not experiment YAML.

### Checks
```text
direct clean GMM loader smoke: ok
clean build_data_bundle smoke: fails on old mp_data import
clean data module import sweep: fails on old mp_data import in image loaders
```


## 2026-08-13T15:26:05Z - Add GMM clean smoke notebook

### Decision
Use GMM subspace plus MLP as the first clean end-to-end notebook. This avoids
dataset downloads, image preprocessing, W&B, and cluster job code.

### Cleanup
- Fixed clean data imports from old `mp_data.*` and `mp_config` paths to
  `mem_param_src.*` paths.
- Added `notebooks/sanity/gmm_clean_smoke.ipynb`.
- The notebook uses one conceptual step per cell:
  config, data bundle, intrinsic plot, ambient heatmap, model build, loss
  check, pre-training sample plot, short training, post-training sample plot,
  metrics, nearest-neighbor link plot, and a small exercise.

### Checks
```text
clean build_data_bundle for gmm_subspace: ok
compileall for mem_param_src: ok
clean data module import sweep: ok
stale clean data mp_data/mp_config import scan: clean
nbconvert execute notebooks/sanity/gmm_clean_smoke.ipynb: ok
notebook code cells: 11
notebook execution errors: 0
```

### Notebook Output Snapshot
```text
parameters: 51,720
first loss: 1.318352
step 120 loss: 0.575702
ambient f_mem rho=0.5: 0.000000
intrinsic f_mem rho=0.5: 0.085938
intrinsic_gmm_nll_mean: 9.974727
intrinsic_sliced_w2: 1.146836
```


## 2026-08-13T15:56:58Z - End-of-day clean repo state

### State
- The clean source can run the first GMM end-to-end path.
- `mem_param_src.data.build_data_bundle(...)` works for `gmm_subspace`.
- Synthetic metrics are split into generic vector metrics and dataset-specific
  GMM/torus metrics.
- The executed notebook is
  `notebooks/sanity/gmm_clean_smoke.ipynb`.

### Verified Path
```text
config -> data bundle -> GDFM + MLP -> short training -> Heun sampling -> metrics
```

### Checks
```text
clean gmm_subspace build_data_bundle: ok
clean data imports: ok
compileall for mem_param_src: ok
gmm_clean_smoke.ipynb nbconvert execution: ok
notebook execution errors: 0
```

### Next Action
Keep the next cleanup narrow. Move only the smallest reusable training-runner
pieces needed to turn the notebook path into a scriptable smoke run.


## 2026-08-13T15:58:18Z - Working-location note

### Paths
```text
old source repo:
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par

clean target repo:
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_param_clean
```

### Recommendation
Open and work from the clean target repo. Use the old source repo as reference
material and port code from it only when needed.

### Reason
The clean repo already has the new package shape, clean source root, migration
journal, and GMM smoke notebook. Working from the old repo makes it easier to
keep old imports and extra code by accident.


## 2026-08-13T19:23:30Z - Split image metrics into a package

### Decision
Replace the 1,048-line `mem_param_src/metrics/metrics_images.py` module with a
focused image-metric package:

```text
mem_param_src/metrics/image/
  preprocessing.py  image layouts, value ranges, and RGB uint8 conversion
  features.py       callable and torch-fidelity feature encoders
  memorization.py   pixel- and feature-space memorization entrypoints
  projection.py     fixed random-projected pixel features and metrics
  fid.py            in-memory tensor FID adapter
  records.py        checkpoint-level generated-vs-held-out records
  __init__.py       public image-metric surface
```

### Cleanup
- Removed the duplicate nearest-neighbor implementation from the image code.
- Reused `mem_param_src.metrics.nearest_neighbors` as the only owner of `d1`,
  `d2`, `r = d1 / d2`, and `f_mem` math.
- Removed the old `metrics_images.py` module instead of retaining a legacy
  compatibility wrapper.
- Exported the image-metric API from both `mem_param_src.metrics.image` and
  `mem_param_src.metrics`.
- Promoted the reusable torch-fidelity wrapper to the public
  `TorchFidelityFeatureEncoder` name.
- Added focused image-metric tests under `tests/test_image_metrics.py`.

### Checks
```text
unittest image metrics: 11 passed
ruff image metrics and tests: passed
compileall mem_param_src and tests: passed
setuptools package discovery includes mem_param_src.metrics.image: passed
duplicate nearest-neighbor definition scan: clean
stale metrics_images import scan: clean
gmm_clean_smoke.ipynb nbconvert execution: passed
```

### Verification Limit
The local `memcompgen` environment does not contain `torch-fidelity`. The FID
tests therefore verify tensor conversion, dataset adaptation, arguments, and
result records with a mocked torch-fidelity boundary; they do not run a real
Inception feature extraction.

### Next Action
When the reusable image runner is migrated, import metrics from
`mem_param_src.metrics.image` and add one real torch-fidelity integration check
in the complete project environment.


## 2026-08-13T19:41:57Z - Add one operational TODO

### Decision
Use root-level `TODO.txt` as the single source of truth for unfinished work.
Keep `MIGRATION_JOURNAL.md` as the record of decisions, commands, checks, and
completed work.

### Cleanup
- Replaced the generic numbered list in `docs/todo.md` with prioritized task
  IDs, concrete next actions, and explicit completion conditions.
- Removed `docs/todo.md` to avoid maintaining two backlogs.
- Updated `README.md` and `docs/repo_map.md` to point to `TODO.txt`.
- Recorded confirmed open work: initial Git commit, dependency declaration,
  real torch-fidelity integration, reusable training and artifact paths,
  checkpoint replay, image-loader validation, scientific image-metric
  validation, tests, tracking/visualization, jobs, and external baselines.

### Checks
```text
TODO task IDs T001-T012 unique: passed
stale live docs/todo.md links: clean
whitespace checks for changed documentation: passed
```


## 2026-08-16T14:38:47Z - Separate training and sampling time controls

### Decision
Use plain, separate settings for training and sampling:

```text
training.start_time
training.end_time
training.use_safe_division
training.denominator_epsilon

sampling.start_time
sampling.end_time
sampling.use_safe_division
sampling.denominator_epsilon
sampling.n_steps
sampling.method
sampling.output
```

Training times are now sampled uniformly between the two training limits.
Logit-normal time sampling and its configuration were removed.

Sampling calls must receive every sampling choice explicitly. There are no
fallbacks to the training range, the old schedule section, or numeric defaults.
The step count creates equal intervals from sampling start to sampling end.

### Safe Division
Both training conversions and sampling conversions can use either:

- exact division, when `use_safe_division = False`; or
- sign-preserving denominator clamping, when `use_safe_division = True`.

The clamp changes the conversion equation whenever a denominator magnitude is
below `denominator_epsilon`. Choosing experiment ranges and clamp sizes remains
an explicit scientific TODO rather than a hidden implementation detail.

Velocity-to-noise conversion now uses the direct two-equation solution. It no
longer reconstructs clean data and then divides by `1 - t`, removing an
unnecessary source of instability near the data end of the path.

### Sampling Output
`sampling.output = "state"` returns the integrated state after the final step,
at `sampling.end_time`.

`sampling.output = "clean_x"` is a reserved configuration choice that currently
raises `NotImplementedError`. Defining and testing that conversion is recorded
as T013 in `TODO.txt`.

### Checks
```text
pytest full suite: 21 passed
new model/sampling tests: 10 passed
ruff model/sampling tests: passed
compileall mem_param_src and tests: passed
notebook JSON validation: passed
gmm_clean_smoke.ipynb nbconvert execution: passed
executed notebook error-output scan: clean
stale t_min/t_max/denom_eps/time_sampling config scan: clean
```


## 2026-08-16T14:49:58Z - Rename sampling outputs plainly

### Decision
Replace the temporary sampling-output names with:

```text
sampling.output = "final_sample"
sampling.output = "clean_prediction"
```

`final_sample` means the tensor returned after the last integration step at
`sampling.end_time`.

`clean_prediction` means a model-derived prediction of clean data at the end.
It remains deliberately unimplemented and raises `NotImplementedError`.

Do not retain aliases for the old names in the clean repository.

The safe-division flags remain:

```text
training.use_safe_division
sampling.use_safe_division
```

### Checks
```text
pytest full suite: 21 passed
ruff changed Python files: passed
notebook JSON validation: passed
gmm_clean_smoke.ipynb nbconvert execution: passed
stale output-name scan in live code/docs: clean
```


## 2026-08-16T15:01:27Z - Add the complete configuration reference

### Decision
Use `docs/config_reference.md` as the human-readable meaning and guidance
reference for the clean configuration. Keep `mem_param_src/config.py` as the
programmatic defaults and test that every base-config leaf appears in the
reference.

The reference separates:

- active clean settings;
- settings declared for a runner that has not yet been migrated;
- values that should be derived rather than set manually;
- named but unimplemented behavior;
- legacy runner defaults and CLI flags that are not active in the clean repo.

### Old Repository Review
Reviewed the working old repository at:

```text
/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par
```

The preservation repository remained untouched.

Sources reviewed included the old base config, image config builder, image and
synthetic train runners, synthetic replay runner, memorization-boundary
evaluator, checkpoint schedule, numerical runtime helper, and runner README.

Legacy values are labelled as historical guidance rather than silently
promoted to scientific recommendations. This matters because the old runners
used different solver-step defaults for images, synthetic data, and the
memorization-boundary evaluator.

### Checks
```text
current clean config leaf fields documented: 101/101
configuration and legacy table entries: 223
pytest full suite: 22 passed
ruff config-reference test: passed
documentation trailing-whitespace scan: clean
README and repository-map links: added
```


## 2026-08-16T15:04:55Z - Pair time limits with safe-division epsilon

### Decision
Change both training and sampling defaults to:

```text
start_time = 0.01
end_time = 0.99
use_safe_division = true
denominator_epsilon = 0.01
```

Training and sampling remain independently configurable even though their
current defaults match.

For the linear FM path, the smallest ideal coefficient over `[0.01, 0.99]` is
`0.01`. Matching the clamp threshold to that value makes safe division a
boundary guard rather than intentionally altering a broad part of the default
time interval. Floating-point rounding can still trigger a tiny clamp at an
endpoint, so exact-versus-safe validation remains open as T014.

### Checks
```text
pytest full suite: 22 passed
ruff changed Python files: passed
notebook JSON validation: passed
gmm_clean_smoke.ipynb nbconvert execution: passed
executed notebook error-output scan: clean
config reference and TODO: updated
```


## 2026-08-16T15:53:18Z - Consolidate end-of-day state and verify local image datasets

### Purpose
This entry summarizes the work completed on 2026-08-16. The detailed numerical
and configuration decisions remain in the preceding entries; this is the
single end-of-day state summary.

### Repository Guidance
- Re-read the root `AGENTS.md` and adopted it as the canonical instruction file
  for work in this repository.
- Confirmed that its repository-root location makes it apply to future Codex
  sessions working in this tree. Do not create a duplicate instruction file.
- The active working rules are: state assumptions, prefer minimal and surgical
  changes, preserve user work, define verification criteria, keep the clean
  repository small, separate the journal from the operational TODO, and use
  precise research terminology.

### Numerical And Sampling State
- Replaced ambiguous `t_min` and `t_max` names with independent training and
  sampling start/end times.
- Training times are sampled uniformly over the configured training interval.
  Logit-normal training-time sampling was removed.
- Training and sampling each have an explicit `use_safe_division` flag and
  `denominator_epsilon` value.
- Exact division remains available. Safe division uses sign-preserving
  denominator clamping.
- Both current time intervals are `[0.01, 0.99]`; both current denominator
  epsilons are `0.01`.
- Sampling step count controls the equal numerical intervals between the
  sampling start and end times.
- `sampling.output = "final_sample"` returns the state after the last
  integration step at `sampling.end_time`.
- `sampling.output = "clean_prediction"` is a named configuration choice that
  deliberately raises `NotImplementedError`. Its implementation remains T013
  in `TODO.txt`.
- Exact-versus-safe division validation remains T014. Non-uniform time sampling
  should be reconsidered only for a stated experiment and remains T015.

### Configuration Reference
- Added `docs/config_reference.md` as the human-readable reference for every
  active clean configuration field.
- Documented all 101 clean config leaves and recorded 223 clean and legacy
  configuration entries.
- Reviewed old image, synthetic, replay, evaluation, checkpoint, and numerical
  runner settings as historical guidance rather than scientific defaults.

### Python Test Environment
- Installed `pytest` in the existing `memcompgen` Conda environment without
  changing repository dependencies or model code.
- Verified `pytest 9.1.1` and ran the complete current test suite.

### Legacy Data-Path Review
The old working repository used a relative data root of `data`. On the cluster,
the effective root recorded by the old dataset notebook was:

```text
/proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_par/data
```

The old FFHQ path was `data/ffhq/thumbnails128x128`, and the old AFHQ-v2 path
was `data/afhq_v2/{train,test}`. The local old checkout currently has only a
small MNIST cache under `tmp/data/MNIST/raw`; its top-level `data/` directory is
absent because data paths are ignored by Git.

### Clean Local Image Datasets
Downloaded these datasets into the ignored `mem_param_clean/data/` directory:

```text
MNIST             train=60,000  test=10,000
CIFAR-10          train=50,000  test=10,000
Oxford Flowers102 train=1,020   val=1,020   test=6,149
```

The local data directory occupies approximately `1.1G`, including retained
download archives. FFHQ was not downloaded. The official 128x128 release is a
separate approximately 1.95 GB, 70,000-image download with a non-commercial
CC BY-NC-SA license.

Added and executed:

```text
notebooks/sanity/image_datasets_download_check.ipynb
```

The notebook reloads the three public datasets with `download=False`, asserts
the official split sizes, decodes and displays example images, and reports the
FFHQ legacy path separately.

### Checks
```text
pytest version in memcompgen: 9.1.1
pytest full suite: 22 passed
public dataset split checks: 7 passed
dataset notebook structure: 12 cells, valid nbformat
dataset notebook execution: 5/5 code cells executed
dataset notebook error outputs: 0
FFHQ image count at clean legacy path: 0
model code changed for dataset setup: no
repository dependency files changed for dataset setup: no
```


## 2026-08-16T16:02:23Z - Finalize the agent session protocol

### Decision
Keep `AGENTS.md` at the repository root as the single canonical agent guide.
Add a short session-start protocol so future work begins with the same project
state and does not treat the operational TODO as automatic authorization.

### Guidance Added
- Read the root guide, inspect `git status`, and consult `TODO.txt` plus the
  latest relevant journal entry before editing.
- Use the current user request as the session scope.
- Use `config.py`, `config_reference.md`, `TODO.txt`, and the migration journal
  for their separate executable-default, field-meaning, unfinished-work, and
  completed-decision roles.
- Record the environment, check, result, and limitation for verification.
- Distinguish syntax checks, mocked tests, runtime tests, and integrations.
- Use the existing clean modules as style examples instead of referring to the
  absent `PYTHON_STYLE.md` and `prototyping/gdfm_components.py` files.
- Use `PYTHONPATH=.` for an uninstalled clean checkout.
- Distinguish the local `memcompgen` environment from the Berzelius
  `mem_param_env` environment and record which one produced each result.

### Checks
```text
AGENTS.md root location: confirmed
stale PYTHONPATH=src instruction: removed
missing style-file references: removed
local and cluster environment names: explicit
session scope and TODO authorization rule: explicit
trailing-whitespace scan: passed
```


## 2026-08-18T15:49:43Z - Separate dataset storage from dataset-loader code

### Decision
Keep downloaded dataset files under the repository-root `data/` directory and
rename the first-party source package from `mem_param_src/data/` to
`mem_param_src/datasets/`.

The naming boundary is now explicit:

```text
data/                       local downloaded datasets; not tracked by Git
mem_param_src/datasets/     dataset definitions, generation, and loader code
```

Anchor the ignore rule as `/data/`. This excludes only the repository-root data
store and allows Git to include `mem_param_src/datasets/`. Update active Python
imports, the GMM sanity notebook, and the repository map. Historical journal
entries retain the paths that were accurate when those checks were performed.

### Checks
```text
environment: local memcompgen Conda environment
pytest full suite: 22 passed
ruff renamed package and dependent synthetic metric: passed
compileall mem_param_src and tests: passed
renamed-package import and GMM bundle runtime check: passed
notebook JSON validation: passed
root data/ ignore check: passed
mem_param_src/datasets/ inclusion check: passed
limitation: notebooks and real image loaders were not executed in this change
```
