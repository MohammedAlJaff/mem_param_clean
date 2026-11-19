# Migration Journal

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
training.use_safe_div
training.denominator_epsilon

sampling.start_time
sampling.end_time
sampling.use_safe_div
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

- exact division, when `use_safe_div = False`; or
- sign-preserving denominator clamping, when `use_safe_div = True`.

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
training.use_safe_div
sampling.use_safe_div
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
use_safe_div = true
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
- Training and sampling each have an explicit `use_safe_div` flag and
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


## 2026-08-18T15:57:45Z - Create the first reviewed Git baseline

### Decision
Create root commit `d3b26f3` (`Create initial research baseline`) on `main` as
the first identifiable code state for the clean study.

The snapshot contains 47 reviewed files: first-party source, configuration,
documentation, tests, and the two sanity notebooks. It excludes downloaded
datasets, Python and tool caches, experiment outputs, checkpoints, W&B files,
and run directories. Remove T001 from `TODO.txt`; dependency reconstruction
remains open as T002 and real torch-fidelity integration remains open as T003.

### Checks
```text
environment: local memcompgen Conda environment
credential-pattern scan of upload set: clean
raw dataset and generated-artifact scan of upload set: clean
largest staged file: 464 KiB
git staged whitespace check: passed
pytest full suite: 22 passed
ruff mem_param_src and tests: passed
compileall mem_param_src and tests: passed
notebook JSON validation: passed
git status immediately after root commit: clean
limitation: declared dependencies are not yet sufficient for a fresh environment
```


## 2026-08-18T19:03:33Z - Declare and reconstruct the complete local environment

### Decision
Use `pyproject.toml` as the authoritative Python dependency declaration. Keep
PyTorch and `ml-collections` in the core package, and separate optional
dependencies into `image`, `evaluation`, and `dev` profiles. Do not add a
duplicate `requirements.txt`.

Use `environment.yml` to create Python 3.10 and install the repository in
editable mode with all three profiles:

```text
.[image,evaluation,dev]
```

Keep PyTorch and Torchvision platform-neutral so local CPU/MPS and cluster CUDA
builds can satisfy the same package contract. Pin Ruff to the validated 0.14.8
policy because a newer Ruff release changed its default findings for unchanged
source code.

Remove T002 from `TODO.txt`. T003 now starts from an environment containing
torch-fidelity but still requires a real Inception feature and FID integration
check.

### Fresh-Environment Checks
```text
environment: /tmp/mem_param_clean_t002_env created from environment.yml
platform: macOS arm64
Python: 3.10.20
torch: 2.13.0
torchvision: 0.28.0
torch-fidelity: 0.4.0
editable mem_param_src import: passed
pip dependency consistency check: passed
pytest full suite: 22 passed
ruff 0.14.8 mem_param_src and tests: passed
compileall mem_param_src and tests: passed
GMM sanity notebook execution: 12/12 non-empty code cells executed
executed notebook error-output scan: 0
limitation: Berzelius and real torch-fidelity feature extraction were not tested
```


## 2026-08-18T19:08:52Z - Run the real torch-fidelity integration

### Decision
Complete T003 using the repository's real `TorchFidelityFeatureEncoder` and
`compute_image_fid` adapters with torch-fidelity 0.4.0. Remove T003 from
`TODO.txt`.

The check used two distinct deterministic tensor batches generated with seed
37. Each batch contained three RGB images with shape `[3, 3, 32, 32]` and
values in `[0, 1]`. The integration ran on CPU with
`inception-v3-compat`, caching downloaded weights only under `/tmp`.

### Checks
```text
environment: /tmp/mem_param_clean_t002_env
platform: macOS arm64, CPU
torch: 2.13.0
torchvision: 0.28.0
torch-fidelity: 0.4.0
feature adapter spec: torch_fidelity:inception-v3-compat:2048
feature result: shape [2, 2048], all values finite
FID tensor batches: n_samples=3, n_reference=3
FID result: 89.5920389865937, finite
Inception weights: weights-inception-2015-12-05-6726825d.pth
Inception weights size: 96 MiB
Inception weights SHA-256: 6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2
mocks used: none
limitation: the tiny random batches validate integration, not scientific FID quality
```


## 2026-08-18T19:28:13Z - Add a visual FID walkthrough on real image datasets

### Decision
Add `notebooks/sanity/fid_cifar_flowers_walkthrough.ipynb` as an executed,
research-facing explanation of the repository's real `compute_image_fid` path.
Use deterministic 32-image subsets from CIFAR-10 train, CIFAR-10 test, and
Flowers102 validation, render the exact tensors, and keep the Inception
extractor, preprocessing, sample count, batch size, and CPU device fixed across
comparisons.

Treat the resulting FID values as a visual runtime diagnostic rather than
dataset-level estimates. The notebook explicitly separates distributional FID
from nearest-neighbor memorization measurements and explains that a tiny
negative identical-input FID is floating-point roundoff around zero.

Declare `scipy>=1.8` in the `image` optional dependency profile because
Torchvision's Flowers102 loader requires `scipy.io.loadmat` for the official
split metadata.

### Checks
```text
environment: /tmp/mem_param_clean_t002_env, macOS arm64, CPU
Python: 3.10.20
torch: 2.13.0
torchvision: 0.28.0
torch-fidelity: 0.4.0
scipy: 1.15.3
dataset loading: CIFAR-10 train/test and Flowers102 val passed with download=False
notebook execution: 11/11 code cells executed, 0 error outputs
identical CIFAR control FID: -0.00018111240910911874, interpreted as numerical zero
CIFAR train-vs-test FID: 247.9516615643539
CIFAR train-vs-Flowers val FID: 359.8505496497645
embedded figure inspection: sample grid and FID comparison chart passed
editable dependency installation: passed
pip dependency consistency check: passed
pytest full suite: 22 passed
git diff whitespace check: passed
limitation: N=32 estimates demonstrate behavior but are not scientific FID estimates
```


## 2026-08-18T19:35:19Z - Make notebook code locally understandable

### Decision
Adopt a repository notebook-commenting rule: every code cell starts with a
plain-language purpose comment and includes concise comments beside or directly
above non-obvious operations. Standalone Markdown remains useful for equations
and interpretation, but it is not the only explanation of runnable code.

Apply the rule to every code cell in
`notebooks/sanity/fid_cifar_flowers_walkthrough.ipynb`. Preserve the notebook's
cell order, experiment protocol, outputs, and research interpretation.

### Checks
```text
AGENTS.md notebook rule: added
FID notebook code cells with top purpose comments: 11/11
FID notebook code-cell syntax compilation: passed
FID notebook execution: 11/11 code cells executed, 0 error outputs
embedded figures: 2
deterministic FID values: unchanged
notebook JSON validation: passed
git diff whitespace check: passed
TODO.txt: not changed by this task
```


## 2026-08-18T19:50:42Z - Apply ASD-STE100 to notebook text

### Decision
Use ASD-STE100 Simplified Technical English, Issue 9, for notebook text. Apply
the standard to titles, Markdown, instructions, descriptions, and code
comments.

Use a maximum of 20 words in an instruction. Use a maximum of 25 words in a
descriptive sentence. Use active voice and one topic in each sentence. Keep
exact research and code terms as subject-field technical terms.

Revise all text in `notebooks/sanity/fid_cifar_flowers_walkthrough.ipynb`.
Keep the code, protocol, results, cell order, and figures constant.

### Checks
```text
ASD-STE100 source: official Issue 9, 2025-01-15
notebook code cells with top purpose comment: 11/11
sentence-length audit: 0 violations
passive-voice heuristic: 0 candidates
notebook code-cell syntax compilation: passed
notebook execution: 11/11 code cells executed, 0 error outputs
embedded figures: 2
deterministic FID values: unchanged
notebook JSON validation: passed
git diff whitespace check: passed
certified ASD-STE100 checker or trained human review: not used
TODO.txt: not changed by this task
```


## 2026-08-18T20:20:00Z - Rename the safe-division denominator field

### Decision
Rename the active `denominator_epsilon` field to `safe_div_denominator_floor`. Apply a
strict rename with no compatibility alias. Update the model API, sampling API,
configuration, tests, documentation, TODO wording, and the GMM sanity notebook.

Audit all six target-space conversions in
`/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_param_clean/mem_param_src/model.py`.
Compare them with the reference implementation in
`/Users/mohammadaljaff/Desktop/PhD/projects/mem_param/mem_par/src/gdfm/core.py`.

The old `v`-to-`eps` conversion first recovered `x`. It then converted `x` to
`eps`, which divided by `b(t) = 1 - t`. The clean model already uses the direct
two-equation solution:

```text
eps_hat = (a'(t) z_t - a(t) v_hat) / (a'(t)b(t) - a(t)b'(t))
```

For the linear FM path, the determinant is one. This direct conversion avoids
an unnecessary division by `1 - t`. No conversion formula needed a change in
this task.

The `x`-to-`v` conversion still requires division by `b(t)`. The `eps`-to-`v`
conversion still requires division by `a(t)`. Algebraic expansion does not
remove those singular denominators.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
focused model and sampling tests: 10 passed
full pytest suite: 22 passed
Ruff on changed Python files: passed
GMM sanity notebook execution: passed, output written under /tmp
notebook JSON validation: passed
active-code search for denominator_epsilon: zero matches
git diff whitespace check: passed
limitation: T014 remains open; this task did not select the experiment epsilon
```


## 2026-08-18T20:58:09Z - Add the simplest image runner prototype

### Decision
Add and execute
`notebooks/prototypes/prototype_simplest_runner.ipynb`. Keep the runner linear
and visible. Do not add helper functions or a runner class.

Use deterministic Flowers102 train and validation subsets. Train a small
two-level UNet in the `v` prediction and loss spaces. Run two epochs over 64
training images with Adam. Generate eight images from fixed noise with ten Heun
steps.

Save the exact data bundle, initial noise, checkpoint, samples, and figures in
`outputs/prototypes/simplest_runner_flowers102/`. The output directory remains
Git-ignored.

The first notebook execution could not import `mem_param_src`. Jupyter started
the kernel in the notebook directory. The setup cell now finds the repository
root and adds it to the import path before project imports.

### Checks
```text
environment: memcompgen, macOS arm64, MPS
notebook template: experiment
code cells: 8/8 executed
error outputs: 0
optimizer steps: 16
epoch 1 mean loss: 1.161882
epoch 2 mean loss: 0.909726
first step loss: 1.309464931488037
last step loss: 0.8198249340057373
sample shape: [8, 3, 32, 32]
finite samples: true
saved artifact files: 7
notebook JSON validation: passed
code-cell syntax compilation: passed
top purpose comments: 8/8
git diff whitespace check: passed
limitation: this short run validates execution, not image quality
```


## 2026-08-19T16:45:00Z - Run the safe-division diagonal sanity experiment

### Decision
Add and execute
`notebooks/experiments/safe_division_diagonal_sanity.ipynb`. Use Flowers102 and
train all six labeled diagonal conditions: `x/x`, `eps/eps`, and `v/v`, with
safe division disabled and enabled.

Use the same initialization, images, noise, and time values for each
exact-safe pair. Use 128 training images, 64 reference images, batch size 8,
100 optimizer steps, Adam with learning rate `1e-3`, and no weight decay. Use a
two-level UNet with 32 base channels and 669,251 parameters.

Sample 16 fixed initial tensors with Heun, 20 steps, and the time interval
`[0.01, 0.99]`. Set `safe_div_denominator_floor = 0.01`. Compare exact division with
safe epsilon values from `1e-6` through `5e-2` in the numerical sweep.

The safe-division flag did not change diagonal training. Each exact-safe pair
had equal loss values and equal final parameter tensors. Safe division with
epsilon `0.01` also produced samples that were equal to exact division. This
result is expected because `t` and `1 - t` do not fall below `0.01` in the
sampling interval.

Safe epsilon `0.05` changed the `x` and `eps` conversion results inside the
active interval. The maximum conversion RMSE was approximately `0.80`. Do not
use `0.05` with this interval. Keep `0.01` as the in-house boundary guard.

The `eps/eps` samples were numerically large after 100 steps. Their RMS was
`11.92`, compared with `0.43` for `x/x` and `0.45` for `v/v`. This short run
does not establish sampling quality or paper settings.

### Checks
```text
environment: memcompgen, macOS arm64, MPS
notebook template: experiment
code cells: 11/11 executed
error outputs: 0
training conditions: 6
checkpoints: 24
exact-safe loss maximum absolute difference: 0.0 for x, eps, and v
exact-safe parameter maximum absolute difference: 0.0 for x, eps, and v
exact-safe sample RMSE: 0.0 for x, eps, and v
saved run directory size: 85 MB
notebook JSON validation: passed
code-cell syntax compilation: passed
top purpose comments: 11/11
limitation: one seed and 100 steps provide in-house evidence only
limitation: T014 remains open until the paper time ranges are fixed
```


## 2026-08-19T16:56:05Z - Name sampling outputs and cross division controls

### Decision
Replace the sampling output names with `x_t_max` and `x_1`. Do not keep aliases
for `final_sample` or `clean_prediction`.

`x_t_max` returns the integrated solver state at `sampling.end_time`. The model
equations still call this interpolated state `z_t`. The output API uses the
selected sampling terminology without changing the internal equations.

`x_1` evaluates the model at the final state and time. It converts that
prediction to clean-data space with `model_to_x`. It does not integrate from
`t_max` to one.

Keep the two safe-division controls independent. Training reads
`training.use_safe_div`. Inference reads `sampling.use_safe_div`. Do
not add a duplicate inference configuration section.

Expand `notebooks/experiments/safe_division_diagonal_sanity.ipynb` to cross
both controls. Each of the six trained models now uses exact and safe sampling.
The result record contains 12 sampling conditions and names both Boolean flags.
Remove T013 because both sampling outputs now have explicit tested behavior.

### Checks
```text
environment: memcompgen, macOS arm64, CPU and MPS
focused sampling tests: 10 passed, 4 subtests passed
Euler x_t_max behavior: passed
Heun x_t_max behavior: passed
Euler x_1 behavior: passed
Heun x_1 behavior: passed
full pytest suite: 22 passed, 4 subtests passed
Ruff on mem_param_src and tests: passed
safe-division notebook code cells: 11/11 executed
safe-division notebook error outputs: 0
training conditions: 6
sampling conditions: 12
training-flag sample maximum absolute difference: 0.0 for all conditions
sampling-flag sample maximum absolute difference: 0.0 for all conditions
notebook JSON validation: passed for all three edited notebooks
git diff whitespace check: passed
limitation: the experiment uses x_t_max; x_1 has focused runtime tests only
limitation: T014 remains open until the paper time ranges are fixed
```


## 2026-08-20T15:08:46Z - Add standalone image and synthetic runners

### Decision
Add four installed commands: `mem-param-image-train`,
`mem-param-synthetic-train`, `mem-param-image-eval`, and
`mem-param-synthetic-eval`. Keep the optimizer loop and checkpoint replay shared
between the two data domains. Keep each command-line module thin.

Require training configs to define `get_config()`. Apply repeatable `--set`
overrides only to existing fields. Standalone evaluation requires a saved run
directory. It can also accept an evaluation config and `--set` overrides under
only `sampling` and `evaluation`. The setting order is saved run, evaluation
config, then command-line override.

Save one shallow run directory with the resolved config, complete CPU data
bundle, fixed `z_init`, checkpoints, and training history. Synthetic runs also
save the exact sliced-Wasserstein directions. Evaluation reads these artifacts
without rebuilding the dataset. It records metrics for every checkpoint and
saves the complete generated tensor only for the final checkpoint.

Use `x_1` as the default sampling output. Keep `x_t_max` as the integrated-state
diagnostic. Support explicit, interval, and `bonnaire-1m` checkpoint schedules.
Do not add resume, optimizer-state checkpoints, W&B, live monitors, plots, FID,
feature metrics, sweeps, or attempt hierarchies in this runner version.

Remove T004, T005, and T006 after the saved-run and replay checks pass. Keep
T007, T008, T010, and T014 open.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
baseline before runner work: 22 passed, 4 subtests passed
final full pytest suite: 30 passed, 4 subtests passed
Ruff lint on mem_param_src, configs, and tests: passed
Ruff format check on new runner/config/test files: passed
Python compileall on mem_param_src, configs, and tests: passed
config-reference synchronization test: passed
git diff whitespace check: passed
editable installation: passed with --no-deps --no-build-isolation
installed command help checks: 4/4 passed
installed synthetic train and standalone eval commands: exit 0
installed Flowers102 train with --evaluate-after-training: exit 0
GMM runtime: 4 steps, 3 checkpoints, 3 metric records
GMM final x_1 tensor: [16, 8], finite
Flowers102 runtime: download=False, 2 steps, 3 checkpoints, 3 metric records
Flowers102 final x_1 tensor: [4, 3, 32, 32], finite
named exact-division evaluation: passed for GMM and Flowers102
deterministic saved-run replay test: exact tensor and metric equality passed
dataset rebuild guard during evaluation: passed
limitation: smoke budgets verify execution, not scientific model quality
limitation: image loaders other than Flowers102 still need end-to-end checks
limitation: repository-wide Ruff formatting still reports older source files
limitation: T014 remains open; no paper safe-division setting was selected
```


## 2026-08-20T16:13:36Z - Persist finite dataset partition identity

### Decision
Upgrade saved runs to artifact format version 2. Save the exact post-transform
training and reference tensors in `data_bundle.pt`. Add `data_manifest.json` as
the readable record of source pools and selected partition members.

Record stable sample identifiers in exact tensor order for every image loader.
Use split-relative integer indices for MNIST, CelebA32, and Flowers102. Use
relative file paths for AFHQ v2 and FFHQ. Record source sizes and an SHA-256
value for each ordered source catalog. Keep source indices when the loader has
them.

Record SHA-256 values for the complete tensor bundle and each saved tensor.
The catalog hash identifies the ordered identifier catalog. It is not a hash
of all raw dataset bytes. The tensor hashes identify the exact selected values
that training and evaluation use.

Treat `reference` as a selected held-out partition. Do not imply that it is the
complete complement of `train`. Do not copy unused source tensors into each
run. Keep the existing `Dataset` and `DataLoader` training path unchanged.

Require complete finite-dataset identity metadata before image training.
Validate saved manifest membership during standalone evaluation. Evaluation
continues to read the saved tensor bundle and does not rebuild the source
dataset.

Do not add a dataset cache, a new dataset abstraction, or a new TODO item.
Version-1 runner artifacts are not compatible with the version-2 replay
contract.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
full pytest suite: 33 passed, 4 subtests passed
Ruff lint on mem_param_src, configs, and tests: passed
Ruff format check on changed dataset, runner, and runner-test files: passed
Python compileall on mem_param_src, configs, and tests: passed
git diff whitespace check: passed
partition identity repeatability and same-source disjointness tests: passed
artifact bundle, manifest, tensor, and label SHA-256 checks: passed
real MNIST loader with download=False: identity validation passed
real Flowers102 loader with download=False: identity validation passed
installed Flowers102 training with evaluation: exit 0
installed Flowers102 standalone named exact-division evaluation: exit 0
Flowers102 runtime: 2 steps, checkpoints 0/1/2, default and named evaluations
dataset rebuild during standalone evaluation: not used by design and regression tested
limitation: selected tensor hashes do not hash unused raw source files
limitation: live AFHQ v2, FFHQ, and CelebA32 catalogs were not available for runtime checks
```


## 2026-08-20T16:48:03Z - Execute a 10k Flowers102 runner sanity notebook

### Decision
Add `notebooks/sanity/flowers102_runner_artifact_sanity.ipynb`. Run the
installed `mem-param-image-train` command as one blocking subprocess. Continue
the notebook only after training and checkpoint evaluation finish.

Use eight Flowers102 training images at `64 x 64` resolution. Keep the tested
669,251-parameter UNet with `v` prediction and `v` loss. Use Adam, batch size
eight, learning rate `1e-3`, and 10,000 optimizer updates. Save checkpoints at
steps 0, 2,500, 5,000, 7,500, and 10,000.

Replay all five checkpoints with one saved 64-sample noise bank. Use 50 Heun
steps and return `x_1`. Display the exact eight-image training tensor and the
64 final samples with `torchvision.utils.make_grid`.

After the runner returns, verify the complete artifact tree, resolved config,
run state, tensor shapes, stable sample identifiers, manifest hashes,
checkpoint schedule, loss history, metric records, and final sample tensor.
Do not rebuild the Flowers102 dataset during these checks.

Keep the notebook linear. It contains no function or class definitions. Start
every code cell with a purpose comment and apply the repository notebook text
rules.

### Result
The full MPS run finished in 11.67 minutes. The first recorded loss was
`1.338079`. The final recorded loss was `0.030367`. All losses and all 64 final
samples were finite. The saved sample range was `[-1.115, 1.126]` before the
display clamp.

Pixel-space `f_mem` values at steps 0, 2,500, 5,000, 7,500, and 10,000 were
`0.0`, `0.03125`, `0.0`, `0.21875`, and `0.265625`. The final value represents
17 of 64 generated samples at `rho = 0.5`. Held-out `f_mem` stayed zero. Mean
`r` changed from `0.995849` at step zero to `0.688950` at step 10,000.

The final grid visibly concentrates on the eight training-image modes. This
single small run verifies runner behavior. It does not establish a paper-scale
memorization result or a parameterization comparison.

The complete run directory is
`outputs/sanity/flowers102_runner_64_n8_10k_20260820T162900Z` and uses 22 MB.
The directory remains excluded from Git. The executed notebook contains the
plots and compact evidence outputs.

### Checks
```text
environment: memcompgen, macOS arm64, MPS
two-update notebook preflight: 11/11 code cells executed, zero errors
full notebook execution: 11/11 code cells executed, zero errors
runner return code: 0
runner state: evaluated
optimizer updates: 10000
checkpoint files: 5/5
checkpoint metric records: 5/5
final x_1 tensor: [64, 3, 64, 64], finite
data-manifest full hash validation: passed
artifact-tree assertions: passed
torchvision training and sample grids: rendered and visually inspected
nbformat validation: passed
code-cell syntax compilation: passed
top purpose comments: 11/11
notebook function and class definitions: 0
notebook text line-length check: passed
full pytest suite: 33 passed, 4 subtests passed
git diff whitespace check: passed
limitation: one seed and eight training images provide integration evidence only
limitation: current image metrics use pixels and do not establish perceptual identity
```


## 2026-08-20T18:59:24Z - Standardize the configured f_mem threshold

### Decision
Rename `evaluation.ratio_threshold` to `evaluation.f_mem_threshold`. Set its
canonical value to `0.3`. Do not keep the old name as an alias. Keep `rho` as
the mathematical symbol and as the metric-record key.

Keep the image sensitivity fields at `rho = 0.3`, `0.5`, and `0.7`. The
configured threshold selects the canonical `f_mem` field. The other fields are
diagnostics and do not change the primary threshold.

Reject a resolved config that has an `evaluation` section but lacks
`evaluation.f_mem_threshold`. This prevents old saved settings from silently
using a new default. Keep the earlier `rho = 0.5` Flowers run unchanged as
historical evidence.

Training and sampling still enable safe division by default with
`safe_div_denominator_floor = 0.01`. The Flowers sanity run uses `v` prediction and `v`
loss. Its training conversion and sampling velocity conversion do not divide.
The final linear-path conversion from `v` to `x_1` has denominator one. Safe
division is therefore enabled but does not clamp any conversion in this run.

### Result
Re-execute the Flowers sanity notebook with the canonical threshold. The new
MPS run finished in 12.05 minutes. Its final canonical `f_mem` was `0.015625`
at `rho = 0.3`. The auxiliary final values were `0.265625` at `rho = 0.5` and
`0.484375` at `rho = 0.7`. The generated grid remains concentrated around the
eight training images.

The complete new run directory is
`outputs/sanity/flowers102_runner_64_n8_10k_20260820T184554Z`. It remains
excluded from Git.

### Checks
```text
environment: memcompgen, macOS arm64, CPU and MPS
targeted runner, metric, config, and sampling tests: 34 passed, 4 subtests passed
full pytest suite: 34 passed, 4 subtests passed
Ruff on mem_param_src and tests: passed
GMM sanity notebook: 11/11 code cells executed, zero errors
GMM top purpose comments: 11/11
Flowers sanity notebook: 11/11 code cells executed, zero errors
Flowers top purpose comments: 11/11
Flowers runner return code: 0
Flowers run state: evaluated
optimizer updates: 10000
checkpoint files: 5/5
checkpoint metric records: 5/5
metric record rho: 0.3 for every checkpoint
final x_1 tensor: [64, 3, 64, 64], finite
training and final sample grids: rendered and visually inspected
limitation: the run does not exercise a division that approaches zero
limitation: T014 remains open until the scientific safe-division policy is fixed
```


## 2026-08-20T19:28:26Z - Run eps/eps with exact division on Flowers102

### Decision
Add `notebooks/sanity/flowers102_runner_eps_eps_exact_division_sanity.ipynb`.
Preserve the executed `v/v` notebook and its run artifacts.

Repeat the 10,000-update Flowers102 protocol with `eps` prediction and `eps`
loss. Disable safe division for training and sampling. Keep the inactive
denominator epsilon at `0.01`. Keep the architecture, optimizer, data budget,
seed, checkpoint schedule, sampler, evaluation threshold, and sample count
fixed.

The `eps/eps` training loss does not require a target-space conversion. The
training safe-division flag is therefore inactive. Sampling converts `eps` to
velocity as `v_hat = (z_t - eps_hat) / t`. Exact sampling uses `t` directly.

Replay the same trained checkpoints in a named `safe_sampling` evaluation.
This changes only `sampling.use_safe_div` to `True`. Use the saved data
and fixed noise bank for both evaluations.

### Result
The full MPS run completed in 11.85 minutes. The first recorded `eps` loss was
`0.993115`. The final recorded loss was `0.007049`. Do not compare this loss
directly with the `v` loss because the targets have different scales.

All 64 final samples were finite, but the qualitative result failed. The saved
sample range was `[-14.723, 16.013]`, and only `31.56%` of scalar values were
inside `[-1, 1]`. The grid contained colored noise instead of flower images.

Final `f_mem` was zero at `rho = 0.3`, `0.5`, and `0.7`. Final mean `d1` was
`270.293`, and final mean `r` was `0.988833`. The controlled `v/v` run had
final mean `d1 = 39.953`, mean `r = 0.688950`, and canonical
`f_mem = 0.015625`.

The exact and safe sampling evaluations produced bitwise-identical tensors and
identical metrics. Their maximum absolute sample difference was zero. The
sampling grid starts at `t = 0.01`, and safe division clamps only denominators
smaller than `0.01`. The safe clamp therefore does not activate.

The exact data tensors, sample identifiers, `z_init`, and initial model state
matched the controlled `v/v` run. Under these controls, the observed failure is
associated with the `eps/eps` parameterization, not with the safe-division
flag. A plausible mechanism is the `1/t` amplification of prediction error.
The amplification reaches `100` at the first sampling time. This run does not
separate model error from the 50-step uniform solver discretization.

The complete run directory is
`outputs/sanity/flowers102_runner_eps_eps_exact_64_n8_10k_20260820T191129Z`.
It remains excluded from Git.

### Checks
```text
environment: memcompgen, macOS arm64, CPU preflight and MPS full run
two-update eps/eps exact-division preflight: passed
full notebook execution: 11/11 code cells executed, zero errors
top purpose comments: 11/11
runner return code: 0
runner state after default evaluation: evaluated
optimizer updates: 10000
checkpoint files: 5/5
checkpoint metric records: 5/5
final x_1 tensor: [64, 3, 64, 64], finite
data tensor equality with v/v control: exact
sample identifier equality with v/v control: exact
z_init equality with v/v control: exact
initial model state equality with v/v control: exact
safe versus exact final samples: bitwise equal
safe versus exact checkpoint metrics: equal
final sample grid: rendered and visually inspected
notebook schema and code-cell compilation: passed
git diff whitespace check: passed
limitation: one seed and eight training images provide integration evidence only
limitation: T014 remains open until the scientific safe-division policy is fixed
```


## 2026-08-20T19:35:48Z - Add diagonal-equivalent velocity loss weights

### Decision
Keep `framework.loss_weighting = "unit"` as the default. Add two optional
weightings for native `v/v` training:

```text
x_equivalent:   w(t) = (1 - t)^2
eps_equivalent: w(t) = t^2
```

For the linear FM path,
`z_t = t x + (1-t) eps` and `v = x - eps`. A velocity prediction implies
`x_hat = z_t + (1-t) v_hat` and `eps_hat = z_t - t v_hat`. Therefore,
the `x` prediction error is `(1-t)` times the velocity error. The `eps`
prediction error is `-t` times the velocity error.

Do not normalize these weights. Their raw scale makes the weighted `v/v` MSE
equal to the corresponding diagonal `x/x` or `eps/eps` MSE for the same
implied prediction.

Require both `framework.prediction_space = "v"` and
`framework.loss_space = "v"` for these options. Reject their use with another
prediction or loss space. This keeps the options scoped to the requested
native velocity control.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
analytical x-equivalent weight values: passed
analytical eps-equivalent weight values: passed
weighted v/v loss equals implied x/x loss: passed
weighted v/v loss equals implied eps/eps loss: passed
invalid non-v/v configurations: rejected
strict runner override parsing for x_equivalent: passed
two-update x_equivalent synthetic runner preflight: passed
two-update eps_equivalent synthetic runner preflight: passed
focused tests: 25 passed, 6 subtests passed
full pytest suite: 37 passed, 6 subtests passed
Ruff lint on mem_param_src and tests: passed
git diff whitespace check: passed
limitation: repository-wide Ruff formatting still reports older model.py formatting
```


## 2026-08-20T20:49:15Z - Diagnose eps/eps solver resolution

### Decision
Create `protoypinging_and_trubbleshoting/eps_eps_sampling` with separate
`scripts` and `notebooks` directories. Preserve the requested top-level name.

Add `replay_heun_resolution.py`. Replay the saved Flowers102 `eps/eps`
checkpoints with 200 and 500 uniform Heun steps. Keep the checkpoints, data
tensors, fixed `z_init`, device, time interval, exact division, `x_1` output,
sample count, and evaluation settings fixed. Change only `sampling.n_steps`.

Add `measure_eps_conditioning.py`. Pair the eight saved training images with
the saved noise prefix. Measure epsilon error and induced velocity error on
exact path states at nine fixed times. Check the implemented conversion against
the direct linear-path formula.

Add and execute `eps_eps_heun_resolution.ipynb`. Use linear code cells with no
function or class definitions. Start every code cell with a purpose comment.
Show the exact training tensor, matched 50-, 200-, and 500-step sample grids,
checkpoint metrics, sample differences, and endpoint-conditioning errors.

### Result
The 200- and 500-step evaluations completed on MPS for all five checkpoints.
All final tensors were finite. Increasing solver resolution did not recover
flower samples. The 50-, 200-, and 500-step grids all contained colored noise.

The final raw sample ranges were `[-14.723, 16.013]`, `[-14.627, 15.847]`, and
`[-14.618, 15.833]`. Their fractions inside `[-1, 1]` were `0.3156`, `0.3175`,
and `0.3177`. Final mean `d1` changed from `270.293` to `268.527` and
`268.379`. Every final `f_mem` value remained zero at rho `0.3`, `0.5`, and
`0.7`.

Matched 50- and 200-step samples had RMSE `0.029318`. Matched 200- and 500-step
samples had RMSE `0.002376`. Their convergence ratio was `0.081034`. The small
200-to-500 change and unchanged failure show that coarse Heun resolution is not
the main cause for this saved model.

On exact path states at `t = 0.01`, epsilon RMSE was `0.027713`. The induced
velocity RMSE was `2.771305` because the conversion amplified error by `100`.
The implemented conversion matched `(z_t - eps_hat) / t` within `2.25e-5`.
The relative velocity-MSE identity error stayed below `1.3e-7`. Conversion
arithmetic is therefore not the observed failure source.

The evidence implicates the learned epsilon field and its endpoint
conditioning. This test does not separate the diagonal training objective,
the pure-noise start approximation, and off-path error accumulation. The next
controlled model comparison is native `v/v` with `eps_equivalent` weighting.
T014 remains open because this test did not compare active clamp thresholds.

### Checks
```text
environment: memcompgen, macOS arm64, MPS
200-step checkpoint replays: 5/5
500-step checkpoint replays: 5/5
fixed z_init samples per checkpoint: 64
final sample tensors: finite, shape [64, 3, 64, 64]
endpoint probes: 9 times, 8 fixed train-noise pairs
executed notebook: 12/12 code cells, zero errors
notebook function and class definitions: 0
notebook top purpose comments: 12/12
notebook schema validation: passed
notebook artifact-reuse execution: passed
sample grids and diagnostic plots: rendered and visually inspected
standalone script Ruff lint and formatting: passed
full pytest suite: 37 passed, 6 subtests passed
git diff whitespace check: passed
limitation: the endpoint probe uses exact path states, not generated trajectory states
limitation: one seed and eight training images provide diagnostic evidence only
```


## 2026-08-20T20:59:13Z - Catalogue the unresolved eps/eps cross-protocol contradiction

### Decision
Do not generalize the failed Flowers102 sanity result into a claim that
`eps/eps` cannot generate images. Record the result as an unresolved
cross-protocol contradiction.

Add
`protoypinging_and_trubbleshoting/eps_eps_sampling/OPEN_PROBLEM.md` as the
durable problem catalogue. Add T016 as an open research task. Keep T014 open
because the sampling division policy remains unsettled.

### Historical evidence
The old CelebA32 evaluation table contains 21 completed `eps/eps` runs across
seven training-set sizes and three seeds. Each run has 72 checkpoint rows.
This gives 1,512 checkpoint evaluations with Heun-50 sampling and 10,000
generated samples per checkpoint.

The best FID across each run ranges from `6.0959` through `19.8533`. Its median
across all 21 runs is `11.8436`. At `N_train = 16384`, the three best FID values
range from `6.0959` through `7.3663`. These saved values provide direct evidence
that the old `eps/eps` protocol produced useful CelebA32 samples.

The old project documentation also reports that `eps/eps` memorizes later and
less than `x/x`. That ordering remains visible at the best-FID checkpoints.
The user separately reports visually good CelebA32 grayscale samples.

The current old checkout does not contain the checkpoint tree referenced by
the evaluation CSV. Therefore, this catalogue pass did not replay or inspect
the historical sample tensors.

### Current Flowers102 evidence
The clean run uses eight fixed RGB Flowers102 images at `64 x 64`, a
669,251-parameter UNet, Adam at `1e-3`, and 10,000 updates. The diagonal epsilon
loss decreased from `0.993115` to `0.007049`, but the final samples remained
colored noise.

Replays with 50, 200, and 500 Heun steps converged to the same failed result.
The 200-to-500 sample RMSE is `0.002376`. At `t = 0.01`, epsilon RMSE is
`0.027713`, while induced velocity RMSE is `2.771305`. The tested conversion
matches the direct linear-path formula.

These checks exclude coarse Heun resolution and the checked conversion
arithmetic as primary explanations for this saved model. They do not identify
the cause.

### Important protocol difference
The historical sampler starts at `t = 0.001` and always applies safe division
with epsilon `0.05`. The clean failed run starts at `t = 0.01` and uses exact
division. The earlier clean safe replay used epsilon `0.01`, so its clamp did
not activate.

The old early conversion limits epsilon sensitivity to about `21`. The clean
exact conversion reaches sensitivity `100` at its first time. This difference
is a credible cause, but it is not established.

Other uncontrolled differences include dataset, image dimension, model
capacity, attention, dropout, learning rate, optimizer updates, batch size,
output convention, precision, device, and evaluation protocol.

### Next tests
1. Replay the fixed Flowers102 checkpoints with a `0.05` sampling clamp.
2. Replay the same checkpoints with the complete historical sampling policy.
3. Recover one historical CelebA32 checkpoint and replay it through both codes.
4. Reproduce one old CelebA32 `eps/eps` run through the clean runner.
5. Build a matched bridge that changes one protocol factor at a time.
6. Compare exact-path errors with generated-trajectory errors.
7. Run the `v/v` plus `eps_equivalent` representation control afterward.

### Checks
```text
old evaluation CSV: 1,512 eps/eps rows across 21 runs
old run grid: seven N_train values times three seeds
old checkpoint count: 72 rows per run
historical sampler: Heun, 50 steps, [0.001, 0.999], safe epsilon 0.05
historical output: integrated state at t_max
clean failed sampler: Heun, 50 steps, [0.01, 0.99], exact division, x_1 output
historical checkpoints in current old checkout: not found
status: open and unresolved
```


## 2026-08-20T21:04:00Z - Close the session and set the next priority

### Decision
End the 2026-08-20 research session without starting another experiment.
Make T016 the first task for 2026-08-21.

Start by replaying the saved Flowers102 checkpoints with the historical `0.05`
sampling clamp. Keep the checkpoints, `z_init`, sampling interval, output, and
evaluation settings fixed. This test changes only the active clamp threshold.

After that control, test the complete historical sampling policy in a separate
named evaluation. Do not start retraining before these replay controls finish.

Treat input dimensionality as the second explicit question. The old CelebA32
state has 1,024 values, while the failed Flowers102 state has 12,288 values.
After the replay controls, compare the same Flowers image identities at RGB64,
RGB32, and grayscale32. Keep the resolved training and sampling protocol fixed.

Report per-coordinate errors and L2 distances normalized by `sqrt(D)`. Raw
pixel distances are not directly comparable across these tensor shapes.

### Session state
```text
first next task: T016
problem status: open and unresolved
first test: fixed Flowers102 replay with safe_div_denominator_floor = 0.05
second test: matched Flowers RGB64, RGB32, and grayscale32 dimensionality screen
new experiment started during closeout: no
repository commit created: no
repository push performed: no
```


## 2026-09-01T10:44:34Z - Add image training loss and sample monitoring

### Decision
Add W&B monitoring only to the image training runner. Keep W&B and generation
monitoring disabled by default. Do not add a tracker class, callback system,
resume behavior, synthetic tracking, or train-time scientific evaluation.

Derive the W&B project from dataset and model. Derive the run name from
dataset, model, training-set size, batch cap, prediction and loss spaces,
training steps, and seed. Derive the comparison group from dataset, model, and
training steps.

Log loss, learning rate, elapsed time, and progress when training history is
recorded. Generate qualitative image grids at step zero and saved checkpoints.
Reuse the same saved `z_init` prefix at every step and display every grid with
the fixed `[-1, 1]` range. Do not calculate FID, `f_mem`, nearest-neighbor
statistics, or validation losses in the training loop.

### Result
The image command now supports `--wandb`, `--no-wandb`, `--wandb-mode`,
`--generation-monitor`, and `--no-generation-monitor`. W&B state is finished
in the training loop's `finally` block. Non-strict W&B failures disable logging
and let model updates continue.

The generation monitor temporarily uses evaluation mode, samples without
gradients, restores the previous model mode, rejects non-finite samples, and
saves one PNG per monitoring step inside the run directory. The Flowers102
smoke configuration uses all four tensors in its saved noise bank.

The final monitored two-update Flowers102 run logged loss at steps one and two.
It saved and uploaded sample grids at steps zero, one, and two with no duplicate
final grid. A second two-update run logged loss with generation monitoring
disabled and created no monitoring directory.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
W&B version: 0.27.0
focused runner tests: 17 passed
full pytest suite: 43 passed, 6 subtests passed
Ruff lint on mem_param_src, tests, and smoke config: passed
Ruff formatting on changed Python files: passed
git diff whitespace check: passed
monitored offline run: /private/tmp/mem_param_wandb_monitor_verified_20260901/run
monitored W&B run id: dwjx5boy
monitored sample grids: steps 0, 1, and 2 only
loss-only offline run: /private/tmp/mem_param_wandb_loss_only_20260901/run
loss-only monitoring directory: absent
sample grids: rendered and visually inspected
limitation: W&B online authentication was not tested because api_key is unset
repository commit created: no
repository push performed: no
```


## 2026-09-01T11:28:27Z - Download and verify CelebA locally

### Decision
Use Torchvision's official CelebA downloader. Store the dataset under the
git-ignored root `data/celeba`. Verify the first-party grayscale loader with
`download=False` after extraction.

### Result
Torchvision downloaded and extracted 202,599 aligned images. The complete
local directory uses 3.1 GB. The clean loader produced four training and four
validation tensors with shape `[1, 32, 32]`. All checked values were finite and
inside `[-1, 1]`.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
Torchvision CelebA total samples: 202,599
extracted aligned image files: 202,599
train source size: 162,770
validation source size: 19,867
train tensor: [4, 1, 32, 32], finite
validation tensor: [4, 1, 32, 32], finite
loader mode: download=False
local size: 3.1 GB
git ignore boundary: /data/ confirmed
repository commit created: no
repository push performed: no
```


## 2026-09-01T11:47:21Z - Validate configurable CelebA image shapes

### Decision
Preserve the historical 140-pixel aligned-face crop. Make the subsequent
resize, final square crop, and channel conversion follow `data.resize_size`,
`data.image_size`, and `data.image_channels`.

Use the same four seeded CelebA identities for every visual comparison. Check
grayscale and RGB at 16, 32, 64, and 512 pixels. Keep normalization fixed at
`[-1, 1]`.

### Result
The complete local loader produced all eight requested tensor configurations.
Every tensor had the requested shape, contained only finite values, and stayed
inside `[-1, 1]`. The matched panels show the expected loss of detail at small
sizes and preserved color in RGB mode.

The 512-pixel variants are upsampled from the 140-pixel crop. They verify tensor
shape and channel handling but do not contain additional source information.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
real CelebA loader variants: 8/8 passed with download=False
resolutions: 16, 32, 64, 512
channels: grayscale 1, RGB 3
matched seeded identities across variants: confirmed
fixed normalized range: [-1, 1]
focused transform tests: 2 passed, 8 subtests passed
full pytest suite: 45 passed, 14 subtests passed
Ruff lint on mem_param_src and tests: passed
Ruff formatting on changed Python files: passed
git diff whitespace check: passed
visual panels: rendered and inspected
repository commit created: no
repository push performed: no
```


## 2026-09-01T12:18:12Z - Validate the first-party image dataset loaders

### Decision
Add CIFAR-10 to the first-party data bundle dispatcher. Permit MNIST to emit
one grayscale channel or three repeated grayscale channels. Use one executed
notebook to test the public loaders for MNIST, CIFAR-10, Flowers102, and CelebA.

Compare fixed source identities across image sizes and channel modes. Check
nested fixed training subsets, different seeded memberships, exact repeated
loads, finite values, and the fixed `[-1, 1]` range. Keep `download=False` so
the notebook proves local data readiness.

### Result
The executed notebook passed all available loader checks. It validated 32
size-channel configurations, 12 fixed training-set configurations, four seed
comparisons, and four exact repeat comparisons. Eight embedded figures make
the transformations and subset changes directly visible.

AFHQ-v2 and FFHQ128 remain `NOT TESTED` because their local data is absent.
The 512-pixel checks verify output shape and deterministic preprocessing. They
do not imply that low-resolution sources contain additional image detail.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
notebook: notebooks/sanity/image_dataset_loader_validation.ipynb
notebook execution: 15/15 code cells executed, 0 error outputs
embedded figures: 8 rendered and visually inspected
required local datasets: 4/4 passed with download=False
size-channel configurations: 32/32 passed
resolutions: 16, 32, 64, 512
channels: grayscale 1, RGB 3
training-set sizes: 4, 8, 16; 12/12 passed
seed comparisons: seeds 7 and 19; 4/4 different memberships
deterministic repeats: 4/4 exact tensors and source indices
notebook structure and language checks: passed
full pytest suite: 47 passed, 14 subtests passed
Ruff lint on mem_param_src and tests: passed
Ruff formatting on changed Python files: passed
git diff whitespace check: passed
repository commit created: no
repository push performed: no
```


## 2026-09-01T12:47:17Z - Centralize deterministic image bundle construction

### Decision
Keep `mem_param_src.datasets.build_data_bundle(config)` as the public entrypoint.
Move dataset dispatch out of `common.py`. Put deterministic image selection,
exact class balancing, subset materialization, bundle assembly, and provenance
fields in one shared image pipeline.

Keep only source-specific behavior in the dataset adapters. Preserve MNIST and
CIFAR tensor interpolation, the CelebA 140-pixel crop, Flowers split handling,
and folder-specific AFHQ/FFHQ identities. Apply the existing
`balance_train_classes` and `balance_reference_classes` settings to both MNIST
and CIFAR-10. Require exact balanced counts to be divisible by ten.

### Result
The available real datasets produce the same tensors, indices, shapes, and
source identifiers as before the refactor. The six core real-image files now
contain 985 lines, compared with 1,394 before this refactor. Dataset adapters
open their source and then call the shared image pipeline.

Real MNIST and CIFAR-10 balance checks selected two examples from every class
for both training and reference subsets at `N=20`. The visual dataset notebook
passed every previous assertion through the new path.

AFHQ-v2 and FFHQ128 remain runtime-unverified because their local data is
absent. Their source-specific adapters remain pending the dataset-retention
decision in T007.

### Checks
```text
environment: memcompgen, macOS arm64, CPU
pre/post real tensor hashes: exact match for MNIST, CIFAR-10, Flowers102, CelebA
pre/post source indices and identifiers: exact match
real balanced MNIST: 2 examples per class in train and reference
real balanced CIFAR-10: 2 examples per class in train and reference
focused loader and runner tests: 24 passed, 8 subtests passed
full pytest suite: 50 passed, 14 subtests passed
notebook execution: 15/15 code cells executed, 0 error outputs
notebook loader assertions: all passed
Ruff lint on mem_param_src and tests: passed
Ruff formatting on changed Python files: passed
git diff whitespace check: passed
repository commit created: no
repository push performed: no
```


## 2026-09-07 - Start a paper wording and ideas file

### Decision
Add root-level `PAPER_NOTES.txt` for draft prose, research framing, source
references, and assumptions that need evidence. Link it from `README.md`.
Keep operational tasks in `TODO.txt`.

The first entry relates the proposed memorization study to Li and He's
`Back to Basics: Let Denoising Generative Models Denoise`. Frame the question
around regimes where the compared parameterizations produce useful samples.
Mark the latent-autoencoder setting as proposed, not an established result.

### Checks
- Read the original arXiv v1 source online; record its URL and relevant sections.
- Read back the notes and check documentation whitespace.
- No runtime tests: this change only adds research notes and a discovery link.


## 2026-09-07 - Simplify paper wording and record the writing rule

- Shortened the first PAPER_NOTES.txt entry and put the draft paragraph first.
- Added a plain-English writing rule to PAPER_NOTES.txt and AGENTS.md for future
  paper text, documentation, comments, and discussion.
- Kept the source, proposed experiment, and evidence limits.
- Read back the edited text and checked whitespace. No code changed.


## 2026-09-07 - Record how implementation should support user understanding

- Added a collaboration rule to AGENTS.md: explain the relevant file layout,
  data flow, design choice, and a short usage example for substantial changes.
- Require real API examples or clearly labelled pseudocode, with execution
  status and links to the implementation and saved results.
- Use docs/repo_map.md for the layout and existing documentation for reusable
  examples. Keep explanations brief and involve the user in meaningful choices.
- Checked the documentation diff for whitespace. No runtime code changed.


## 2026-09-07 - Prepare the online Flowers102 monitoring demonstration

- Added configs/runners/flowers102_wandb_demo.py and a README launch command.
- Selected v/v, eight RGB32 images, 1,000 updates, and 16 fixed-noise previews.
- Local memcompgen config import passed. Checkpoints resolve to 0/250/500/750/1000.
- Initial launch stopped before training: CLI booleans require lowercase true;
  the corrected command then found MPS unavailable inside the sandbox.
- Automatic approval review rejected the host-access online launch because the
  destination and upload contents were not explicit. No online run started.
- Prepared an explicit maj90/flowers102-unet destination and a command that
  disables W&B code and Git capture. Await user confirmation of the destination
  and upload contents before retrying.


## 2026-09-07 - Change the proposed monitoring run to MNIST

- Replaced the unrun Flowers demo config with configs/runners/mnist_wandb_demo.py.
- Selected eight grayscale MNIST images at 32 x 32, 1,000 updates, and a two-level
  UNet with 32 base channels, one residual block per level, and no attention.
- Kept v/v, seed 7, online W&B, and 16 previews every 250 updates.
- Local memcompgen loader check passed with download=False: train [8,1,32,32],
  reference [16,1,32,32]. Checkpoints resolve to 0/250/500/750/1000.
- Updated the README command. Training remains unstarted pending confirmation
  of the explicit W&B destination and upload contents after the review rejection.


## 2026-09-07 - Complete the online MNIST monitoring run

- User approved uploads to maj90/mnist-unet. Launched the README command with
  WANDB_DISABLE_CODE=true and WANDB_DISABLE_GIT=true using host GPU access.
- Environment: local memcompgen, MPS, W&B 0.27.0. Runner exited with code 0.
- Config: configs/runners/mnist_wandb_demo.py. Eight grayscale 32 x 32 images,
  1,000 updates, v/v, seed 7, 668,097-parameter UNet, and 50-step Heun sampling.
- Run: outputs/mnist_wandb_live_20260907_01.
- W&B: https://wandb.ai/maj90/mnist-unet/runs/xg3q5n39.
- W&B reported successful sync of five media files and training history.
  run.json records success=true, no warning, and final state evaluated.
- First logged loss: 1.891039. Final logged loss: 0.149338. These are batch losses.
- Training log elapsed time: 35.30 seconds; this excludes some setup and evaluation.
- Verified five grids and metric records at steps 0/250/500/750/1000.
- Final sample tensor: [16,1,32,32], all finite. Final pixel f_mem=0 at rho=0.3.
- Viewed the final PNG: digit-like samples are visible. This is a small workflow
  demonstration; it does not establish model quality or a memorization ordering.
- Post-training metrics are saved locally; the training W&B run logs loss and
  previews, not the later checkpoint evaluation metrics.


## 2026-09-07 - Disable safe division in the MNIST demo config

- Set training.use_safe_div and sampling.use_safe_div to False in
  configs/runners/mnist_wandb_demo.py. Time limits remain [0.01,0.99].
- Local memcompgen config-load assertions passed for both flags.
- Completed run artifacts retain their original settings. No new run launched.


## 2026-09-07 - Select eps/eps for the next MNIST demo

- Set framework.prediction_space and framework.loss_space to eps in
  configs/runners/mnist_wandb_demo.py. Both safe-division flags remain disabled.
- Local memcompgen config-load assertions passed. No new run launched.


## 2026-09-07 - Run MNIST eps/eps with exact division and online monitoring

- Ran configs/runners/mnist_wandb_demo.py in local memcompgen on MPS using
  WANDB_DISABLE_CODE=true and WANDB_DISABLE_GIT=true. Runner exit code: 0.
- Run directory: outputs/mnist_wandb_eps_eps_exact_20260907_01.
- W&B: https://wandb.ai/maj90/mnist-unet/runs/xfc8d7un.
- Eight grayscale MNIST images at 32 x 32, 1,000 updates, small UNet, eps/eps,
  seed 7. Both safe-division flags disabled. Heun-50, [0.01,0.99], x_1 output.
- W&B reported successful sync of five media files and training history.
- First batch loss: 0.995535. Final batch loss: 0.009982. Logged training elapsed
  time: 41.14 seconds, excluding some setup and post-training evaluation.
- Verified run state evaluated, W&B success=true, and five checkpoint records.
- Final samples: [16,1,32,32], all finite; raw range [-12.5795,15.4053].
- Inspected the final grid: a few digit-like samples, many saturated or noisy.
  Low epsilon loss did not imply useful generation across the sample bank.
- Final pixel f_mem=0 at rho=0.3. Do not interpret this as better generalization
  because generation quality is poor. One seed and 16 samples are diagnostic only.
- The run's saved data-bundle and manifest hashes match the earlier v/v demo.
  The earlier run enabled safe division; this run disabled it. Preserve this
  distinction when describing the comparison.


## 2026-09-07 - Increase MNIST UNet capacity and plan the attention comparison

- Updated configs/runners/mnist_wandb_demo.py to 64 base channels, multipliers
  (1,2,4), two residual blocks per level, and 10,000 updates. Kept eps/eps,
  exact division, eight images, seed, optimizer, and preview schedule.
- Added T017 to TODO.txt: compare eps/eps with and without attention under
  matched settings. True attention-off needs control of the middle block too.
- Corrected the previous no-attention description: attn_resolutions=() skips
  optional encoder attention but the middle SelfAttention always executes.
  This was also true for the earlier small-UNet runs.
- Local memcompgen CPU check: model construction and loss/backward passed on
  [2,1,32,32] inputs, with finite loss and gradients. Parameters: 14,088,449.
  Three attention modules exist; two encoder modules are bypassed with ().
- Updated README with current settings and separate historical run links.
- No training launched. A larger/longer run alone cannot isolate capacity from
  training-budget effects; the attention comparison remains a separate task.


## 2026-09-07 - Verify the official Bonnaire network

- Read Bonnaire et al., arXiv:2505.17638v2, and the official repository at
  de6d4c3abecdce49b6e8ae3dc7898d098228457b. Recorded pinned source links and
  the comparison in docs/config_reference.md.
- The main text says multipliers (1,2,3); Appendix A and released training code
  use (1,2,4). Official attention is four-head at 16/8 pixels in both encoder
  and decoder, plus the middle block. Dropout is 0.1; GroupNorm uses 8 groups.
- Read the downloaded Unet.py before importing it from /tmp. Local memcompgen
  CPU checks instantiated widths 16/32/64 and passed finite [1,1,32,32] forwards.
  Trainable counts: 1,010,657 / 4,023,233 / 16,054,145; each has 11 attention blocks.
- The local BonnaireUNet differs in attention placement/heads, normalization,
  time embedding/conditioning, and initialization. It is not an exact replica.
- No network implementation or run config changed. No training launched.


## 2026-09-07 - Select channel multipliers (1,2,3) for the MNIST demo

- Changed configs/runners/mnist_wandb_demo.py as requested. With base width 64,
  the three channel widths are 64/128/192. Trainable parameters: 9,393,985.
- Kept num_res_blocks=2: two blocks per encoder level, three per decoder level,
  and two middle blocks. Updated README to match the current configuration.
- Local memcompgen CPU model construction and loss/backward passed with finite
  loss and gradients on [2,1,32,32] input. No training launched.


## 2026-09-07 - Launch the larger 10k MNIST eps/eps run

- Launched configs/runners/mnist_wandb_demo.py on MPS in local memcompgen with
  online W&B and code/Git capture disabled. User requested this run.
- Run: outputs/mnist_wandb_eps_eps_123_10k_20260907_01.
- W&B: https://wandb.ai/maj90/mnist-unet/runs/en9xtd1x.
- Config: eight grayscale 32 x 32 images, channels 64/128/192, 9,393,985
  parameters, eps/eps, exact division, seed 7, 10,000 updates. Save previews and
  checkpoints every 250 updates; run evaluation after training.
- Startup verified: online logging active, initial grid saved, finite loss
  through the first 250 updates. State at this entry: training, not complete.


## 2026-09-07 - Add optional projected f_mem during image generation monitoring

- Reviewed ../mem_par/src/mp_training/trainer.py preparation/run functions and
  ../mem_par/src/mp_metrics/metrics_images.py. The old metric is already present
  in metrics/image/projection.py and uses shared nearest_neighbors.py math.
- Direct CPU comparison used identical seeded tensors, exact training copies,
  and out-of-range queries. Old/clean projection matrices and complete metric
  records matched exactly (f_mem=0.25 at rho=0.3). No duplicate metric port added.
- Added --f-mem-monitor / --no-f-mem-monitor and --f-mem-samples. Default is off;
  enabling requires generation monitoring. Sample count alone does not enable it.
- The grid and metric share one sample batch at step zero and checkpoints. Use
  the larger sample count, keep the grid prefix small, and expand the saved noise
  bank before saving the resolved config. Projection seed defaults to seed+3001.
- Cache/save the projected actual training tensors and projection in monitoring/.
  Write f_mem_step_<step>.json at each checkpoint and log canonical fields under
  generation/random_projected_pixels/ in W&B. W&B remains optional.
- Approximation follows old preprocessing: clamp normalized pixels to [0,1],
  resize to 16x16, fixed 256-dimensional Gaussian projection, and d1/d2 ratio.
  Use the clean configured rho=0.3. Interpret alongside generation quality;
  this is distinct from full-resolution post-hoc metrics and FID.
- Local memcompgen: 53 tests and 14 subtests passed. Ruff lint and whitespace
  checks passed. Tests cover CLI settings, gating, fixed-noise sample reuse,
  larger metric count, grid prefix, exact training matches, artifacts, and logs.
- Real W&B 0.27.0 offline integration: /private/tmp/mem_param_f_mem_monitor_20260907_02.
  Two CPU MNIST updates; three grids/records at 0/1/2; n_samples=8, N=4, rho=0.3.
  W&B history contains f_mem/mean_r/rho/n_samples/N; finish success=true.
- Initial offline test was blocked by sandbox service sockets; host-access retry
  passed. No test data was uploaded. No new research training run launched.
- Existing 10k process/config is unchanged; the new option applies to new runs.


## 2026-09-08 - Session closeout and next-session handoff

Completed this session:
- Added PAPER_NOTES.txt for reusable paper prose. Recorded the Li and He
  comparison: study memorization where both parameterizations can generate
  useful samples. This is a research question, not an established result.
- Saved the standing plain-English and code-understanding rule in project
  notes and the requested persistent memory notes. Explain file roles, data
  flow, and a short usage example when making substantial changes.
- Completed three online monitored MNIST runs: small v/v (xg3q5n39), small
  exact eps/eps (xfc8d7un), and larger exact eps/eps (en9xtd1x). All use eight
  training images at 32x32 and saved noise for checkpoint previews.
- Audited the official Bonnaire network and recorded source links and local
  differences in docs/config_reference.md. Our UNet is not an exact replica.
  The selected larger config uses widths 64/128/192, two encoder residual
  blocks per level, and 9,393,985 parameters. Middle attention remains active.
- Added optional projected f_mem during image generation monitoring. Reused
  the existing clean metric after comparison with the old source. Use
  --generation-monitor --f-mem-monitor --f-mem-samples 512 on a new run.
  README.md contains the command; docs/config_reference.md explains the
  approximation; docs/repo_map.md shows the file roles. No FID was added.

Final larger-run result, checked at closeout:
- Local directory: outputs/mnist_wandb_eps_eps_123_10k_20260907_01.
- W&B: https://wandb.ai/maj90/mnist-unet/runs/en9xtd1x.
- run.json reports evaluated, final_step=10000, W&B success=true and inactive.
  There are 41 checkpoints and 41 fixed-noise grids, including step zero.
- Final batch loss: 0.0094292928. Final full-pixel f_mem=0 at rho=0.3;
  mean_r=0.9801455. The saved 16x1x32x32 samples are finite, with raw range
  [-14.439889, 28.545033]. Visual inspection shows mostly noise or saturation,
  with a few digit-like samples. Low loss did not establish useful generation.
  Low f_mem here cannot support a better-generalization claim.
- This run changed capacity and training duration together. It does not
  isolate either factor. It predates the new train-time f_mem option, which
  was off. No attention ablation or matched multi-seed comparison was run.

Next session:
- Begin with the larger run's saved grids and evaluation records. Select one
  matched comparison before launching more training. T017 holds the requested
  attention experiment; a true middle-attention-off option is still needed.
- T016 remains the separate unresolved Flowers102/CelebA32 contradiction.
  Historical clamp replay and dimension controls remain undone.
- Updated TODO.txt to replace its stale dated starting instruction and record
  completed loader/monitoring work. Corrected README's pending-run statement.

Validation and publication scope:
- Closeout check in local memcompgen: python -m pytest -q passed 53 tests and
  14 subtests. Ruff passed for mem_param_src, tests, and configs. Whitespace
  checks passed. These complement the runtime and offline integration checks
  recorded above; they are not a new online f_mem research run.
- The requested commit includes the accumulated clean-repository changes since
  the prior commit, including earlier dataset validation, objective-weight
  controls, and executable diagnostic notebooks. Earlier journal entries
  retain their provenance; they are not all work performed today.
- Publish the current codex/reproducible-runners branch to origin as requested.
  Raw data, training outputs, and checkpoints remain local and Git-ignored.
  Notebook figures remain included as executable research evidence.

## 2026-09-08 - Add next-session note for one visible results folder

- The user cannot easily find the run outputs and requested a next-session
  reminder to collect results under one clearly named temporary folder.
- Added this as the first action in TODO.txt. Suggested outputs/session_runs/,
  with one subfolder per new run and a short index linking settings, W&B,
  final grids, metrics, and existing MNIST runs. Report its absolute path.
- This entry records the request only. No artifacts were moved and no output
  folder was created. The final location remains a next-session choice.


## 2026-09-08 - Add training comparison grids and local monitoring curves

- User requested one image-runner command for training, generation grids, loss,
  and f_mem curves. Clarified N=32 training images, 5,000 updates, v/v, and 500
  generated samples per monitoring checkpoint, with eight images per grid row.
- Added generation_monitor.n_train_samples and --training-grid-samples. Zero
  keeps generated-only grids. Positive counts show a fixed prefix of the exact
  saved training tensor above generated samples in monitoring/comparison_grids/.
- Added runners/image_plotting.py. It draws comparison grids and reads saved
  train_history.json / f_mem_step_*.json records for monitoring/plots/loss.png
  and f_mem.png. Curves refresh at preview checkpoints; loss records now persist
  at checkpoints. The existing projected metric and shared generation are reused.
- Declared Matplotlib in the image dependency profile. Updated the config
  reference, repository map, and README command and saved-record plotting example.
- Added configs/runners/mnist_vv_monitor.py. Retained the current 9,393,985-
  parameter UNet, widths 64/128/192, two encoder blocks per level, middle attention,
  seed 7, Adam at 1e-3, exact division, and Heun-50 on [0.01,0.99]. Batch size is 32.
  Preview every 250 updates; show 32 training and 64 generated images; measure
  projected f_mem on all 500 generated samples at rho=0.3. W&B is disabled.
- Created outputs/session_runs/INDEX.md with links to this run and all three
  historical MNIST runs, their settings, grids, metrics, and historical W&B links.
  No historical artifacts were moved.
- Validation in local memcompgen: 22 targeted runner/config tests passed,
  including real CPU training with mocked source data/generation, fixed training
  prefix checks, shared metric samples, checkpoint artifacts, and live curves.
  Ruff and whitespace checks passed. Real local MNIST loading returned train
  [32,1,32,32] and held-out [500,1,32,32]; config/model construction passed.
- Started the requested MPS run in outputs/session_runs/mnist_vv_n32_5k_20260908_01
  through the image runner alone. Host access was needed because MPS is unavailable
  in the sandbox. The step-zero 500-sample generation and projected metric passed.
  Viewed its saved grid and verified eight columns, training above generation.
  State at this entry: training, not complete. No post-hoc evaluation was requested.


## 2026-09-08 - Stop the 32x32 run and prepare a user-launched MNIST16 command

- User stopped the 32x32 5k run, selected 16x16 MNIST and 1,000 updates, then
  requested the CLI command to launch it personally. No MNIST16 training started.
- Sent Ctrl-C to the running process and verified exit. The last observed loss
  log was step 1,200; the last saved checkpoint, grid, and metric are step 1,000.
  Marked its run.json as aborted with the user-requested reason. Preserved outputs.
- The step-1,000 projected f_mem was 0.008 (4 of 500 samples) at rho=0.3.
  This is a partial run; it did not complete the planned 5,000 updates.
- Updated configs/runners/mnist_vv_monitor.py and README for MNIST16, N=32,
  1,000 updates, 500 generated samples, 32 training / 64 generated grid images,
  and eight columns. Kept the UNet, seed, batch size, optimizer, and sampler.
- Real local memcompgen preflight passed: train [32,1,16,16], held-out
  [500,1,16,16], 9,393,985 parameters, checkpoints 0/250/500/750/1000.
- Added terminal generation announcements and projected f_mem details (mean_r,
  rho, n_samples, N), plus saved plot paths. Training already prints batch loss
  at training.log_every. The command uses --no-capture-output for live stdout.
- Saved 32x32 data/manifest hashes and fixed noise shape [500,1,32,32] were
  verified. The README saved-record plotting API ran and produced both curves.
- Final validation in memcompgen: python -m pytest -q tests/test_runners.py
  tests/test_config_reference.py passed all 22 tests. Ruff on changed Python
  files and git diff --check passed. The MNIST16 research run remains unstarted.


## 2026-09-08 - Add minimal descriptions of image-runner stages

- User requested short terminal messages that explain the current operation.
- Added image setup/data/model summaries and training start/continue/finish
  messages. Kept the existing loss interval and numerical metric details.
- Generation states the checkpoint, sample count, and sampler. Metric computation
  announces when samples are ready. One saved-folder line replaces repeated PNG
  paths. The image CLI reports optional post-training evaluation and completion.
- Used the existing console stage settings for setup/training descriptions.
  No new flags, training changes, or research runs were added.
- Updated the README to use conda activate followed directly by Python, as the
  user requested for their interactive terminal.
- Local memcompgen: one tiny CPU monitoring integration with fixture data and
  mocked generation/W&B, plus two CLI tests, passed (3 tests in 2.13 seconds).
  Inspected the emitted stage order. Ruff and whitespace checks passed.


## 2026-09-08 - Retain full generated tensors at image checkpoints

- User requested a flag to save checkpoint generation/evaluation samples.
  Monitoring had saved PNG grids and metric records but discarded the full
  generated batch. Separate evaluation had saved only its final batch.
- Added --save-generated-samples / --no-save-generated-samples to image_train,
  backed by evaluation.save_generated_samples. Base default is False; enabled
  in configs/runners/mnist_vv_monitor.py for future user-launched runs.
- Monitoring saves the existing full raw CPU batch before any display or metric
  preprocessing to monitoring/generated_samples/step_<step>.pt. It does not
  sample again. MNIST16 saves [500,1,16,16], while its grid displays a 64-sample
  prefix. Tensor order matches the saved z_init prefix.
- Optional post-training image evaluation saves each batch under
  evaluations/<name>/generated_samples/ and retains its original final-sample
  file. Older saved configs default the missing setting to False and accept
  --set evaluation.save_generated_samples=true for a subsequent named replay.
- Updated README, the config reference, and repository map with paths and a
  torch.load example. README uses a fresh _02 run directory for the next launch.
- Read the user's existing MNIST16 _01 run.json: state=trained, final_step=1000.
  Its five grids and five projected f_mem records exist. This run predates sample
  storage; no full batches were recovered, and no research run was launched.
- Local memcompgen: python -m pytest -q tests/test_runners.py
  tests/test_config_reference.py passed 22 tests in 2.64 seconds. Checks include
  exact full-batch tensor round trips with mocked monitoring samples, storage
  off by default, real CPU checkpoint replay, and legacy config compatibility.
  Ruff on changed Python files and git diff --check passed.


## 2026-09-08 - Prepare CelebA32 eps/eps and direct CLI controls

- User requested a command for CelebA32, N=32 faces, eps/eps, and convenient
  CLI controls for safe division and Heun steps. Prepared the command only;
  no research training or checkpoint evaluation was launched.
- Added configs/runners/celeba32_eps_eps_monitor.py. Selected grayscale32 to
  match the historical representation, with the preserved aligned 140-pixel
  crop, train/valid source splits, 32 training and 500 held-out images.
  Retained the current 9,393,985-parameter UNet and seed 7; use unit-MSE eps/eps,
  1,000 updates, Adam at 1e-3, batch size 32, and exact Heun-50 sampling.
- Kept monitoring every 250 updates, including step zero: 500 generated
  samples, 32 training / 64 generated grid images, projected f_mem, local curves,
  and full raw sample tensors. W&B remains disabled.
- Added image-training CLI flags for steps, lr, seed, batch size, checkpoint
  interval, sampler, sampling steps, sampling safe division and its denominator
  floor, sampling endpoints, and sampling output. Explicit flags override --set.
  An explicit checkpoint interval replaces the configured named/explicit schedule.
- Sampling flags leave training-time division and the objective unchanged.
  The monitor announces the actual solver, steps, interval, and division policy.
- README documents the command, grayscale/RGB choice, and a fixed-checkpoint
  clamp replay using image_eval. Config reference maps flags to saved fields.
  T016 remains open; this setup does not establish a cause or reproduce the old study.
- Local memcompgen validation: real CelebA loader returned train [32,1,32,32]
  and reference [500,1,32,32]. Model construction and a two-image loss/backward
  check passed with finite gradients. This was a preflight, not a training run.
- python -m pytest -q tests/test_runners.py tests/test_config_reference.py
  tests/test_celeba_loader.py passed 25 tests and 8 subtests in 2.71 seconds.
  CLI tests check sampler settings, precedence, schedule resolution, and unchanged
  training division. Ruff and whitespace checks passed.


## 2026-09-09 - Preserve the critical eps/eps audit for Berzelius

- Completed the requested read-only audit of the current CelebA32 execution
  path and selected April N=256 seed-0 eps/eps study. The current model already
  has model.sample; external Euler/Heun reads model.prediction_space and calls
  model.convert. No lost parameterization or incorrect Heun stage was found.
- Current run outputs/session_runs/celeba32_eps_eps_n32_exact_01 completed
  1,000 updates on MPS. Saved final samples span [-10.23212,11.74243]; 59.3% of
  pixels exceed [-1,1]. Final batch loss is 0.01442527. Low loss and f_mem=0
  do not establish useful generation. Some images are recognizable faces.
- Saved step-zero samples agree with z_init/0.01 for the zero-output epsilon
  network: RMSE 2.9974661e-5. This is an existing-artifact comparison, not a new
  inference run. Training/reference tensors have the expected shapes and range.
- Historical records show poor early FID and a best FID of 19.62167 at 25,000
  updates. Old/current sampling policy, LR, architecture, and budget differ.
  Exact deployed historical config/checkpoints are absent locally; settings
  reconstructed from source are labelled accordingly. Clamp causality is open.
- Saved the full findings, equations, source pointers, persistence limitations,
  metric distinctions, artifact hashes, and unexecuted GPU replay command in
  CRITICAL_EPS_EPS_SAMPLING_AUDIT_2026-09-09.txt, as requested. Linked it from
  README and docs/repo_map.md. Updated TODO to prioritize Berzelius diagnosis.
- Next session: obtain the current saved run on Berzelius, use mem_param_env,
  verify numerical contracts, and compare exact versus floor=0.05 with the
  same checkpoint/noise/device before controlled training. Recover the old
  checkpoint/config for matched replay. The user currently launches compute.
- Audit verification: static source/Git inspection, existing figures and CSV/
  JSON records, CPU reads of saved tensors in local memcompgen. No training,
  inference, checkpoint replay, or test suite ran during this audit. The audit
  itself made no code changes. Earlier implementation checks remain above.
- Closeout scope: publish the accumulated monitoring/CLI/sample-storage changes
  and this documentation on codex/reproducible-runners. Generated data, noise,
  checkpoints, and samples stay Git-ignored; no artifact transfer was performed.
- Closeout validation: local memcompgen Python parsed all eight changed/new
  Python files and the documented GPU replay example without executing them.
  git diff --check passed; saved run artifacts remain ignored and untracked.
  The earlier 25-test result above was not rerun during this closeout.

## 2026-09-09 - Final Berzelius access handoff

- The user supplied the existing cluster parent directory:
  /proj/rep-learning-robotics/users/x_moalj/project_mem_par. The requested
  layout preserves the historical checkout as old_mem_par and places the
  new mem_param_clean checkout beside it on codex/reproducible-runners.
- Provided rename/clone and HTTPS authentication instructions. The user
  reports access is sorted. No remote commands were executed by the agent;
  clone, rename, environment readiness, and artifact transfer remain unverified.
- Recorded the path and the user's instruction to assume less in the critical
  handoff and TODO. Check remote state before installing or launching compute.
  The earlier mem_param_env name remains a convention to verify.
- Before this documentation update, the local tree was clean at 7606bed and
  matched the fetched origin branch. That commit was previously verified on
  GitHub. No additional training, inference, or tests ran in this closeout.
- Documentation validation: git diff --check passed.

## 2026-09-09T13:19:36+00:00 - Compare epsilon-generation failure against recovered cluster artifacts

- User requested a code comparison and prioritized checks for restoring eps/eps
  generation. Read the critical audit and open problem, then inspected old/clean
  model, conversion, samplers, UNet, optimizer loop, monitor, data, and runtime.
- Recovered the actual old N=256 seed-0 April run under old_mem_par/outputs.
  Its step_025000 checkpoint, config, training/reference tensors, noise bank,
  optimizer state, figures, and historical quality CSV are available here.
  The Mac-era notes about missing old artifacts do not apply to this cluster.
- All 228 state tensors loaded strictly into a clean model configured from the
  saved historical network/data sections; parameter count is 3,917,409. No
  trained-network forward or generation ran. Existing 750/25k grids show the
  historical transition from mostly noisy/saturated outputs to clear faces.
- Saved config confirms LR=1e-4, N/batch=256, widths 32/64/128, time embedding
  derived as 128, dropout=0.1, attention=(16,8), uniform [0.001,0.999], Heun50,
  floor=0.05, and integrated-state output. The recent clean settings differ.
- CPU mem_param_env: Python 3.10.20, PyTorch 2.5.1.post303. The pytest entrypoint
  failed because pytest is absent; python -B tests/test_model_sampling.py ran
  the same module via unittest and passed all 13 tests in 0.147 seconds.
- Ad hoc analytical CPU checks compared old/clean Euler and Heun for three
  endpoint/floor settings: all six matched exactly. Identical direct epsilon
  loss also matched exactly. Zero-output exact epsilon sampling matched the
  expected z_init/0.01 scaling within FP32 tolerance (max abs 1.90735e-5).
- Added docs/eps_eps_recovery_audit_2026-09-09.md with the ranked causes, actual
  protocol differences, artifact locations, verification limits, and a
  syntax-checked but unexecuted five-case GPU replay example. The next decisive
  comparison is old/clean sampling of the same successful historical weights,
  followed by isolated floor changes. Current clean outputs/ is absent here;
  transfer the Mac N=32 run before replaying that separate failed checkpoint.
- No research training, checkpoint inference, dataset rebuilding, or Slurm jobs
  were launched. No implementation/config changes were made. Preserved the old
  checkout's pre-existing changes. Diagnosis remains open pending GPU replay.


## 2026-09-10T11:02:36+00:00 - Record independent division controls and stability options

- User asked to record the separation of training/sampling clamps and reason
  about numerical alternatives. Verified external sample/Euler/Heun expose
  use_safe_div and safe_div_denominator_floor; GDFM.sample forwards them. Training
  uses separate stored settings. No alternate clamp-policy selector exists.
- Added docs/division_stability_options_2026-09-10.md. Direct eps/eps training
  has no conversion division. The note distinguishes conditioning, cancellation,
  and solver error, and explains why matching cross-loss floors can bias the
  target relationship. Separate training/sampling floors should be preserved.
- Derived exact logit-time fields with no explicit endpoint division. Proposed
  them as a sampling-only experiment with existing weights. Explained that the
  same inaccurate continuous flow remains inaccurate after a time change.
- Documented the existing v/v plus eps_equivalent control as equivalent to a
  structured residual epsilon head. Also analyzed bounded loss weights, time
  importance sampling, precision, adaptive steps, projection, and variance-based
  preconditioning. None was implemented or claimed to repair generation.
- Read primary EDM and DPM-Solver sources for general preconditioning/integration
  principles; specialized FM equations are labelled as our derivations.
- Exact rational and scalar numerical checks passed for logit fields, residual
  loss, cross-loss clamp bias, and bounded preconditioning coefficients. The
  zero-epsilon logit-Heun solution converged toward the same 100x endpoint
  amplification, confirming the method's limitation. No model was evaluated.
- No research compute or implementation/config changes. Prior audit notes and
  pre-existing worktree changes remain intact. Recovery still awaits matched
  historical trained-checkpoint replay.


## 2026-09-10T11:16:19+00:00 - Consolidate the full answers into the critical epsilon audit

- User requested all numerical options and the preceding code-comparison answer
  in the existing critical text file, with a simple executive summary at its top.
- Added a plain-language summary covering what worked before, recovered cluster
  artifacts, changed settings, separate training/sampling clamps, proposed
  numerical approaches, and the next diagnostic checks. Kept the issue open.
- Preserved the original Mac audit verbatim under a dated historical label.
  Clarified that its old-artifact availability statements describe the Mac;
  historical artifacts exist on Berzelius while recent clean run outputs do not.
- Included the complete code-comparison and numerical-options documents as
  dated follow-up answers. Preserved equations, API examples, verification
  limits, source references, and the unexecuted GPU command. Adjusted relative
  document links for the root-level text file.
- Verification: checked exact preservation of the original audit and inclusion
  of both full follow-up documents; parsed the copied GPU Python example without
  executing it. Documentation only; no model/config changes or research compute.


## 2026-09-10T12:50:16+00:00 - Add direct epsilon sampling and optional final Euler

- User requested the simpler epsilon sampling conversion and an explicit check
  of Heun's final step. Implemented in mem_param_clean only.
- sampling.model_to_velocity uses (z_t-eps_hat)/t with the existing optional
  sampling denominator floor. sampling.eps_to_v="via_x" retains the previous
  reconstruction route for comparisons. Training conversions, targets, losses,
  and the final x_1 readout are unchanged.
- New configs and direct Python calls default to eps_to_v="direct" and
  heun_final_euler=True. Fifty intervals use 49 Heun intervals plus one Euler
  interval. False keeps full Heun. GDFM.sample, monitoring, and checkpoint replay
  forward both settings. Image CLI adds --heun-final-euler and its negative;
  --set supports both fields. Monitoring prints the chosen sampling behavior.
- New config.json and evaluation settings.json record both settings. Evaluation
  fills missing historical fields with via_x and False before overrides, so
  older saved runs preserve their behavior unless changed explicitly.
- Both inspected old samplers used full Heun; the successful historical config
  selected Heun50 over [.001,.999]. Missing final Euler is not an established
  old-to-clean regression. Epsilon divides by t near the start, while the
  paper's x prediction divides by 1-t near the end.
- x_1 still makes a separate model call at end_time. For epsilon with inactive
  endpoint clamping, this equals an Euler extension to t=1. Final Euler skips
  the final Heun correction, not that separate readout. x_t_max returns state.
- Updated README with real Python usage, comparison flags, and expected shape;
  config reference, repo map, TODO, and critical audit describe these changes.
  Made historical parity example and its critical-text copy explicitly request
  via_x/full-Heun. Original Mac audit remains unchanged with a current notice.
- Validation: Berzelius mem_param_env, Python 3.10.20, CPU.
  PYTHONPATH=. python -B tests/test_model_sampling.py: 20 tests passed.
  PYTHONPATH=. python -B -m unittest discover -s tests -p test_runners.py:
  24 tests passed in 3.106 seconds. Initial failure was a new test comparing
  JSON-loaded lists with original tuples; corrected the test comparison.
  The corresponding test_config_reference.py discovery passed 1 test.
  Total: 45 passing tests. Checks cover direct/clamped algebra, tensor shapes,
  unchanged training losses and gradients, all sampler entrypoints, final-step
  model calls, endpoint readout, CLI precedence, runner forwarding, legacy
  replay defaults, and small temporary CPU integrations with mocked tracking.
- Changed Python, README API example, and historical replay examples parsed
  with the project interpreter. System Python 3.6 cannot parse the existing
  Python 3.10 tests; the successful syntax check used mem_param_env.
  git diff --check passed before this append and is checked at closeout.
- No research training, saved research checkpoint inference, GPU generation,
  Slurm submission, commit, or push. Generation recovery remains unverified.


## 2026-09-10T13:08:24+00:00 - Prepare Berzelius epsilon recovery smoke jobs; do not execute

- User requested a comparison plan and scripts for 2,000-update W&B smoke jobs,
  explicitly reserving all execution/submission to the researcher.
- Selected CelebA32 grayscale to anchor comparisons to the recovered successful
  historical study. Added configs/runners/celeba32_eps_recovery_smoke.py with
  N=batch=256, seed0, UNet32/64/128, time128, dropout0.1, attention16/8,
  Adam1e-4, times [.001,.999], 50 sampling intervals, integrated-state output,
  FP32 plus CUDA TF32, checkpoints every250, and 128 saved raw samples.
- Added six rows: old via_x/full-Heun/floor.05; direct/full-Heun/floor.05;
  direct/final-Euler/floor.05; direct/full-Heun/exact; direct/final-Euler
  with LR1e-3; and native v/v reference. Training/sample settings otherwise
  match. Sampler-only jobs repeat training to obtain separate W&B histories
  through the existing runner; later checkpoint replay is the cheaper control.
- Added jobs/berzelius/eps_recovery_smoke/submit.sh and run_array.sbatch.
  Default array0-5%2, oneGPU/fourCPUs/32GB per task, two-hour limit, historical
  account berzelius-2026-78 with account/time overrides. No partition forced.
  submit.sh --dry-run prints only; task selection allows one job. Each actual
  submission will create a unique batch, copy config/matrix, and record source
  commit/diff and commands. Workers import the live source tree.
- Workers load Mambaforge and mem_param_env, set clean PYTHONPATH, read existing
  old_mem_par/data cache, and disable downloads. W&B uses existing login and
  WANDB_ENTITY/current default maj90, project celeba32-unet, common batch group,
  distinct variant names, and strict tracking failure behavior. No credentials
  were inspected; current allocation validity and runtime were not verified.
- Added read-only inspect_results.py for future user execution: stdout CSV
  reports raw finite/min/max/RMS/outside-range values, recorded losses and
  projected f_mem, run state, recorded data hashes, and computed noise hash.
  It saves no files. W&B itself currently shows loss, panels, and rough f_mem;
  raw range diagnostics are obtained from the helper after user-run jobs.
- Wrote the job README with exact commands, controlled contrasts, output paths,
  W&B setup, duplicate-training cost, checks, and next-stage decisions. Linked
  from root README/repo_map; recorded prepared-but-unrun status in TODO and
  critical audit. Two thousand updates are early-behavior checks, not restored
  historical convergence; the historical best result was around25,000 updates.
- Validation only: bash -n on both shell files; ast.parse on the new config
  and inspector without imports; static config-field and six-row matrix checks;
  confirmed each advertised pair changes only its intended settings. Independent
  read-only review found no blocking script/path/forwarding/hash issue.
  git diff --check passed. No project runtime tests or script dry-run executed.
- No training, inference, dataset loading, W&B connection, scheduler command,
  job submission, commit, or push. Existing source edits from the previous
  sampler task were preserved; this task changes scripts/config/docs only.


## 2026-09-14T13:33:20+02:00 - Launch one-million-step CelebA32 epsilon recovery array

- User requested a broad epsilon sweep at one million optimizer updates with the
  Bonnaire checkpoint schedule, plus native velocity and clean-data references.
- Added `configs/runners/celeba32_eps_recovery_1m.py` and
  `jobs/berzelius/eps_recovery_1m/`. The eight-task matrix contains six eps/eps
  runs covering via-x/direct sampling conversion, sampling floor on/off, final
  Euler on/off, and Adam learning rates 1e-4/1e-3, followed by v/v and x/x
  references. All tasks use seed 0, CelebA32 N=batch=256, the recovered
  3,917,409-parameter UNet recipe, and 72 `bonnaire-1m` checkpoints.
- Requested one GPU, four CPUs, 32 GB, and 48 hours per task. Submitted array
  `17513629`, tasks `0-7%4`, under account `berzelius-2026-78`. Batch snapshot:
  `outputs/job_sets/eps_recovery_1m/20260914T113123Z_2308577`.
- Validation before submission: both shell scripts passed `bash -n`; Python
  config and inspector parsed; the matrix resolved eight rows with six epsilon
  tasks and x/v references; checkpoint resolution returned 72 steps from 0 to
  1,000,000; dry-run and `git diff --check` passed. CelebA cache exists.
- Startup check: tasks 0-3 entered RUNNING. All loaded 256 training and 1,024
  held-out grayscale images and constructed the expected UNet. W&B initialized
  for all four. Tasks 0-2 completed step-0 sampling and entered training; their
  losses agreed through the first 225 updates. Tasks 4-7 remained pending under
  the four-task array cap. No commit or push ran.


## 2026-09-14T13:36:36+02:00 - Establish the Berzelius run journal

- User requested one durable record of every clean-repository compute launch and
  each later status check.
- Added root `RUN_JOURNAL.md` with a current active-runs table, a closed-runs
  section, append-only chronological checks, fixed status vocabulary, required
  Slurm/application evidence, failure-reason recording, and links to configs,
  matrices, batch directories, and W&B.
- Started the journal with job `17513629`, including its launch configuration,
  first Slurm/application check, task progress, and W&B links. Historical jobs
  remain in `old_mem_par/CODING_JOURNAL.md`; the new journal starts on 2026-09-14.
- Updated `AGENTS.md` so future launch and monitoring work must read and update
  the journal. Updated `docs/repo_map.md` to define its role beside TODO and the
  migration journal. Link existence and `git diff --check` passed. No additional
  job was submitted, cancelled, or changed during this documentation task.


## 2026-09-14T16:35:04+02:00 - Make small arrays fully parallel

- User set the standing launch rule: submit all tasks concurrently for arrays
  of about 80 tasks or fewer. For larger arrays, agree on and record a throttle.
- Raised live job `17513629` from a four-task throttle to eight. Slurm confirmed
  all eight tasks running; tasks 4-7 started at 16:33:53 CEST without cancelling
  or resubmitting tasks 0-3.
- Changed the epsilon-recovery launcher default from `0-7%4` to unthrottled
  `0-7`. Recorded the standing rule in `AGENTS.md` and `RUN_JOURNAL.md`.
- Validation: launcher syntax passed; its dry run selected `0-7` and emitted
  `--array=0-7`; `git diff --check` passed.


## 2026-09-14T17:16:25+02:00 - Align W&B metrics to optimizer updates

- Confirmed that W&B internal `_step` counted `run.log` calls, not gradient
  updates. Sparse training logs and separate checkpoint-generation logs made
  that default axis diverge from training progress.
- Added `optimizer_step` to every training and generation payload. Defined both
  `train/*` and `generation/*` against that shared W&B step metric. Step zero
  represents the initial generation panel; later values are completed updates.
- Retained `train/step` and `generation/step` as descriptive payload fields.
  Documented the shared axis in `docs/config_reference.md`.
- Added assertions for train, generation, and projected-f_mem payloads and for
  the W&B metric definitions. All 24 runner tests passed. The full suite remained
  at 66/67: the existing config-reference coverage test still lacks unrelated
  entries for `network.middle_attention`, `sampling.final_pixel_clamp`, and
  `tracking.wandb.subexperiment`.
- Job `17513629` was not restarted. Its eight active processes loaded the old
  logger before this edit, so their histories must still use `train/step` or
  `generation/step`; new runs will receive `optimizer_step`.


## 2026-09-15T12:38:57+02:00 - Add three-point eps/eps configuration evaluation

- Added a small read-only decision evaluation for the six completed `eps/eps`
  runs from job `17513629`. It checks only steps 100,000, 390,000, and 695,000;
  step zero, the full checkpoint trajectory, and the `v/v` and `x/x` references
  are excluded.
- Each source run uses its as-launched sampler: old via-x/full Heun, direct
  safe/full Heun, direct safe/final Euler, exact/full Heun, or exact/final
  Euler. The sixth run repeats direct safe/final Euler with training LR 1e-3.
- The evaluator reads saved checkpoints, `data_bundle.pt`, and `z_init.pt`
  without modifying source runs. It records FID from 10,000 generated samples,
  pixel-space f_mem at rho 0.3/0.5/0.7, held-out losses, and raw range/finite
  diagnostics. Exact replay noise is persisted and its first 128 samples are
  verified against the saved training noise.
- FID uses vendored torch-fidelity 0.4.0-beta with explicit -1_1 normalization.
  The saved held-out artifact limits its reference to 1,024 images, so absolute
  values are for comparison within this decision run, not old 10K-reference
  tables.
- `submit.sh` launches array `0-5` without a throttle and requests two hours per
  task. Results are append-safe. The dependent combiner writes one comparison
  CSV, one FID/f_mem/loss plot, and a summary.
- Added T018 to `TODO.txt` for the workshop-reviewer-motivated feature-space
  f_mem follow-up, including a fixed versioned encoder/layer and separate rho
  calibration rather than silently reusing pixel-space thresholds.
- Validation: the manifest resolves six tasks and three common nonzero
  checkpoints; launcher and Slurm shell syntax pass; eight focused evaluator
  tests pass. The full suite is 74/75 with only the pre-existing config-reference
  documentation failure for three unrelated fields.
- No evaluation job was submitted. No commit or push ran.


## 2026-09-15T13:17:47+02:00 - Correct FID-10k reference protocol

- Corrected a scientific protocol error identified after job `17522748`: its
  10,000 generated samples were compared with only 1,024 reference images.
  Those FID values are not FID-10k and must not drive the epsilon-configuration
  decision. Its held-out losses, pixel-space f_mem, and raw diagnostics remain
  valid. Added `FID_INVALID.md` to that output directory.
- Added a dedicated Berzelius reference-preparation job. It deterministically
  materializes 10,000 images from the CelebA `valid` split with the exact
  training preprocessing, saves the tensor and provenance manifest, and
  verifies that its first 1,024 tensors, labels, source indices, and sample IDs
  exactly match the training artifact.
- The evaluator now separates the frozen 10,000-image FID reference from the
  saved 1,024-image held-out set used for loss and pixel-space f_mem. Every row
  records the reference artifact path and SHA-256. FID feature caches are keyed
  by that artifact hash.
- `fid_10k` is now populated only when both the generated count and reference
  count are 10,000. The combiner rejects a 10k-generated run with fewer than
  10k references, and the launcher enforces the same condition before submit.
- The launch sequence is reference preparation, fully parallel array `0-5`,
  then combine, all connected by `afterok` dependencies.
- Validation: all Python and shell files pass syntax checks, the dry run reports
  `10000/10000/1024`, and all nine focused evaluator tests pass.
- Submitted replacement chain: reference job `17523389`, evaluation array
  `17523390`, and combine job `17523391`. Output is
  `outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point`.

- Completion: reference job `17523389`, all six tasks in array `17523390`, and
  combine job `17523391` completed with exit `0:0`. The combined summary has
  18/18 valid rows, `sample_counts=[10000]`, and
  `fid_reference_counts=[10000]`.
- Corrected FID-10k confirms direct safe/full-Heun as the next-run choice:
  20.57/24.43/27.13 at 100k/390k/695k versus old via-x/full-Heun at
  21.90/25.15/26.65. The old path is slightly better only at 695k and remains
  the close late-training reference.


## 2026-09-15T13:40:55+02:00 - Add v/v and x/x to corrected three-point eval

- Added a two-task manifest and Berzelius launcher for saved sources 6 (`v/v`)
  and 7 (`x/x`). Both reuse the exact corrected 10,000-image FID reference,
  deterministic evaluation noise, and FID feature cache, and use their saved
  sampling configurations at 100k, 390k, and 695k.
- Added a dependent reference combiner that checks for exactly six 10k/10k
  rows, writes separate reference artifacts, and also creates one 24-row
  all-parameterization CSV and plot. Cross-parameterization plots show only FID
  and pixel f_mem because eps/v/x loss magnitudes are not directly comparable.
- Validation: Python and shell syntax pass, the preview resolves exactly two
  tasks at three checkpoints, and all nine focused evaluator tests pass.
- Array `17523970` and combine job `17523971` completed with exit `0:0`.
  FID-10k is 25.86/28.72/30.19 for `v/v` and 25.69/27.70/24.97 for `x/x` at
  100k/390k/695k. All six reference rows are finite and use 10k/10k.


## 2026-09-15T13:52:00+02:00 - Prepare epsilon recovery work for publication

- Updated the root and job READMEs to point to the completed one-million-step
  training and corrected three-point evaluation. The docs now include the
  10k/10k FID protocol, the v/v and x/x references, and the unthrottled rule
  for small arrays.
- Added the three previously missing active settings to
  `docs/config_reference.md`: `network.middle_attention`,
  `sampling.final_pixel_clamp`, and `tracking.wandb.subexperiment`.
- Publication validation in `mem_param_env`: `git diff --check` passed; all
  relevant Slurm shell files passed `bash -n`; all changed notebooks passed
  JSON parsing; and the complete test suite passed, 76/76.
- The publication scope includes the safe-division and direct-sampling API,
  optimizer-step logging, recovery launchers, three-point evaluator, tests,
  audit notes, and run journals. It excludes generated outputs, the empty
  `tea_debug.log`, and unrelated untracked N=16 launcher experiments.


## 2026-09-16T13:15:07+02:00 - Implement raw-first feature f_mem and matched three-seed workflow

- Kept three distinct metric contracts. Training monitoring now records
  `monitor/random_projected_pixels`; post-hoc native pixels record
  `eval/full_pixels`; learned results record `eval/features/sscd`,
  `eval/features/lpips`, or `eval/features/dinov2`. Their rho values and absolute
  f_mem values are not treated as interchangeable.
- Added public fixed adapters and batched extraction for official SSCD
  `sscd_disc_mixup`, LPIPS v0.1 AlexNet, and DINOv2 ViT-S/14 final CLS. The
  LPIPS embedding is constructed so squared Euclidean distance matches package
  LPIPS. DINOv2 keeps the checkpoint-native 518 positional grid and interpolates
  it for the primary 224 input. Extended nearest-neighbor output with the second
  neighbor identity while retaining the existing `nn_idx` key.
- Added `jobs/berzelius/eps_recovery_1m/feature_f_mem/`. It freezes matched
  seed-0 or nine-trajectory manifests; replays 100k, 390k, and 695k using one
  seed-2003 noise bank and 50-step full Heun; saves exact 10,000-image banks;
  caches every train/validation/generated feature tensor; saves `d1`, `d2`, `r`,
  and both neighbor IDs; retains pixel rho 0.3/0.5/0.7 and held-out percentile
  curves; computes FID-10k; and rejects cache reuse when hashes or preprocessing
  metadata differ.
- Added the seed-0 checkpoint-390k sensitivity matrix: SSCD 288/320, LPIPS
  32/64, and DINOv2 224/518. Primary multi-seed resolutions remain 320, 32, and
  224. The combiner enforces expected record counts and writes seed values,
  mean/sample standard deviation, paired differences, per-image Spearman
  correlations, memorized-set Jaccard overlap, parameterization ordering, raw
  tables, and plots.
- Added `jobs/berzelius/eps_recovery_multiseed/`: one unthrottled six-task array
  for seeds 1 and 2 crossed with matched `eps/eps`, `v/v`, and `x/x`. It reuses
  the one-million-update config, N=batch=256, Adam `1e-4`, unit weighting, and
  the fixed sampler settings. Feature evaluation stays outside training.
- Staged and verified all encoder assets under the ignored
  `outputs/model_assets/feature_f_mem/` cache (594 MiB). The manifest pins
  DINOv2 source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and hashes
  SSCD, LPIPS calibration, AlexNet, DINOv2, and FID Inception weights. Compute
  workers use explicit local paths and cannot silently download another model.
- Validation: seed-0 manifest creation resolves exactly eps/v/x at the three
  checkpoints; both launchers pass `bash -n` and non-mutating dry-run; all 84
  repository tests pass; and offline real-weight smoke outputs are SSCD
  512-D at 320/288, LPIPS 5,760-D at 32 and 31,872-D at 64, DINOv2 384-D at
  224/518, and Inception 2,048-D. No Slurm job was submitted, so
  `RUN_JOURNAL.md` was not changed.


## 2026-09-16T14:05:31+02:00 - Add interactive learned-feature sanity notebook

- Added `notebooks/sanity/feature_f_mem_encoder_sanity.ipynb` for joint
  researcher and implementation review. It uses the staged offline assets,
  one saved seed-zero training artifact, and the frozen CelebA validation
  artifact.
- The notebook verifies file hashes, the pinned DINOv2 commit, feature
  dimensions, finite outputs, batch-size determinism, normalized SSCD/DINOv2
  descriptors, exact-copy neighbor identities, both neighbor indices, and
  held-out threshold mechanics. It plots controlled perturbations beside each
  space's nearest training images.
- Added an explicit LPIPS checkpoint-layout check. Every calibration tensor
  must match the loaded model state. The notebook also compares direct official
  LPIPS scores with squared Euclidean distances from the cached embedding.
- Corrected two notebook assumptions during execution. CelebA indices are
  split-local, so the data check now uses recorded source and split fields.
  CPU is the safe default because a visible CUDA runtime does not prove that
  the process owns an allocated GPU. An allocated session can set
  `FEATURE_F_MEM_DEVICE=cuda`.
- Full default execution passed in the Berzelius `mem_param_env` environment
  on CPU. Outputs were SSCD 512-D, LPIPS 5,760-D, and DINOv2 384-D. Exact-copy
  distances were zero with training index zero. The maximum direct-versus-
  embedding LPIPS error was `1.49e-8`.
- A second full CPU execution enabled the optional 288/64/518 resolution cell.
  Nearest-neighbor identity agreement on the six diagnostic queries was 1.000
  for SSCD 320/288, 0.833 for LPIPS 32/64, and 0.833 for DINOv2 224/518.
  These small-query values test the path and do not estimate experiment-level
  resolution sensitivity.
- Final checks passed for JSON parsing, unique cell IDs, empty saved outputs,
  required code-cell purpose comments, notebook sentence lengths, nbformat
  validation, and `git diff --check`. The environment did not contain
  `pytest`; the built-in unittest runner passed all 17 image-metric tests and
  all three feature-workflow tests.
- Linked the notebook from the feature workflow README, repository map, and
  T018 status. No Slurm job was submitted, so `RUN_JOURNAL.md` was not
  changed.

## 2026-09-16T16:42:53+02:00 - Replace encoder audit with visual feature-space study

- Replaced the prior assertion-led encoder notebook with a visual-first
  feature-space `f_mem` study. The committed `quick` profile uses saved CelebA
  tensors on CPU: A=32, calibration=128, validation=128, B=32, eight pairs,
  a 32-image query bank, primary resolutions, and staged local weights.
- The notebook displays A/B galleries, held-out ratio calibration, one complete
  interpolation/noise grid, SSCD/LPIPS/DINOv2 heatmaps for d1/d2/ratio/f_mem,
  nearest-A examples, known noisy-copy recovery curves, a compact audit, and a
  direct LPIPS parity scatter. It uses public feature and nearest-neighbor APIs
  directly and has no notebook-local helper functions.
- Full quick execution passed in `mem_param_env` and rendered nine
  primary-resolution figures in a disposable `/tmp` notebook. The audit passed:
  SSCD=512-D, LPIPS=5,760-D, DINOv2=384-D; each exact copy recovered A index
  zero; and LPIPS parity maximum error was `1.49e-08`. Pixel-exact DINO queries
  showed a separately-batched numerical d1 of `5.18e-04`, so the audit requires
  the recovered identity and d1 below `1e-3`, rather than bitwise-zero ratios.
- Focused validation passed: 17 image-metric tests, three feature-workflow
  tests, notebook JSON/schema and unique-ID validation, empty committed outputs,
  required code-cell purpose comments, no notebook-local functions, and
  `git diff --check`.
- The full GPU profile and its SSCD 288/320, LPIPS 32/64, and DINOv2 224/518
  resolution comparisons remain researcher-run checks. No Slurm job was
  submitted, so `RUN_JOURNAL.md` remains unchanged.


## 2026-09-18 - Repair feature notebook audit and add controlled A/B mixture diagnostic

- Reviewed the saved full-profile notebook audit without rerunning it. SSCD and LPIPS failed only the batch-size repeat limit.
- The recorded repeat errors were 6.03e-5 for SSCD and 6.16e-5 for LPIPS. The prior limit was 1e-5.
- Dimensions, finite features, exact-copy neighbor identities, exact-copy distances, and LPIPS parity all passed in the saved output.
- Set the explicit repeat tolerance to 1e-4. The new limit covers the recorded GPU numerical variation without weakening identity or distance checks.
- Added a tutorial definition of d1, d2, r, rho, and f_mem. Added line comments for artifact loading, source encoding, calibration, interpolation, the new mixture, and the audit.
- Added a controlled A/B mixture section. It replaces a known fraction of held-out B queries with noisy A copies. Each feature space plots mean and variance of r and of the binary f_mem decision.
- Restored the notebook default to quick CPU. The full GPU and resolution paths remain explicit researcher-run options.
- Validation only: notebook JSON parse, unique cell IDs, Python AST parse for every nonempty code cell, and git whitespace check passed. No notebook cell, feature extraction, model inference, test suite, or Slurm job ran.


## 2026-09-18 - Launch recoverable 3x3 and data/image-format gates

- Added complete process-boundary recovery for image and synthetic training.
  Two rotating states retain model, optimizer, Python/NumPy/Torch CPU and CUDA
  RNGs, the exact shuffled-batch cursor, history, elapsed time, and W&B run ID.
  A run lock prevents concurrent resumes; USR1 and TERM stop after an optimizer
  boundary; resumes verify saved config and data hashes and reuse the W&B run.
- Added physical/effective batch separation with exact gradient accumulation.
  The CelebA32 prediction/loss grid uses physical/effective batch 256 on thin
  A100s. The data/image-format study uses physical batch 128 and effective batch
  256 on fat A100s, including a fresh CelebA32 gray control and a deeper Flowers
  capacity probe.
- Added an opt-in deterministic CUDA mode and enabled it for both studies.
  Fresh processes now seed Python, NumPy, and Torch; monitoring and W&B preserve
  those streams. CUDA smoke `17549382` proved bitwise equality for model, Adam,
  all RNGs, sampler, and history across a planned stop and new process.
- Both study configs target one million updates but stop first at the scientific
  200k checkpoint. Resume wrappers default to the 200k gate and accept an
  explicit `1000000` target only after manual review. Source, config, and matrix
  snapshots are frozen per submission.
- Submitted unthrottled arrays `17549419` for all nine CelebA32 gray
  prediction/loss cells and `17549408` for seven EPS/EPS dataset, resolution,
  color, and capacity rows. The data/image-format array passed startup for all
  seven rows; the 3x3 array is waiting for thin GPUs.
- The 200k review is specified in both job READMEs: fixed 10k noise, full Heun,
  direct and legacy epsilon conversion where applicable, safe and exact
  division, FID, full-pixel f_mem, and SSCD/LPIPS/DINOv2 f_mem. Its generalized
  evaluator launcher still needs to be implemented before the gate completes.
- Validation passed: 87/87 local tests, all launch/resume shell syntax, config
  and real-loader shape checks, finite loss for all nine 3x3 cells, and clean
  whitespace checks. Operational attempts and immutable batch paths are in
  `RUN_JOURNAL.md`.


## 2026-09-19 - Complete and review both 200k training gates

- All 16 Slurm tasks completed with exit `0:0` and the applications paused at
  the planned step 200,000 boundary. Every run has a valid rotating recovery
  state and explicit step-200k checkpoint, so selected cells can resume in a
  new process without replaying the first 200k updates. All W&B finishes report
  success and all step-200k sample tensors and comparison grids are present.
- The online 128-sample projected-pixel monitor separates the data/image-format
  rows sharply at 200k: CelebA32 gray and MNIST32 are 0.977, CelebA32 RGB is
  0.727, CelebA64 gray is 0.117, CelebA64 RGB and Flowers64 base are 0.000, and
  Flowers64 deep is 0.031. Visual inspection agrees on the failure modes:
  CelebA64 RGB and both Flowers64 rows are poor or collapsed, and the deeper
  Flowers network does not solve that problem at this gate. CelebA64 gray is
  visibly more coherent than its projected-pixel score alone suggests.
- All nine prediction/loss cells produce recognizable CelebA32-gray faces.
  Their projected-pixel f_mem values span 0.469-1.000; `eps/x` is visibly the
  noisiest cell. Raw loss magnitudes are not ranked across different target
  parameterizations because their scales differ.
- The online monitor is only `monitor/random_projected_pixels` with 128 samples,
  resize 16, projection dimension 256, and rho 0.3. It is a triage gate, not a
  substitute for fixed-10k FID, full-pixel f_mem, or learned SSCD/LPIPS/DINOv2
  f_mem. The generalized evaluator named in the study READMEs remains the next
  implementation dependency before a defensible continuation ranking.
- No run was resumed toward one million and no evaluator job was submitted in
  this review. Exact process-boundary restart support was already accepted by
  CUDA smoke `17549382`; the remaining decision is scientific cell selection,
  not recovery feasibility.


## 2026-09-19 - Stage the EPS/EPS 64x64 N=512 capacity gate

- Added a deterministic 24-row matrix: four data/color formats crossed with
  small, deep, and wide-deep UNets and complete attention on/off controls.
  Small/deep rows route to 16 thin tasks; wide-deep rows route to eight fat
  tasks. The shared config keeps N=512, seed 0, effective batch 256, a one-
  million-update ceiling, and a hard 400k review gate.
- Added six real-GPU capacity probes. Each profile measures physical batches
  512/256/128/64 in a fresh process on CelebA64 RGB. Selection uses finite loss
  and peak-reserved limits of 32 GiB on A100-40GB or 64 GiB on A100-80GB.
  Every GPU job requests the full named GRES, so fat jobs cannot receive a
  10GB MIG slice.
- Added six online-W&B integration smokes and an online wide-deep attention
  exact-resume comparison. The resume check requires exact model, Adam, RNG,
  shuffled-batch, history, and monitoring artifacts, plus reuse of the saved
  W&B run ID.
- Added four fixed-noise sampling profiles at every requested checkpoint:
  direct-safe, via-x-safe, direct-exact, and via-x-exact. Each saves raw
  diagnostics, tensors, grids when finite, projected-pixel f_mem, and W&B
  visuals. Direct-safe remains required; nonfinite alternates remain recorded
  diagnostics during full training.
- Added guarded direct/split launchers. The runtime plan combines measured
  optimizer throughput and complete smoke-monitor walltime with a 30 percent
  margin. It runs directly below 40 hours and otherwise uses exact 200k
  `aftercorr` segments only when both segments stay below 40 hours with their
  actual 11/3 monitor counts. Submission IDs are recorded incrementally so
  partial failures cannot cause blind duplicate arrays.
- Isolated every worker and finalizer from the live checkout. Processes change
  into the frozen batch directory and assert the exact imported
  `mem_param_src/__init__.py` path before training, probing, or verification.
- Added a dependent 400k verifier for all 24 runs. It checks paused state,
  all 14 checkpoints, latest recovery, W&B success, and all four sampling-
  profile records at every monitor step before writing `GATE_400K_COMPLETE`.
  Manual recovery IDs are append-only, and a separate replacement-gate
  launcher can verify repaired jobs without depending on a stale failed gate.
- Validation passed in `mem_param_env`: all shell files pass `bash -n`; Python
  files compile; the static validator confirms 24 rows, six profiles, exact
  profile references, 16 thin routes, and eight fat routes; direct and split
  dry-run handoffs pass; `git diff --check` passes; and the combined repository
  suite passes 93/93 tests. No Slurm job was submitted in this implementation
  step, so `RUN_JOURNAL.md` was not changed.


## 2026-09-21 - Complete EPS/EPS 64x64 N=512 capacity gate

- All 24 seed-zero rows completed exact 0-200k and 200k-400k process segments.
  The dependent verifier passed every run, checkpoint, recovery state, W&B
  finish, and four-profile monitoring record before writing
  `GATE_400K_COMPLETE`.
- GPU probes selected physical/effective batch 256 without accumulation for all
  profiles. Actual small/deep segment runtimes were 10.3-11.5 hours. Actual
  wide-deep segment runtimes were 18.9-20.3 hours, below the 48-hour limit.
- The 128-sample direct-safe projected-pixel monitor identifies capacity as the
  dominant factor. At 400k, attention-averaged wide-deep f_mem is 0.930 for
  CelebA gray, 0.551 for CelebA RGB, 0.992 for MNIST gray, and 0.156 for
  Flowers RGB. Small/deep CelebA RGB and Flowers remain near zero or zero.
- Visual review shows the same transition. Wide-deep models are the first rows
  with recognizable RGB faces and flowers. Attention has mixed, secondary
  effects and does not rescue a low-capacity model.
- Both safe samplers remain finite and visually usable. Exact division creates
  large endpoint excursions and saturated grids for most small/deep face and
  flower rows. Direct-exact and via-x-exact remain numerically close, while the
  safe denominator floor makes direct and via-x distinct sampling paths.
- Added the complete preliminary report at
  `outputs/job_sets/eps_eps_64_n512/20260919T155341Z_3213634/RESULTS_400K.md`.
  It explicitly limits these findings to seed zero and the online projected-
  pixel monitor. No fixed-10k, full-pixel, learned-feature, or multi-seed claim
  is made, and no training continued beyond 400k.


## 2026-09-21 - Consolidate evidence and prepare the full-run paper decision

- Added a detailed 200k report for the seven-row EPS/EPS data/image-format
  study at
  `outputs/job_sets/eps_eps_data_image_format/20260918T164620Z_3539380/RESULTS_200K.md`.
  All seven runs completed with finite final tensors, step-200k checkpoints,
  rotating recovery states, and successful W&B finishes.
- The data/image-format trajectory is now recorded in full. Projected-pixel
  f_mem at 25k/50k/100k/200k is `0.672/0.805/0.906/0.977` for CelebA32 gray,
  `0.891/0.945/0.961/0.977` for MNIST32 gray,
  `0.031/0.328/0.609/0.727` for CelebA32 RGB,
  `0.000/0.023/0.102/0.117` for CelebA64 gray,
  `0.000/0.000/0.000/0.000` for CelebA64 RGB,
  `0.000/0.000/0.000/0.000` for Flowers64 base, and
  `0.000/0.000/0.008/0.031` for Flowers64 deep.
- Visual review separates quality from memorization. CelebA64 gray contains
  coherent faces despite f_mem 0.117. CelebA64 RGB mixes faces with saturated
  color blocks. Both Flowers rows are dominated by color blocks. Adding a
  fourth depth level does not recover Flowers at 200k.
- The seven-row matrix is not a full factorial design. Only four comparisons
  isolate one declared change: CelebA32 gray versus RGB, CelebA32 gray versus
  CelebA64 gray, CelebA64 gray versus RGB, and Flowers base versus deep. The
  result identifies failure regimes but does not estimate independent dataset,
  resolution, and color main effects.
- Consolidated all completed evidence in `PAPER_NOTES.txt`. The corrected
  one-million CelebA32 evaluation shows that direct-safe full-Heun EPS/EPS has
  FID-10k `20.57/24.43/27.13` and pixel f_mem
  `0.9059/0.9589/0.9709` at 100k/390k/695k. The x/x and v/v references already
  reach pixel f_mem about one at 100k, but their saved sampler uses final Euler.
  A common no-final-Euler replay is required before a paper-level FID ordering.
- Recorded the complete 3x3 result and its trajectory. The 200k matrix is
  `[[1.000, 0.469, 0.977], [0.961, 0.961, 0.875], [1.000, 0.961, 1.000]]`.
  x/x, v/x, and v/v are near saturation by 25k. x/v rises late. x/eps remains
  lower at 200k. eps/x and eps/v are non-monotonic at the displayed monitor
  points. All cells generate recognizable faces, so the timing differences are
  not explained by complete generation failure.
- Recorded the cross-study interpretation. Parameterization changes the route
  to memorization. Quality and memorization are distinct axes. A valid
  parameterization comparison first requires a useful-generation regime.
  Capacity dominates the difficult 64x64 RGB cases, while optional attention
  has only mixed seed-zero effects. Safe division is required for robust
  EPS/EPS sampling in `[0.001,0.999]`; exact division remains a diagnostic.
- Recorded the evidence limits. The 3x3, data/image-format, and 64x64 capacity
  summaries use seed-zero, 128-sample random-projected-pixel monitoring. They
  do not establish fixed-10k, full-pixel, learned-feature, or multi-seed
  rankings. The 1M endpoints also lack the corrected fixed-10k evaluation.
- Added T019 and T020 to `TODO.txt`. The next session must choose between a
  full multi-seed 3x3, a three-seed diagonal study, or a predeclared selected
  matrix. It must then freeze the common sampler, checkpoint and quality-
  matching rules, feature/FID artifact contract, compute budget, and AISTATS
  figure/table plan before submission.
- Added a provisional figure plan to `PAPER_NOTES.txt`: a main quality-
  memorization trajectory, prediction/loss heatmaps, qualitative nearest-
  neighbor panels, a 64x64 operating-regime control, a sampler-stability
  appendix, metric-agreement analysis, and a final multi-seed result table.
  These are candidates, not final paper selections.
- No compute was submitted. Existing runs remain unchanged and recoverable:
  nine 3x3 rows at 200k, seven data/image rows at 200k, and 24 capacity rows at
  400k. `squeue -u x_moalj` was empty at the final check.
- Added the current conservative abstract to `PAPER_NOTES.txt`. It removes the
  unsupported claim of a consistent parameterization ordering. The revised
  prose follows a research-paper question, evidence, and implication structure
  instead of listing the protocol like a report. Dense technical noun phrases
  are now replaced by short descriptions of what the model or measurement does.
  The abstract states only that the experiments indicate different
  memorization trajectories, separation between quality and memorization, and a
  capacity requirement for interpretable results. Multi-seed and learned-
  feature results remain future validation rather than completed claims.

## 2026-09-25 - Add Small DiT as a plug-and-play backbone

- Added `SmallDiT` with the shared unconditional `forward(z, t)` contract and
  raw image-shaped output.
- Adapted official `willisma/SiT` commit
  `cbde832a40b153ccc79603412409da9c9b0c568c`. Retained its MIT license beside
  the implementation and included that license in package data.
- Extended the existing factory and base config with `network.arch="dit"` and
  the Small defaults. Image size and channels still come from `cfg.data`.
- Kept GDFM prediction, loss, conversion, sampling, checkpoint, and evaluation
  semantics unchanged.
- Added a thin sanity notebook for factory construction, shape and parameter
  inspection, loss, and existing sampling calls.
- Verified reduced 8x8 depth-1 models across supported config layouts, all nine
  prediction/loss cells, Heun sampling, state-dict reload, continuous times,
  zero initialization, and invalid dimensions.
- Ran one bounded 32x32 forward with the actual depth-12, width-384, six-head
  Small configuration. It has 32,563,972 parameters and returned `[1,1,32,32]`.
- `python -m unittest discover -v tests` passed all 105 tests in
  `mem_param_env`, including the 12 new DiT tests and existing MLP/UNet tests.
- Executed `notebooks/sanity/dit_backbone_sanity.ipynb` through nbconvert. It
  constructed Small DiT through GDFM and returned finite loss and sample APIs.
- No jobs were submitted and no shared environment was changed.


## 2026-09-25 - Launch the requested Small-DiT diagonal pilot

- The researcher explicitly selected a compact follow-up: CelebA32 gray at
  32x32, N=256, seed 0, 100,000 optimizer updates, and the diagonal `x/x`,
  `eps/eps`, and `v/v` cells. This operational pilot does not settle T019's
  later paper-level multi-seed or full-matrix design.
- Added `configs/runners/celeba32_dit_diagonal_100k_viax.py` and the matching
  `jobs/berzelius/celeba32_dit_diagonal_100k_viax/` launch and exact-resume
  wrappers. They reuse GDFM, the image runner, checkpointing, monitoring, and
  sampling without architecture-specific branches.
- Fixed the shared controls to unit loss weighting, Adam at 1e-4, physical
  batch 64, effective batch 256 through accumulation 4, 28 Bonnaire
  checkpoints, full 50-step Heun, safe denominator floor 0.05, and
  `sampling.eps_to_v=via_x`. The via-x setting affects epsilon prediction
  conversion only; native x and v predictions do not traverse that route.
- Config and dry-run validation constructed the actual 32,563,972-parameter
  Small DiT. Shell syntax passed, the CelebA cache was present, and the batch
  snapshot includes frozen source/config/job files plus SHA-256 hashes.
- Submitted array `17639669` under `berzelius-2026-78`. Task 0 passed CUDA
  startup and reached step 500 with finite loss and monitoring; tasks 1-2
  were pending for resources at the recorded check. `RUN_JOURNAL.md` owns all
  subsequent operational status.


## 2026-09-26 - Launch the Small-DiT 3x3 training-size production matrix

- The researcher selected one seed of the complete prediction/loss 3x3 across
  CelebA32-gray training sizes. Interpreted the typed `1028` as the standard
  power-of-two size 1024 and froze `N={512,1024,2048,4096}`, seed 0, and all
  36 Cartesian-product rows.
- Added `configs/runners/celeba32_dit_3x3_nscale_1m_viax.py` and
  `jobs/berzelius/celeba32_dit_3x3_nscale_1m_viax/`. Every run declares the
  full one-million-update target before training. Each allocation lasts 24
  hours and uses signal-triggered exact recovery for researcher-selected
  manual resume; no automatic requeue or resume chain is enabled.
- Fixed Small DiT-S/2, unit loss weighting, Adam at 1e-4, physical/effective
  batch 64/256, exact training conversions, and the 72-checkpoint
  `bonnaire-1m` schedule. Sampling uses via-x epsilon conversion, safe
  denominator floor 0.05, full 50-step Heun, `x_t_max`, and unclipped raw
  output.
- Kept W&B strict for this launch. The current non-strict failure path can
  preserve local training but later overwrite the recovery payload's W&B ID
  with null, risking a second W&B identity. Strict tracking instead stops the
  allocation after a valid local recovery point and preserves tested same-ID
  manual resume behavior.
- The production validator confirmed exactly 36 rows, the actual
  32,563,972-parameter model, and all 72 checkpoints. Shell and Python syntax,
  array validation, data-cache presence, dry-run submission, and all 115
  frozen snapshot hashes passed.
- Submitted unthrottled array `17641282` under `berzelius-2026-78`. Each task
  requests one thin A100-SXM4-40GB, four CPUs, 32 GB, and 24 hours. The
  completed N=256 pilot implies about four allocations per row and roughly
  3,100 A100-hours for the whole matrix; this remains an estimate.
- This seed-zero matrix does not provide seed aggregation. Fixed-10k FID,
  full-pixel f_mem, learned-feature f_mem, and paper figures remain post-hoc
  work on saved artifacts. `RUN_JOURNAL.md` owns operational status.

## 2026-09-27 - Add and launch fresh MNIST diagonal production runs

- Added `configs/runners/mnist_diagonal_1m.py` and
  `jobs/berzelius/mnist_diagonal_1m/` for 36 new trajectories. The matrix
  covers MNIST32 U-Net-S and MNIST64 U-Net-L, N={512,2048}, seeds 0-2,
  and the `x/x`, `eps/eps`, and `v/v` diagonals. No prior run is reused.
- Fixed physical/effective batch 256/256 with accumulation one, Adam at
  1e-4 with established defaults, exact training conversions, deterministic
  fp32 tensors with TF32 enabled, and one million optimizer updates.
- Fixed checkpoint sampling to 50-step full Heun, safe floor 0.05,
  `eps_to_v=via_x`, `x_t_max`, and no final clamp. Both resolutions use
  the same 20 checkpoint steps.
- The launcher freezes source, config, job files, environment packages, and
  hashes. MNIST32 requests thin A100-40GB nodes for 36 hours. MNIST64 uses
  five 200k stages on fat A100-80GB nodes with 30 hours per stage.
- The dedicated W&B project is `mnist-unet-diagonal-1m`. Exact resumes keep
  one W&B identity per trajectory.
- Validation passed for the 36-row matrix, parameter counts, paired and
  nested MNIST IDs, shell and Python syntax, and a write-free submission
  preview. The real cache produced paired train/test IDs, shapes
  `[2048,1,32,32]` and `[2048,1,64,64]`, and range `[-1,1]`.
- All 56 focused model and runner tests passed in `mem_param_env`.
- Submitted the production arrays and dependent completion verifier.
  `RUN_JOURNAL.md` records job IDs and live scheduler state.

## 2026-09-27 - Add and launch fresh CelebA32 U-Net-S full 3x3 production runs

- Added `configs/runners/celeba32_unet_3x3_1m.py` and
  `jobs/berzelius/celeba32_unet_3x3_1m/` for 108 fresh trajectories: the full
  prediction/loss 3x3 at N={512,1024,2048,4096} and seeds 0-2.
- Fixed U-Net-S at 3,917,409 parameters, Adam at 1e-4, physical/effective batch
  256/256, deterministic FP32 with TF32 disabled, unit weighting, exact
  training conversions, and the 72-point Bonnaire schedule through one million.
- Fixed monitoring to safe division with floor 0.05, `eps_to_v=via_x`, 50-step
  full Heun on `[0.001,0.999]`, `x_t_max`, no final Euler, and no final clamp.
- Added frozen-source enforcement, exact-resume selection validation, strict
  completion checks, paired target-cell identities, prefix-nested N identities,
  dedicated W&B project `celeba32-unet-3x3-1m`, and operator documentation.
- Validation passed for all 108 rows, model size, Adam defaults, 72 checkpoints,
  batch and arithmetic contracts, shell/Python syntax, dry-run submission, and
  56 focused model/runner tests.
- Submitted unthrottled array `17645398` with task IDs 0-107. Each task requests
  one thin A100-40GB for 24 hours and targets the absolute one-million step.
  `RUN_JOURNAL.md` owns live status and tomorrow's exact-resume audit.

## 2026-09-27 - Prepare fresh CelebA64 RGB diagonal production runs

- Added `configs/runners/celeba64_rgb_diagonal_1m.py` and
  `jobs/berzelius/celeba64_rgb_diagonal_1m/` for 18 fresh trajectories.
  The matrix is CelebA64 RGB, N={512,2048}, seeds 0-2, and diagonal
  `x/x`, `eps/eps`, and `v/v` targets. Existing gray and prototype runs
  remain supporting evidence and are not reused.
- Fixed the successful 24,386,627-parameter U-Net profile: base width 64,
  multipliers `(1,2,4,4)`, two residual blocks, dropout 0.1, time embedding
  128, attention at 16 and 8 pixels, and middle attention.
- Fixed Adam at 1e-4, physical/effective batch 256/256 with accumulation one,
  exact training conversions, deterministic FP32 with TF32 enabled, one
  million optimizer updates, and the 72-point Bonnaire checkpoint schedule.
- Fixed monitoring to one sampler: 50-step full Heun on `[0.001,0.999]`,
  safe floor 0.05, `eps_to_v=via_x`, `x_t_max`, no final Euler step, and
  no final clamp. The prototype's three diagnostic sampler profiles are not
  part of production.
- Added five unthrottled `aftercorr` arrays on fat A100-80GB nodes. They stop
  at 200k, 400k, 600k, 800k, and 1M, with 30 hours per stage and exact
  signal-triggered recovery. The expected total is about 1,800 A100-hours.
- The completion verifier requires all 72 checkpoints, monitoring artifacts,
  successful unique W&B identities, contiguous recovery histories, fresh
  non-hard-linked artifacts, paired target-cell source IDs and fixed noise,
  and prefix-nested N=512/N=2048 training IDs.
- Validation passed in `mem_param_env`: the 18-row matrix, architecture,
  optimizer defaults, batch contract, sampler, 72 checkpoints, stage stops,
  shell/Python syntax, and write-free submission preview. The real CelebA
  cache produced train shape `[512,3,64,64]`, reference shape
  `[1000,3,64,64]`, and range `[-1,1]`.
- The focused model/sampling suite passed 24 tests. The runner suite passed
  32 tests. No batch was frozen and no Slurm job was submitted.



## 2026-09-28 - Add and launch native-v/v loss-weighting controls

- Added `configs/runners/celeba32_unet_vv_weighting_1m.py` and
  `jobs/berzelius/celeba32_unet_vv_weighting_1m/` for the explicit weighting
  block from the eight-block training plan.
- The matrix contains exactly nine fresh trajectories: CelebA32 gray at N=512,
  seeds 0-2, native `v/v`, and raw `x_equivalent`, `unit`, or
  `eps_equivalent` loss weighting. No weight is normalized or deduplicated.
- Fixed the existing 3,917,409-parameter U-Net-S, Adam at 1e-4, batch 256/256,
  accumulation one, exact training conversions, the 72-point Bonnaire schedule,
  deterministic FP32 with TF32 disabled, and one million optimizer updates.
  Monitoring remains safe via-x, 50-step full Heun, and unclamped `x_t_max`.
- Within each seed, all three weights use the same source IDs, initial model,
  training order and random stream, and fixed monitoring noise. Each trajectory
  still owns fresh files, a fresh run directory, and a unique W&B identity.
- The launcher freezes source, config, job files, environment, and hashes. Thin
  A100-40GB tasks get 24 hours and a five-minute recovery signal. Continuation
  is manual after an exact-recovery audit; automatic requeue is disabled.
- Validation passed for matrix uniqueness, `(1-t)^2`, `1`, and `t^2` formulas,
  model size, optimizer defaults, batch, 72 checkpoints, sampler, shell/Python
  syntax, and write-free submission. All 56 focused model/runner tests passed.
- Ruff was unavailable in `mem_param_env`; that optional lint check did not run.
  The failed pre-launch command created no batch and submitted no job.
- Submitted unthrottled array `17655496`, tasks 0-8, under account
  `berzelius-2026-78`. Frozen batch is
  `outputs/job_sets/celeba32_unet_vv_weighting_1m/20260928T144550Z_222375`.
  `RUN_JOURNAL.md` owns live status and later exact-resume audits.


## 2026-09-28 - Add and launch the N=2048 weighting extension

- Added `configs/runners/celeba32_unet_vv_weighting_n2048_1m.py` and
  `jobs/berzelius/celeba32_unet_vv_weighting_n2048_1m/` as a separate sibling
  to the already-frozen N=512 weighting block.
- The new matrix contains nine fresh trajectories: N=2048, seeds 0-2, native
  `v/v`, and raw `x_equivalent`, `unit`, or `eps_equivalent` weighting. No
  submitted N=512 run or artifact was changed or reused.
- Kept the same 3,917,409-parameter U-Net-S, optimizer, batch 256/256, exact
  training, 72 checkpoints, deterministic FP32, safe via-x full-Heun monitor,
  thin A100-40GB allocation, and manual exact-resume policy.
- The validator proves that each deterministic N=512 training selection is the
  prefix of its matching N=2048 selection. Identical seeds also pair model
  initialization, data order, random streams, and monitor noise across sizes.
- Validation passed for shell/Python syntax, nine unique rows, all raw weight
  formulas, model size, batch, checkpoints, sampler, prefix identities, three
  focused weighting tests, write-free submission, and all frozen hashes.
- Submitted unthrottled array `17655777`, tasks 0-8, under
  `berzelius-2026-78`. Frozen batch is
  `outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/20260928T151300Z_373224`.
  It shares W&B project `celeba32-unet-vv-weighting-1m` with a separate group.


## 2026-09-29 - Add and launch seed-1 Small-DiT robustness runs

- Added `configs/runners/celeba32_dit_robustness_s1_1m_viax.py` and the
  isolated `jobs/berzelius/celeba32_dit_robustness_s1_1m_viax/` package.
- The matrix contains 18 fresh CelebA32-gray trajectories: seed 1, full
  prediction/loss 3x3, and N={512,2048}. Seed 2 remains unprepared and
  CelebA64 grayscale is deferred.
- Exact comparison against the seed-0 config proved that only the study name,
  output name, seed, and W&B identity metadata differ. Architecture, optimizer,
  batch 64/256, exact training, deterministic FP32/TF32 runtime, one-million
  target, 72 checkpoints, and safe via-x full-Heun monitoring are unchanged.
- Real CelebA selection validated that seed-1 N=512 is the prefix of N=2048
  and differs from seed 0. All 18 W&B names are unique fresh seed-1 identities.
- Shell/Python syntax, write-free submission, the 32,563,972-parameter model,
  matrix factors, frozen recovery workflow, all hashes, and 68 focused
  DiT/sampler/runner tests passed. Pytest is not installed in `mem_param_env`,
  so the standard-library unittest runner executed the focused suite.
- Submitted unthrottled thin-A100 array `17662195`, tasks 0-17, under account
  `berzelius-2026-78`. Frozen batch:
  `outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178`.


## 2026-10-02 - Package the MNIST metric and add dataset-aware evaluation

- Validated the completed interactive encoder artifacts under
  `outputs/mnist_jointae32_interactive_v1/`. `TRAINING_COMPLETE` is present.
  The checkpoint, test-feature bank, and split-index SHA-256 values match the
  training manifest. All four local raw MNIST files also match its hashes.
- Confirmed the selected epoch is 10. Validation accuracy is 0.9930 and test
  accuracy is 0.9935. The saved test features are finite `[10000,128]` values.
  Recomputed float64 mean and covariance match their saved values exactly.
- Added `metrics/image/mnist_jointae.py` and bundled the verified checkpoint,
  provenance manifest, and compact global/per-class test statistics under
  `metrics/image/assets/`. The public loader uses `importlib.resources`, checks
  the weight hash, freezes all parameters, and returns the raw linear 128D
  representation.
- Froze one MNIST32/MNIST64 preprocessing rule: declared range to `[0,1]`,
  replicated RGB to grayscale only, clamp, bilinear resize to 32, and antialias
  on downsampling. No learned alignment is applied.
- Added raw-feature float64 Fréchet statistics, stable symmetric PSD square
  roots, global JointAE FD, per-class FD records, and run-training-bank
  standardization plus row L2 normalization for JointAE-space memorization.
- Removed the old 0.001 default held-out quantile. The frozen public alpha set
  is now `{0.005,0.01,0.02,0.05}`.
- Added `runners/image_protocol.py` and protocol flags to the existing image
  evaluator. MNIST `fid` resolves to JointAE FD. Natural-image `fid` resolves
  to Inception FID. Natural-image f_mem accepts pinned SSCD, DINOv2, and LPIPS
  spaces through the verified schema-v3 asset manifest. Incompatible
  dataset/encoder requests fail explicitly.
- The protocol reads and hashes saved run artifacts, selects checkpoints,
  samples once, caches raw features and float64 Fréchet statistics, retains
  both neighbor identities, writes per-alpha f_mem and per-space completion
  markers, and resumes only from matching cache metadata. Manifests record the
  saved noise seed, encoder hashes, environment, elapsed time, and failures.
  The legacy all-checkpoint evaluator remains unchanged.
- Replaced the 1.5 MB exploratory MNIST notebook with an 18 KB output-free
  qualification notebook. It uses the standalone trainer, restores the
  validation-selected checkpoint, and demonstrates reconstruction,
  classification, FD controls, and memorization without owning runtime code.
- Added `docs/image_evaluation.md`, package-data settings, repository-map and
  config-reference entries, and focused asset, metric, registry, and
  restartability tests.
- Verification used `mem_param_env`. Sixty-five focused and regression tests
  passed, and a no-build-isolation wheel contained all three metric assets.
  Fixed-weight CPU evaluation covered all 55,000 training,
  5,000 validation, and 10,000 test images. Validation/test accuracies exactly
  matched 0.9930/0.9935; losses differed from the GPU-saved values by less than
  `2e-7`. All 10,000 CPU test features were finite and differed from the saved
  GPU features by at most `1.0009e-3`; cross-device output was not bitwise
  equal. Frozen-weight controls increased monotonically with Gaussian noise,
  translation, rotation, and progressive class-support collapse. The official
  test half-split FD floor was 6.8939. The notebook schema validated, all ten
  code cells compiled, and no notebook outputs or blank cells remain. Full
  ten-epoch notebook execution and a real production-checkpoint protocol run
  were not performed.


## 2026-10-02 - Add minimal one-checkpoint image evaluation notebooks

- Added `notebooks/sanity/mnist_checkpoint_metrics.ipynb`.
  It selects one saved checkpoint, reuses saved noise, generates eight samples,
  and reports JointAE FD, class-aware FD, calibrated JointAE f_mem, raw sample
  diagnostics, a four-by-two grid, and saved artifact paths.
- Added `notebooks/sanity/celeba_checkpoint_metrics.ipynb`.
  It follows the same flow and reports Inception FID plus calibrated SSCD
  f_mem through the verified schema-v3 encoder asset manifest.
- Both executed notebooks use step 1,000,000 from the seed-zero, 512-example,
  x/x U-Net-S run. Their settings cells expose the run, step, and noise bank.
  The notebooks label all eight-sample values as smoke metrics.
- Updated `docs/image_evaluation.md` and `docs/repo_map.md` with the two
  entrypoints and their metric scope.
- Validation used `mem_param_env`. Both notebooks executed all seven code cells
  on CPU and store their tables and four-by-two sample grids without errors.
  Each protocol completed with a finite `(8, 1, 32, 32)` sample tensor.
- The MNIST smoke values are JointAE FD `134.9258`, macro present-class FD
  `76.6609`, and calibrated JointAE f_mem `1.0` at all four alpha levels.
- The CelebA smoke values are Inception FID `212.7642` and calibrated SSCD
  f_mem `1.0` at all four alpha levels. These eight-sample values are not
  paper-scale estimates.


## 2026-10-03 - Add and launch the sampling runtime gate

- Added `docs/unified_checkpoint_evaluation_plan.md`. It fixes one scheduler
  task per run stage, checkpoint-level completion, one deterministic 10,000-
  image bank per checkpoint, dataset-aware quality metrics, four memorization
  spaces, held-out threshold calibration, compact top-two neighbor evidence,
  storage retention, and metric-specific restart boundaries.
- Recorded a required input gap: the selected CelebA production run stores
  1,024 held-out images. Paper-scale evaluation needs one separately frozen,
  hashed, disjoint 10,000-image reference bank before the proposed 1,024/8,976
  calibration and validation split can run.
- Added `mem_param_src/runners/sampling_benchmark.py`. It loads exact saved
  artifacts, extends the recorded noise stream while proving its saved prefix,
  times complete sampling batches, keeps diagnostics and previews outside the
  timed section, and projects the 10,000-sample result across the actual saved
  checkpoint count. It does not compute metrics or save generated images.
- Added the output-free, dataset-selectable
  `notebooks/sanity/checkpoint_sampling_runtime.ipynb` and the frozen short-job
  package under `jobs/berzelius/checkpoint_sampling_runtime/`.
- Validation used `mem_param_env`. Four focused unit tests passed. Python and
  shell syntax passed. Ruff passed. All six notebook code cells compiled; the
  eight-cell notebook has no outputs. The write-free submission preview passed.
- A real one-sample CPU integration loaded the selected step-1,000,000 CelebA32
  checkpoint, found all 72 checkpoints, used the saved full-Heun settings,
  generated finite output, and wrote the timing artifacts successfully. This
  was an execution check, not a useful GPU throughput estimate.
- The first submission attempt stopped before `sbatch` because of one Conda
  wrapper shell setting. After the launcher fix, submitted job `17692093` under
  project `berzelius-2026-269`. Frozen batch:
  `outputs/job_sets/checkpoint_sampling_runtime/20261002T224818Z_1228250`.
- The normal submission remained pending. Slurm denied an in-place QoS change,
  so job `17692093` was cancelled with zero elapsed time. Added the permitted
  `devel` QoS to the launcher and submitted replacement job `17692132`. Its
  frozen batch is
  `outputs/job_sets/checkpoint_sampling_runtime/20261002T225610Z_1277373`.

## 2026-10-03 - Record historical full-evaluation runtimes

- Added the retained old-repository timing evidence to
  `docs/unified_checkpoint_evaluation_plan.md` and linked its scope from the
  repository map.
- Cross-checked the old per-checkpoint CSV timers against retained Slurm
  accounting for jobs `16473994`, `16479328`, and `16484437`.
- Slurm assigned both A100 classes. The three arrays used 111 A100-40GB tasks
  and 81 A100-80GB tasks in total. Runtime summaries therefore describe the
  historical mixed allocation rather than one GPU-memory class.
- The original 84-run evaluation processed 72 checkpoints and 10,000 generated
  images per checkpoint. Its median checkpoint took `78.897` seconds. Its
  median one-run task took `01:34:58`. The suite consumed `135.10` GPU-hours.
- The 97-run capacity evaluation used the same 72-checkpoint and 10,000-image
  protocol. Its median checkpoint took `143.707` seconds. Its median task took
  `02:53:37`. The suite consumed `243.05` GPU-hours.
- Sampling dominated both measurements. Median sampling time was `65.224`
  seconds per original checkpoint and `129.319` seconds per capacity
  checkpoint. Median FID time stayed near ten seconds.
- The summed checkpoint timers matched median Slurm wall time within about 31
  seconds for the original suite and 72 seconds for the capacity suite.
- These values support one run per staged scheduler task. They do not include
  the new SSCD, LPIPS, DINOv2, or JointAE branches, so they are lower-bound
  evidence for the complete unified protocol.
- This audit was read-only. It launched no compute and changed no old outputs.

## 2026-10-03 - Add the full interactive CelebA32 checkpoint metric gate

- Added `runners/checkpoint_metrics_benchmark.py`. It samples one saved
  checkpoint into one deterministic 10,000-image bank, reports raw-pixel
  `f_mem` at fixed `rho={0.3,0.5,0.7}`, reports calibrated raw and SSCD `f_mem`
  curves, and computes Inception FID-10k.
- The evaluator uses the frozen 10,000-image CelebA validation bank. It verifies
  its artifact and tensor hashes, source catalog, preprocessing, and exact
  1,024-image prefix against the selected production run before evaluation.
- It saves the generated bank, SSCD and Inception features, FID statistics, and
  nearest and second-nearest evidence. Threshold changes can therefore reuse
  the saved evidence without sampling or encoding again.
- Added `run_interactive_metrics.sh` as the direct command for an existing GPU
  allocation. The runner prints sampling, encoding, neighbor, metric, and time
  progress and saves a final `summary.json`.
- Updated the command README, unified evaluation plan, and repository map.
- Validation used `mem_param_env`. Thirty-two focused sampling, metric, and
  image-protocol tests passed with `unittest`. Ruff, Python compilation, CLI,
  and shell syntax checks passed.
- A read-only real-artifact check verified the 10,000-image validation bank,
  the exact 1,024-image saved prefix, and both pinned weight hashes. One-image
  CPU smoke checks produced finite SSCD `[1,512]` and Inception `[1,2048]`
  features. The interactive wrapper adds the established pinned
  `old_mem_par/third_party/torch-fidelity` source path.
- No GPU evaluation or scheduler job was launched. The paper-scale runtime is
  still unmeasured until the researcher runs the interactive command.

## 2026-10-03 - Freeze the clean CelebA test-10k reference

- Completed at `2026-10-03T01:40:03Z`. Added
  `runners/celeba_test_reference.py` and built the fixed seed-1 official-test
  bank at
  `outputs/reference_banks/celeba_test_seed1_clean_n10000.{pt,json}`.
- The builder verified 201 saved source runs across the current CelebA32 U-Net,
  DiT, weighting, and CelebA64 suites. It mapped every saved split-local train
  index through the official train catalog. The union contains 11,988 unique
  global filenames, JPEG hashes, and CelebA32 transformed-tensor hashes.
- The builder pinned the partition file hash
  `fc955bcb3ef8fbdf7d5640d9a8693a8431b5f2ee291a5c1449a1549e7e073fe7`
  and preprocessing implementation hash
  `47e9f6a66b2312dc8bdd335c4e661cea856cfc0a23e77a7931b0b38df5d2b46b`.
- The final `[10000,1,32,32]` bank examined 10,001 candidates. It rejected
  one internal JPEG duplicate. It found zero training JPEG overlap, training
  tensor overlap, internal transformed-tensor collisions, or duplicate
  filenames. The tensor SHA-256 is
  `d3575a8576c0feb834a78dade48c99f81433c709ba6b99e0c5e75454992056b0`.
  The artifact SHA-256 is
  `c3c7532abde2e16c3e86368f08b2bbbbf096354a76d5daccad2e2960c9a504b7`.
- Updated `runners/checkpoint_metrics_benchmark.py` to require the official
  test split, 10,000 unique filenames, JPEG hashes, and transformed tensors.
  It now rejects training-tensor contamination before loading metric assets or
  sampling.
- Added explicit `--generated-bank` reuse. The preflight verified the existing
  10,000-image bank, step-1,000,000 checkpoint hash, full-Heun sampling
  settings, noise seed 2003, sample count, and generated tensor hash
  `7b49245859edf1d04e1a3b477329acb6a360a4118d59e5d273d8885e4c1399ea`.
- Updated the interactive wrapper, command README, image-evaluation contract,
  unified evaluation plan, and repository map. The default wrapper writes to a
  new timestamped directory and leaves `20261003T000727Z/` unchanged.
- Verification used the available `mem_param_env`; the documented local
  `memcompgen` environment was not installed. Fifteen focused sampling,
  reference, and metric tests passed. Python compilation, Ruff, shell syntax,
  `git diff --check`, the real
  reference preflight, and the real generated-bank reuse preflight passed.
- No training artifact, training job, GPU evaluation, or scheduler job was
  changed or launched. The clean-test metric command is ready but was not run.

## 2026-10-03 - Complete the paper-scale CelebA32 checkpoint gate

- Ran the N=2048, seed-0, x/x U-Net-S checkpoint at step 1,000,000 with 10,000
  deterministic generated images. Fresh 50-step full-Heun sampling took
  154.401 seconds on one A100-80GB, or 64.77 images per second.
- Retained the first validation-reference result at
  `outputs/interactive_checkpoint_metrics/20261003T000727Z/` as exploratory
  evidence. Its FID-10k was 33.7535, fixed-rho raw-pixel f_mem was 0.5197,
  calibrated raw-pixel f_mem was 0.9493, and calibrated SSCD f_mem was 0.6561.
- Replaced that reference for the paper gate because the official validation
  subset contains one exact JPEG duplicate of a selected seed-0 N=2048 train
  image. The duplicate lies outside the 1,024-image calibration prefix and
  cannot explain the large memorization rates, but the run-specific overlap is
  avoidable.
- Reused the exact generated bank against the frozen clean official-test bank.
  The final output is
  `outputs/interactive_checkpoint_metrics/20261003T084819Z/`. It reports
  FID-10k 34.8159, fixed-rho raw-pixel f_mem 0.5197, calibrated raw-pixel f_mem
  0.9450, and calibrated SSCD f_mem 0.6235.
- The clean-test reuse pass took 31.992 seconds. Combining its metric stages
  with the measured sampling kernel projects 173.191 seconds per checkpoint
  and 3.469 GPU-hours for all 72 checkpoints. The output retains feature
  matrices, float64 FID statistics, and top-two neighbor evidence, so threshold
  changes do not require resampling or re-encoding.
- This result qualifies one complete checkpoint path. It does not yet compare
  parameterizations or checkpoint trajectories.
- Real Slurm replay job `17693255` reused the same generated bank and reference.
  It reproduced all four reported metrics to displayed precision and completed with exit
  `0:0` in 64 seconds. Its output is
  `outputs/interactive_checkpoint_metrics/20261003T090549Z/`.

## 2026-10-03 - Freeze and verify the portable figure-bundle contract

- Added `runners/figure_bundle.py` with schema version 1. It verifies completed
  gate summaries, checkpoint and tensor hashes, test-reference use, full
  neighbor arrays, source counts, and unique run-step identities.
- The exporter writes portable CSV, JSON, NPZ, and PNG files. It writes full
  generated, calibration, and validation neighbor evidence. It also writes
  exact generated and nearest-training image tensors for selected examples.
  It does not load models or run sampling, encoders, neighbor search, or FID.
- Froze the SSCD alpha-0.01 qualitative rule: 16 smallest-r examples, 8 unused
  boundary-nearest examples, and 8 deterministic unused examples above the
  boundary. Final checkpoints are automatic. Explicit `RUN_ID:STEP` keys add
  disputed or planned figure checkpoints.
- Added `tests/test_figure_bundle.py`. The tests reject non-test references,
  check every table and array class, validate checksums and the tar archive,
  open the PNG, compare deterministic example indices, prove sampling is not
  called, and check overwrite protection. All four focused tests passed.
- Exported the completed replay gate to
  `outputs/figure_bundles/celeba32_checkpoint_gate_test_v1/` and its sibling
  tar archive. The manifest reports 12 scalar rows, 8 calibration rows, 306
  distribution rows, 32 exact example triplets, and 14 hashed portable files.
- The 10,000-row raw-pixel and SSCD arrays and the 1,024/8,976 reference arrays
  have the expected lengths. Every bundle hash passed. The archive is 909 KiB
  with SHA-256
  `ce50051f9396e7773a904b26ed5a1fd1a600cb957d2a7af16ff11cbc9fbccf36`.
- Updated the image-evaluation guide, unified plan, command README, and
  repository map. The docs include the required `cd` and CPU-only command.

## 2026-10-03 - Put full evaluation and the final Mac archive on ice

- Added T021 to `TODO.txt` at the researcher's request.
- The existing archive is one clean-gate checkpoint. It validates the schema
  only and is not the final all-runs paper archive.
- Do not submit full-evaluation jobs or create the final combined archive until
  the researcher explicitly resumes T021. T019 and T020 must define the run
  matrix and final figure contract first.

## 2026-10-03 - Implement and launch full metric-suite development smokes

- Added a persistent CPU-generated Gaussian evaluation-noise artifact with
  seed 2003 and shape `[10000, 1, 32, 32]`. Its manifest records generator,
  environment, tensor, artifact, and source-prefix hashes. All three selected
  runs match its first 128 rows exactly. Source monitoring noise is unchanged.
- Extended the restartable evaluator with the `full` metric preset, verified
  external noise and reference artifacts, held-out losses, raw-pixel
  memorization, DINOv2 Fréchet distance, complete MNIST diagnostics, stage
  runtimes, plot generation, strict identities, and development-only flags.
- Added strict full-suite plotting and acceptance validation. Added focused
  noise, resolution, metric, restart, cache, and two-checkpoint integration
  tests for MNIST and natural images. The second integration passes reused all
  expensive artifacts.
- Added `notebooks/sanity/full_metric_suite_smoke.ipynb` and the frozen
  three-task Berzelius launcher. The default profile evaluates all checkpoints
  with 512 generated images. The `final-replica10000` profile evaluates only
  step 1,000,000 with 10,000 generated images.
- Submitted array `17693872` for the 512-sample trajectory smoke. MNIST task 1
  completed all 20 checkpoints and both reuse passes, then found a versioned
  plot-name mismatch in the final validator. The validation-only fix and both
  hashes are recorded in the frozen batch. Restart `17693881_1` completed in
  69 seconds without resampling.
- Submitted array `17693885` for the isolated 10,000-sample final-checkpoint
  replica smoke. This is a production-scale per-checkpoint gate, not a resume
  of the final all-run evaluation or Mac archive.
- Both smoke profiles completed and passed strict acceptance. The trajectory
  outputs cover 72 U-Net, 20 MNIST, and 72 DiT checkpoints at 512 samples. The
  replica outputs cover the three step-1,000,000 checkpoints at 10,000 samples.
- Live validation found two presentation-contract defects: a version suffix in
  the MNIST memorization plot name and grid requirements for checkpoints not
  selected by the final-only profile. Both fixes affect validation only. The
  frozen batches record pre-change and post-change hashes, and task-specific
  restarts reused all expensive artifacts.
- Final audits found finite metrics, bounded f_mem values, correct sample
  shapes, readable plots, complete reuse flags, and unchanged frozen source
  inputs across all outputs.

## 2026-10-03 - Prepare the complete v/v loss-weighting paper evaluation

- Scoped the explicitly resumed evaluation to all 18 CelebA32 U-Net-S runs:
  `N={512,2048}`, seeds `{0,1,2}`, native `v/v`, and `x_equivalent`, `unit`,
  and `eps_equivalent` loss weighting.
- Added an explicit publication mode to the restartable evaluator. It requires
  the complete metric suite, 10,000 generated samples, and plot output. Its
  manifest records the paper tier and publication status.
- Strengthened full-suite acceptance. The validator now checks reuse of every
  sample, feature, statistic, distance, neighbor, and held-out-loss cache.
- Added the frozen 18-task launcher under
  `jobs/berzelius/celeba32_vv_weighting_paper_eval/`. Each task
  selects all 72 checkpoints, uses the common seed-2003 evaluation-noise bank,
  and compares against the complete clean CelebA test bank.
- Kept the final combined paper archive on ice. This request authorizes only
  the selected loss-weighting experiment.

## 2026-10-03 - Write the self-contained AISTATS experiment appendix source

- Added `docs/aistats_experimental_design_appendix.md` as a long-form source
  that later paper writing can shorten.
- Defined the linear path, target notation, common optimizer, model, sampler,
  checkpoint, data-pairing, and post-hoc evaluation contracts in plain
  technical English.
- Described all five production blocks and their 234 one-million-update
  trajectories. The source separates the main CelebA32 U-Net factorial from
  Small-DiT, MNIST, CelebA64 RGB, and native-velocity loss-weighting controls.
- Described the prediction/loss pilot, epsilon recovery, data-format,
  capacity-attention, DiT pilot, and small-set numerical studies as supporting
  blocks. It states why those runs must not be counted as independent main
  replicates.
- Marked metric smokes, the MNIST encoder job, runtime probes, resume tests,
  and notebooks as evaluation infrastructure rather than scientific blocks.
- Added analysis cautions for paired comparisons, raw loss scales, 128-image
  monitors, 512-image smoke metrics, and confounded resolution/capacity
  contrasts. Added one short reusable appendix paragraph at the end.
- Updated `docs/repo_map.md` to make the new source discoverable.

## 2026-10-03 - Audit old storage before cleanup

- Added `mem_param_src/runners/old_storage_cleanup.py`. The command is dry-run
  by default and writes separate keep, delete, and summary manifests. Its apply
  mode validates the manifest hash and each file identity before unlinking only
  explicit paths.
- Added checkpoint retention across resumed attempts. The audit keeps the first
  checkpoint, every tenth saved checkpoint, the final checkpoint, and the
  historical step-25,000 checkpoint used by the epsilon-recovery audit.
- Preserved all configurations, metrics, figures, logs, histories, data
  identities, and reproducibility artifacts. Kept W&B event files whose remote
  sync is not verified. Left current clean outputs, old evaluation tables, old
  datasets, and unrelated old source code outside the scan. The audit now saves
  the exact obsolete intrinsic/ambient prototype source in a hashed archive
  before it lists those live source files for deletion.
- Verified all active Slurm commands and working directories use
  `mem_param_clean`, not `old_mem_par`.
- Created the final dry-run audit under
  `outputs/storage_audits/old_mem_par/20261003T215201Z`. It scanned 175,520
  files, kept 106,192 files, and marked 69,328 files for possible deletion.
  Hard-link accounting predicts 2.272 TiB of recoverable storage. No deletion
  was applied.
- Loaded one retained final checkpoint from each of ten old job-set families in
  `mem_param_env`. All ten payloads were valid dictionaries.
- Verified the untouched historical evaluation tables contain 6,048, 6,984,
  and 792 checkpoint rows. Recorded their SHA-256 hashes during the check.
- The separate obsolete-source group contains 29 files and 400,772 logical
  bytes. Its 29 archive members exactly match its 29 deletion-manifest paths.
- Verification: focused unittest ran 4 tests successfully. Ruff, compilation,
  manifest/archive path equality, archive hashing, and diff checks passed in
  `mem_param_env`.

## 2026-10-04 - Apply the reviewed old-storage cleanup

- Applied all seven groups from the immutable audit at
  `outputs/storage_audits/old_mem_par/20261003T215201Z` after explicit approval.
  The tool removed 69,328 exact manifest paths from `old_mem_par`. It did not
  delete any path under `mem_param_clean`.
- Removed the cancelled `j8825` checkpoints and W&B events, verified temporary
  checkpoint copies, thinned capacity checkpoints and their hard-linked backup,
  failed or incomplete checkpoints, old image checkpoints, and prototype
  checkpoints. Retained metadata, metrics, figures, logs, configurations, and
  the scheduled checkpoints.
- Removed 29 live intrinsic/ambient prototype source files from `old_mem_par`.
  Preserved their exact tracked and untracked contents in the verified
  `INTR_AMB_SOURCE_SNAPSHOT.tar.gz` audit artifact.
- Ran `nscquota` before and after every cleanup group. Project usage decreased
  from 15.2 TiB to 13.0 TiB. This is below the 13.5 TiB stop target and the
  14.6 TiB normal quota, so deletion stopped.
- `POST_CLEANUP_VERIFICATION.json` confirms that all 106,192 keep-manifest
  regular files remain unchanged, all 69,328 delete-manifest files are absent,
  and no unexpected regular file exists in the scanned roots.

## 2026-10-04 - Add fast weighting-analysis plots

- Added `runners/weighting_trajectory_plots.py` for strict complete-matrix
  aggregation and the reproducible full PNG/PDF figure bundle.
- Added `plotting/weighting/` for fast iteration. It splits the accepted
  aggregate into one CSV per metric and provides one small command for a metric,
  quality, memorization, held-out loss, overview, or final-checkpoint figure.
- The quick path reads only the required CSV files and writes one figure. It
  does not scan checkpoints, resample, encode images, or regenerate unrelated
  plots.
- Ruff, compilation, focused aggregation tests, source and output hash checks,
  CSV completeness checks, and all six quick plot types passed.

## 2026-10-04 - Document fast plotting and qualify remaining full evaluation

- Added `docs/fast_plotting.md`. It defines the immutable aggregate, per-metric
  CSV cache, one-figure command pattern, scratch-output rule, PNG/PDF policy,
  and a three-second target for normal figure edits.
- Added `docs/full_evaluation_readiness.md`. It records every production
  block, real checkpoint counts, artifact readiness, storage limits, and the
  gates for rolling paper-evaluation batches.
- Removed the evaluator's one-channel acceptance assumption. Protocol
  manifests and strict validation now carry the saved channel count.
- Generalized the clean CelebA reference builder to the saved image size,
  channel count, and resize size while retaining its grayscale-32 defaults.
- Built verified seed-2003, 10,000-row canonical noise banks for grayscale
  64×64 and RGB 64×64. Every seed-0 source prefix of each shape matched.
- Built the clean 10,000-image RGB64 official-test reference against all 18
  saved RGB64 training banks. It examined 10,001 candidates and rejected one
  collision.
- A real CPU preflight verified encoder assets, data manifests, shape-specific
  noise, clean references, checkpoint counts, final step 1,000,000, and strict
  checkpoint loading for CelebA32 U-Net, MNIST32, MNIST64, CelebA64 RGB, and
  CelebA32 Small-DiT.
- The focused suite passed 32 tests in `mem_param_env`, including two-pass
  full-suite cache reuse for MNIST32, MNIST64, and RGB64. Ruff, compilation,
  shell syntax, notebook code-cell compilation, launcher dry-runs, and
  `git diff --check` passed.
- Project usage is 13.0 TiB of the 14.6 TiB normal quota. The accepted
  weighting evaluation uses 521 GiB. Remaining evaluation must use rolling
  batches and a verified explicit-path cache-retention command.
- No evaluation or training job was submitted, cancelled, or modified.
- Hardened both full-suite launchers to copy the exact modified torch-fidelity
  package and license into each frozen batch. The worker now imports only the
  copied package, and the batch snapshot hashes every copied file.
- Recorded the accepted weighting result's earlier torch-fidelity provenance
  limitation. Source modification times predate the run and the current tree
  hash is preserved, but a publication archive or FID recomputation must close
  the missing submission-time hash.

## 2026-10-04 - Submit all completed 1M paper evaluations

- Added `runners/full_suite_retention.py`. It creates a dry-run manifest by
  default and can remove only explicit generated sample and feature caches
  after paper-tier acceptance. It records size, time, inode, link count, and
  SHA-256 before deletion. It preserves every metric, table, plot, feature
  statistic, neighbor record, and selected qualitative sample tensor.
- Added two focused tests for complete retention and failure before any unlink
  when one candidate identity changes.
- Added the frozen launcher under `jobs/berzelius/all_1m_paper_eval/`. Its exact
  matrix contains 198 trained runs: 108 CelebA32 U-Nets, 18 MNIST32 U-Nets, 18
  MNIST64 U-Nets, 18 CelebA64 RGB U-Nets, and 36 CelebA32 Small-DiT seed-0
  runs. The 18 unfinished Small-DiT seed-1 runs remain excluded.
- The launcher hashes all 677 GiB of source run artifacts before submission.
  It also freezes the repository source, exact torch-fidelity package, encoder
  assets, noise banks, clean references, Conda environment, and task matrix.
- The execution graph uses one real MNIST32 retention gate. Later arrays use
  `afterok` dependencies and contain at most 36 tasks. This prevents concurrent
  uncompressed outputs from exceeding the project quota.
- Each accepted task runs a second cache-only pass, strict paper validation,
  dry-run retention, exact-path apply, and post-retention validation. Interrupted
  evaluator and retention states restart from the same frozen batch.
- Verification passed 35 focused tests in `mem_param_env`, including full
  two-pass MNIST32, MNIST64, and RGB64 integrations. Ruff, Python compilation,
  Bash syntax, launcher dry-run, exact 198-run matrix validation, and
  `git diff --check` passed.
- Submitted jobs `17696049` through `17696056` under project
  `berzelius-2026-269`. The frozen output root is
  `outputs/evals/all_1m_full_10000/20261004T005702Z_3271331`. At the immediate
  status check, the gate ran on `node052` and all dependent arrays waited on
  their expected dependencies.


## 2026-10-04 - Stream retention per checkpoint and replace CelebA32 evaluation

- The original full-suite worker could retain caches only after a complete
  trajectory, a second cache-only replay, and final validation. That exposed
  the project to unnecessary peak storage while large arrays ran in parallel.
- Added checkpoint-level identity manifests and restart-safe retention in
  `mem_param_src/runners/full_suite_retention.py`, integrated it into
  `image_protocol.py`, and taught strict validation and plotting to consume
  retained checkpoints. Each completed checkpoint preserves metrics, feature
  statistics, nearest-neighbor evidence, and selected qualitative samples
  before deleting dense reproducible sample and feature tensors.
- Replaced the two-pass worker with one evaluator pass followed by strict
  validation of all checkpoint-retention records. Run-level retention still
  removes reproducible reference feature matrices after paper-tier acceptance.
- Validation passed five retention tests, fourteen image-protocol integration
  tests, Ruff lint and format checks, shell syntax checks, and an exact 144-run
  launcher preflight.
- Cancelled old CelebA32 arrays `17696053-17696056`. Fourteen U-Net tasks had
  briefly started, but none reached acceptance. Submitted independent,
  unthrottled replacements `17698532-17698535` from frozen root
  `outputs/evals/celeba32_streaming_full_10000/20261004T110904Z_1979099`.
- At session closeout, 36 DiT and two U-Net evaluations were running. The first
  40 checkpoint results and retention records were complete, and startup logs
  contained no failure signature.

## 2026-10-04 - End-of-day tooling note

- Local sandbox creation repeatedly failed with `bwrap: Creating new namespace
  failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)`.
  The reported kernel limits were 2,061,050 for user, mount, and PID namespaces,
  while `lsns` reported only one visible user and mount namespace. This does not
  look like exhaustion of those host limits; it is most consistent with sandbox
  nesting or namespace-lifecycle state in the Codex execution environment.
- The issue did not affect Slurm jobs or experiment artifacts. It did make the
  interactive session slower because normal sandboxed commands and the patch
  helper failed, requiring approved direct-command retries and a narrow POSIX
  patch fallback.


## 2026-10-04 - Harden Mambaforge startup for evaluation retries

- Fifty-six independently scheduled U-Net evaluation tasks failed before
  Python started because Lmod found another loaded module in the `mamba_env`
  family. The already-running 88 replacement tasks were unaffected.
- Updated `jobs/berzelius/celeba32_streaming_paper_eval/run_array.sbatch` to
  purge loaded modules and clear stale Lmod family variables before loading
  the required Mambaforge module. The source and frozen workers otherwise
  remain identical.
- Preserved the original frozen worker and added a separately checksummed
  clean-environment worker to the existing batch. The original full snapshot
  checksum still passes.
- Submitted retry arrays `17699034` for seed-1 tasks 16-35 and `17699035` for
  all seed-2 tasks. The first two retries passed module loading, activated the
  expected environment, and entered checkpoint evaluation.


## 2026-10-04 - Recover CelebA64 finalization with frozen source

- All 18 CelebA64 first passes completed 75/75 checkpoints before their reuse
  passes failed. The worker had frozen `mem_param_src`, but it changed into the
  live repository before launching Python. The repository source therefore
  shadowed the frozen package on `sys.path`.
- The live source gained the `streaming_retention` identity field while the
  first passes ran. The old manifests correctly omit that field. The second
  processes imported the new source and rejected the old manifests.
- Added `repair_celeba64_finalization.sbatch`. It keeps the working directory
  inside the frozen batch, sets `PYTHONPATH` to the original source snapshot,
  verifies the imported module paths, and performs one reuse pass followed by
  strict validation and identity-checked retention.
- Added a read-only identity preflight. A representative CelebA64 command
  reconstructed with the frozen source matched its saved manifest exactly.
  The original batch snapshot and input hashes also passed.
- Submitted repair array `17699894` for tasks 0-17 with no dependency and no
  throttle. This repair does not modify live evaluator source, CelebA32
  outputs, training outputs, or any running job.


## 2026-10-05 - Submit seed-1 DiT streaming paper evaluation

- Added `--dit-seed1` to the existing CelebA32 streaming launcher. It selects
  exactly 18 completed seed-1 DiT runs at N=512/2048 and all prediction/loss
  pairs. Default selection remains the previous 144-run matrix.
- The task builder checks the exact 18-cell matrix and saved checkpoint
  schedule. The freezer accepts the explicitly selected 18-task count.
  The protocol remains 72 checkpoints, 10,000 samples, and 512 held-out losses.
- Fixed source isolation for new streaming batches: workers stay in their
  frozen batch, use only its source on PYTHONPATH, disable bytecode writes,
  and verify imports. Existing frozen batches and their outputs were not edited.
- Validation: shell syntax, Ruff lint/format, actual 18-run and legacy 144-run
  matrix checks, and 19 unittest retention/protocol tests passed using
  Berzelius `mem_param_env`. All 18 saved data bundles and final recovery
  states passed their recorded hashes. All evaluation inputs were checksummed.
  A frozen-source import check passed for the evaluator and metric packages.
- Submitted independent, unthrottled array `17701413`, tasks 0-17, from
  `outputs/evals/celeba32_dit_seed1_streaming_full_10000/20261005T070254Z_4059124`.
  Reusable preview, submit, and restart examples are in the launcher README.
  The final Mac archive remains on ice.

## 2026-10-05 - Add diffable plotting-data bundle

- Added `plotting/export_plot_data.py`. It exports accepted scalar training and
  evaluation records into deterministic plain CSV tables grouped by experiment
  under `plot_data/`. It excludes checkpoints, generated samples, feature
  matrices, datasets, and other large tensors.
- The current bundle contains 216 accepted evaluations across six experiment
  directories. Each directory has a run index, training metrics, evaluation
  metrics, diagnostics, and a hash manifest joined by `run_id`.
- Updated the CelebA32 U-Net 3x3 plotting script to read only the tracked
  `plot_data/celeba32_unet_3x3` tables. The default command regenerated the
  four-N_train, three-seed f_mem grid from 2,592 aggregate rows in 1.84 seconds.
- Ruff, Python compilation, exporter execution, plotting execution, and Git
  whitespace checks passed. Two consecutive exports of all 38 generated files
  were byte-identical. The still-running DiT seed-1 evaluation is intentionally
  absent until those runs reach accepted state and the exporter is rerun.


## 2026-10-05 - Reproduce the portable CelebA32 U-Net figure locally

- Switched directly to `codex/portable-plot-data` at `6341680` and completed
  the requested fast-forward-only pull. No merge or cherry-pick was performed.
- Read the local plotting handoff, bundle index, and experiment records.
  All 24 exported CSVs matched their manifest SHA-256 hashes and byte sizes.
- In local `memcompgen` (Python 3.10, Matplotlib 3.10.8), ran
  `MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl conda run --no-capture-output -n memcompgen python plotting/celeba32_3x3/plot_f_mem_grid.py`.
  Matplotlib was already installed. Numerical inputs came only from `plot_data/`.
- Produced the PNG and compact CSV under
  `outputs/figures/celeba32_unet_prediction_loss_3x3/`, with basename
  `all_n_train__fmem_raw_pixels__rho_0p3__seed_mean_sd`.
  The table has 2,592 rows: four training sizes, nine target pairs, and
  72 checkpoints. Each row preserves three seeds, their mean, and sample SD.
- Visually checked the nine panels, labels, legend, and uncertainty bands.
  Confirmed both outputs are Git-ignored and tracked files were unchanged
  before this journal entry. No training, evaluation, or cluster operation ran.


## 2026-10-05 - Overlay FID on the portable CelebA32 U-Net grid

- Extended `plotting/celeba32_3x3/plot_f_mem_grid.py` to show Inception FID
  as dotted curves on the left axis and raw-pixel `f_mem` at rho=0.3 as
  solid curves on the right. Both metrics retain the training-size colors,
  three-seed means, and sample-SD bands. Each metric uses a common linear
  scale across all nine panels.
- Read only `plot_data/celeba32_unet_3x3/runs.csv` and
  `evaluation_metrics.csv` for plotting inputs. Added an explicit metric
  column to the scratch table; FID rows have no rho threshold.
- Updated the local plotting handoff. The existing command now writes the
  combined PNG and CSV with basename
  `all_n_train__fid__fmem_raw_pixels__rho_0p3__seed_mean_sd` under
  `outputs/figures/celeba32_unet_prediction_loss_3x3/`.
- Runtime validation in local `memcompgen` produced 5,184 aggregate rows
  in 1.69 seconds. All 2,592 prior f_mem rows were preserved exactly.
  All 7,776 FID seed values matched the exported run/checkpoint records;
  their means and sample SDs passed direct checks. Visually inspected axes,
  line styles, colors, bands, and legends. Ruff and whitespace checks passed.
- All 24 exported CSV hashes and sizes remained unchanged. No metrics,
  cluster provenance, checkpoints, or evaluation records were modified.


## 2026-10-05 - Summarize memorization at each seed's best FID

- The researcher selected the lowest-FID checkpoint separately for each seed,
  followed by the mean and sample SD of the paired f_mem values across seeds.
  Used all 72 exported checkpoints and raw-pixel f_mem at rho=0.3. Exact FID
  ties select the earlier step. Different conditions can reach different FIDs.
- Added `plotting/celeba32_3x3/summarize_f_mem_at_best_fid.py`, reusing the
  existing plot-data readers. The script reads only the committed CelebA32
  U-Net tables and retains source run IDs and selected steps for every seed.
- Ran `MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl conda run --no-capture-output -n memcompgen python plotting/celeba32_3x3/summarize_f_mem_at_best_fid.py`.
  Wrote `fmem_at_best_fid__rho_0p3__per_seed.csv` (108 rows) and
  `fmem_at_best_fid__rho_0p3__mean_sd.csv` (36 rows) under
  `outputs/figures/celeba32_unet_prediction_loss_3x3/`. The summary contains
  mean and sample SD for f_mem and the selected FIDs. Updated the local
  plotting handoff and repository map with the command and selection rule.
- Validation in local `memcompgen`: independently sorted source FID records
  to verify all 108 selected minima, checked same-checkpoint f_mem values,
  and recomputed all 36 means and sample SDs with denominator two. Selected
  steps span 2,750 to 1,000,000; selected FIDs span 6.6373 to 70.4362.
  All 24 exported CSV hashes and sizes remained unchanged. Ruff and Git
  whitespace checks passed. No training or evaluation was rerun.


## 2026-10-05 - Audit similar FID trajectories across training sizes

- The researcher questioned the similar FID paths across training sizes in
  the combined grid. Independently joined the exported FID rows to runs.csv
  and checked all 2,592 plotted FID rows, including all three seed values.
  Every plotted value and seed mean matched the corresponding source record.
- All 108 complete FID trajectories are distinct. The run index contains
  108 distinct result paths and config hashes, plus 12 data-bundle hashes
  for four training sizes and three seeds shared across the nine target cells.
  For each target pair and seed, N=512 and N=4096 have equal FID only at
  step zero; all 71 later checkpoint values differ.
- The broad trajectory similarity is present in the export. The combined
  plot's linear FID range, approximately 0-459, compresses late differences.
  At one million updates, mean v/v FID is 14.264, 7.497, 8.168, and 13.584
  for N=512,1024,2048,4096. Mean eps/x FID is 46.911, 53.110, 87.699, 96.619.
- Created and visually inspected `fid_trajectory_audit__log_scale.png` from
  plot_data/ using a local memcompgen Python/Matplotlib CSV analysis. It shows
  separate seed curves and their means on a common logarithmic FID scale.
  Saved all 2,592 plotted records and 36 final-checkpoint summaries as CSVs
  with basenames `fid_trajectory_audit__log_scale` and
  `fid_trajectory_audit__step_1000000` in the existing figure directory.
- Inspected exporter source: each run reads its own accepted result's
  combined/checkpoint_metrics.csv. This checks the export-to-plot mapping;
  the portable bundle cannot establish that upstream checkpoint loading,
  generated features, reference statistics, or cache identities were correct.
  The cluster evaluation artifacts were not accessed. No causal claim was made.
- All 24 exported CSV hashes and byte sizes remain unchanged. No plotting
  implementation, metric, evaluation record, or original figure was changed.


## 2026-10-05 - Display best FID beside the memorization summary

- Updated the summary command and local plotting handoff to display each
  cell as `f_mem mean ± SD (best FID mean ± SD)`. Both metrics use each
  seed's previously selected lowest-FID checkpoint.
- Reran the summary command in local memcompgen. All 36 cells include FID
  to two decimal places. Both numerical CSVs remain byte-identical by SHA-256;
  this change affects display only. Git whitespace checks passed.


## 2026-10-05 - Render the best-FID summary as a 3x3 grid

- Added `plotting/celeba32_3x3/plot_f_mem_at_best_fid.py`. It calls the existing
  summary function using only plot_data/ numerical inputs. Prediction targets
  define rows and loss targets define columns. Each cell stacks the four
  training sizes and retains their existing colors.
- Each entry displays `f_mem mean ± SD (best FID mean ± SD)`, retaining
  per-seed best-FID checkpoint selection, raw pixels, rho=0.3, and sample SD.
  Wrote PNG, vector SVG, and standalone LaTeX with basename
  `fmem_at_best_fid__rho_0p3__grid` in the existing figure directory. Compiled
  the LaTeX to a one-page PDF using pdfTeX from TeX Live 2026.
- Ran the Python renderer in local memcompgen, checked its PNG, and inspected
  the PDF rendered by pdftoppm. No clipped text or overlapping cells were found.
  LaTeX compiled without overflow or LaTeX warnings. Ruff passed.
- Verified that all 108 seed selections and 36 summaries match the existing
  CSVs exactly. All 36 formatted entries appear in both SVG and LaTeX.
  All 24 source CSV hashes and sizes remained unchanged. Updated the handoff
  and repository map with the renderer and reproducible commands.


## 2026-10-05 - Compare diagonal targets and equivalent velocity weights

- Audited the requested comparison between x/x, eps/eps, v/v, and native
  v/v with x_equivalent and eps_equivalent weights. The portable bundle has
  all five conditions at N=512 and N=2048 with seeds 0, 1, and 2. Each run
  has 72 accepted checkpoints through one million updates and 10,000
  generated samples per evaluation.
- Matching N/seed records share the saved data-bundle SHA-256. Exported
  architecture, learning rate, batch size, training length, and evaluation
  coverage match. Compared declared data, network, training, sampling,
  runtime, and evaluation assignments in the matrix and weighting runners;
  only the intended training-size assignment differs between size blocks.
  Complete frozen evaluation manifests are absent from the portable bundle;
  upstream feature caches and checkpoint loading were not independently audited.
- All six unit-weight v/v repeats exactly match matrix v/v training losses,
  monitoring records, FID, and raw-pixel f_mem. Retained one v/v baseline per
  seed, yielding 30 runs rather than counting repeats as independent seeds.
- Verified the implemented raw weights `(1-t)^2` and `t^2`. They equal x
  and eps MSE for predictions implied by a velocity output, corresponding
  to the matrix's v/x and v/eps objectives. They do not assert identical
  networks or training dynamics for direct x and eps outputs. Weighted runs
  are close to, but not bitwise identical to, the corresponding off-diagonal
  runs: maximum absolute FID/f_mem differences are 1.39124/0.00690001 for
  x_equivalent and 0.333299/0.00610000 for eps_equivalent across matched
  sizes, seeds, and checkpoints. No cause for these differences was established.
- Added `plotting/celeba32_3x3/compare_weighted_velocity.py`. Numerical inputs
  come only from plot_data/. It checks matching records, excludes repeated
  unit controls, and plots three-seed means with sample-SD bands. FID uses
  a log scale; f_mem uses raw pixels and rho=0.3. Separate columns show the
  two sizes. Dotted curves identify equivalent velocity weights.
- Ran `MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl conda run --no-capture-output -n memcompgen python plotting/celeba32_3x3/compare_weighted_velocity.py`.
  Outputs under `outputs/figures/celeba32_weighting_comparison/` include PNG,
  SVG, 2,160 paired checkpoint records, 30 per-seed best-FID records, and
  ten mean/sample-SD summaries. Best FID is selected separately per seed;
  exact ties select the earlier step. Both metrics use the same checkpoint.
- Runtime validation in memcompgen independently matched all 4,320 plotted
  metric values to their source records, verified all 30 minima and ten
  summaries, and confirmed that diagonal summaries match the prior table.
  Visually checked the PNG layout. Ruff passed. All 24 source CSV hashes and
  sizes remained unchanged. Updated the local handoff and repository map.
  No training, evaluation, commit, push, merge, or cherry-pick was performed.


## 2026-10-05 - Plot quality trajectories by training size for the other families

- Added `plotting/plot_fid_by_training_size.py` for CelebA32 Small-DiT,
  MNIST32, MNIST64, and CelebA64 RGB. It joins run conditions to accepted
  quality records from plot_data/ and checks size, target, seed, model,
  image-format, sample-count, and checkpoint coverage before plotting.
- DiT has 36 seed-zero runs: four training sizes and the full 3x3 matrix,
  each with 72 checkpoints. MNIST32 and MNIST64 each have 18 diagonal runs
  at N=512/2048, three seeds, and 20 checkpoints. CelebA64 RGB has the same
  18-run design with 75 checkpoints. All use 10,000 generated samples and
  reach one million training steps.
- Used Inception FID for CelebA. MNIST exports contain the primary JointAE
  FD metric `mnist_fd_jointae32_v1`, not Inception FID, so the labels and
  filenames name that representation. CelebA64 retains the legacy dataset
  key celeba32_grey; the image-size/channel fields confirm 64x64 RGB inputs.
- Figures overlay training sizes using the existing colors and share a
  logarithmic quality axis within each figure. DiT is a 3x3 grid with rows
  for prediction and columns for loss. Other figures have three diagonal
  panels. Three-seed curves show arithmetic means, sample SD, and faint
  seed trajectories. DiT shows seed 0 only, without an uncertainty band;
  its summary CSV leaves SD blank. No smoothing or checkpoint resampling ran.
- Executed `MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl conda run --no-capture-output -n memcompgen python plotting/plot_fid_by_training_size.py`.
  Runtime was 2.44 seconds. Each experiment's outputs/figures/ directory
  contains PNG, SVG, individual-seed CSV, and mean/sample-SD CSV with basename
  `all_n_train__<metric>__log_scale`.
- Validation in local memcompgen: all 4,662 seed values across 90 runs
  matched the source records exactly. Independently checked 3,282 summary
  means and sample SDs, blank single-seed SDs, metric labels, and seed labels.
  Visually inspected all four PNGs; no overlapping or clipped labels were
  found. Ruff passed. All 24 exported CSV hashes and byte sizes remained
  unchanged. Updated the handoff and repository map. No training or
  evaluation ran, and no source metric or provenance record was modified.


## 2026-10-05 - Unify prediction-target colors across portable figures

- Applied the researcher's convention: eps/eps is blue (#1F77B4), x/x is
  red (#D62728), and v/v is green (#2CA02C). Color follows the prediction
  target in off-diagonal cells as well. All native v/v weighting controls
  remain green, using solid/unit, dashed/x_equivalent, and dotted/eps_equivalent.
- Centralized TARGET_COLORS, N_TRAIN_LINESTYLES, and N_TRAIN_MARKERS in
  `plotting/celeba32_3x3/plot_f_mem_grid.py`. The other portable plotting
  entrypoints import these definitions. Single-metric figures distinguish
  training sizes by line style. The dual-axis grid preserves dotted FID and
  solid f_mem, with circle/square/triangle/diamond markers for increasing N.
- Updated the best-FID grid and LaTeX to color values by prediction target;
  training-size labels are neutral. Updated the local handoff with the
  convention, exact colors, size encodings, and shared definition location.
  Updated the repository map. The rule supersedes the earlier size palette.
- Reran all four affected entrypoints in local memcompgen with the temporary
  MPLCONFIGDIR: plot_f_mem_grid.py, plot_f_mem_at_best_fid.py,
  compare_weighted_velocity.py, and plot_fid_by_training_size.py. Re-rendered
  the earlier U-Net FID audit through the shared quality-trajectory renderer.
  Recompiled the standalone LaTeX PDF with pdfTeX from TeX Live 2026.
- Visually inspected all eight updated PNG figures and the PDF rendered by
  pdftoppm. Adjusted caption spacing in the dual-axis figure to avoid overlap.
  Checked the trajectory SVG palettes. Ruff and Git whitespace checks passed.
  All 41 pre-existing source and derived CSVs remained byte-identical by
  SHA-256. No metrics, evaluation records, or provenance were changed.


## 2026-10-05 - Distinguish two training sizes with shades and markers

- Refined the shared palette as requested. In two-size comparisons, the
  smaller N uses a lighter shade and the larger N uses the canonical darker
  color. N=512 uses circles; N=2048 uses triangles. Single-size figures keep
  the canonical target color. Four-size figures retain their line-style rule.
- Added `target_color` beside the existing shared palette. Lighter shades
  mix 35% white with the canonical target color, giving red #E47373, blue
  #6DA7CE, and green #76C176. The target meaning remains x red, eps blue,
  and v green. The dual-axis plot also uses this rule when selecting two sizes.
- Regenerated MNIST32, MNIST64, CelebA64 RGB, and the weighted-velocity
  comparison through the two existing entrypoints in local memcompgen.
  The diagonal size overlays use solid lines with distinct shades/markers.
  Weighted velocity controls retain their solid/dashed/dotted weight encoding.
  DiT's four-size figure was regenerated without a style change.
- Visually inspected all four changed PNGs and their legends. Checked
  single-size, two-size, and four-size color selection; Ruff and Git whitespace
  checks passed. All 41 source and derived CSVs remained byte-identical by
  SHA-256. Updated the shared convention and repository map. No numerical
  values, metric definitions, or evaluation records changed.


## 2026-10-05 - Distinguish velocity weights and export vector PDFs

- The researcher requested different colors for the three native velocity
  variants in the CelebA32 weighting comparison. Kept unit v/v green and
  assigned x_equivalent purple (#9467BD) and eps_equivalent orange (#FF7F0E).
  The diagonal x/x and eps/eps controls remain red and blue. Preserved
  lighter/circle N=512 and darker/triangle N=2048, plus the weight line styles.
  Updated legends and documented this extension to the canonical palette.
- Extracted the existing shade calculation into `size_color`, reused by
  target_color and the weighting renderer. No numerical selection or
  aggregation changed.
- The researcher also requested PDF instead of PNG. Changed the four current
  plotting entrypoints to default to vector PDF; the existing SVG exports
  remain. Matplotlib embeds TrueType fonts with pdf.fonttype=42. The dual-axis
  entrypoint still accepts an explicit output suffix. Retained figure sizes.
- The Python best-FID table now writes `fmem_at_best_fid__rho_0p3__grid.pdf`.
  Its standalone LaTeX source and compiled alternative use the `__latex`
  suffix to keep the two PDFs separate. Updated the handoff and repository map.
- Reran plot_f_mem_grid.py, plot_f_mem_at_best_fid.py,
  compare_weighted_velocity.py, and plot_fid_by_training_size.py in local
  memcompgen with MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl. Exported
  the prior U-Net FID audit through the shared trajectory renderer and
  compiled the alternative table with pdfTeX from TeX Live 2026.
- Nine PDFs are available: eight Matplotlib figures and the LaTeX table.
  pdfinfo confirmed one page per artifact; pdffonts confirmed embedded fonts
  without Type 3 fonts; pdfimages found no embedded raster images. Rendered
  all nine with pdftoppm and visually checked colors, labels, and layout.
  All 41 source and derived CSVs remain byte-identical by SHA-256. Ruff and
  Git whitespace checks passed. No training or evaluation was performed.


## 2026-10-05 - Set all figure panels to three inches wide

- The researcher clarified that "size 3" means three inches wide per
  subplot, with height allowed to vary. Added PANEL_WIDTH_INCHES=3.0 and
  set_panel_width beside the shared plotting style. All four rendering
  entrypoints use the helper after defining subplot spacing. It adjusts
  canvas width using the axes position and preserves the chosen height.
- Reflowed the weighting title and legend for its narrower canvas. Kept
  panel heights between 1.76 and 2.91 inches according to the figure family.
  The best-FID table now places FID in parentheses below f_mem in every
  entry; increased its panel height to about 2.42 inches to retain legibility.
  The LaTeX alternative uses nine 3-inch-wide, 2.4-inch-high minipages.
- Regenerated all eight Matplotlib PDFs/SVGs where supported in local
  memcompgen, including the prior U-Net FID audit. Executed the four
  entrypoints through an instrumented runtime that measured each axes box
  before closing the figure. All axes, including the dual-axis twins, were
  three inches wide to numerical precision. Figure-level text fit the canvas.
- Recompiled the LaTeX alternative with pdfTeX from TeX Live 2026. All nine
  PDFs contain one page and embedded fonts without Type 3 fonts. Rendered
  all nine using pdftoppm and visually checked the panel layouts, legends,
  and wrapped table entries. The LaTeX build reported no overflow warnings.
  All 41 source and derived CSVs remained byte-identical by SHA-256.
- Updated the plotting handoff and repository map with the shared width,
  variable heights, and table layout. Ruff and Git whitespace checks passed.
  Color conventions, selected checkpoints, means, and sample SDs are unchanged.


## 2026-10-05 - Two-inch panels and a stable figure catalog

- The researcher reduced the requested subplot width from three to two
  inches and requested short codes for figures and reusable figure types.
  Set PANEL_WIDTH_INCHES=2.0. Both Matplotlib and LaTeX now use that constant.
  Heights vary by template. Adjusted common typography, matrix margins,
  legends, and two-line table headers to fit the narrower panels.
- Added `plotting/figures.py` with stable IDs F01-F09 and type codes QM
  (quality matrix), QD (quality diagonal), DM (dual metric), WV (velocity
  weighting), and BT (best-FID table). It calls the existing renderers and
  supports individual IDs, several IDs, one or more types, --all, and --list.
  F01 makes the previously one-off U-Net FID audit directly reproducible.
- Each catalog build retains the descriptive filename and writes a
  byte-identical short PDF copy such as `outputs/figures/F07.pdf`. The output
  INDEX.md links the available PDFs. The catalog never writes numerical CSVs.
  F09 writes the LaTeX source and compiles its PDF with temporary build files.
  Updated the local plotting handoff with the ID/type map, template owners,
  shared settings, and concrete commands. Updated the repository map.
- Executed `figures.main(['--all'])` in local memcompgen with
  MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl and instrumented figure
  closure. All eight Matplotlib figures, including twin axes, have axes
  boxes exactly two inches wide. Titles, legends, axis labels, and table
  text fit the canvas; matrix prediction labels clear the overall metric
  label. The LaTeX source contains nine two-inch minipages and compiles
  without overflow warnings using pdfTeX from TeX Live 2026.
- All nine PDFs have one page, embedded fonts without Type 3 fonts, and no
  raster images. Rendered every final PDF with pdftoppm and visually checked
  labels, legends, table precision, colors, and spacing. Confirmed all short
  PDF copies match their descriptive counterparts byte for byte. All 41
  pre-existing source and derived CSV hashes remain unchanged.
- Checked argument selection with mocked rendering: individual IDs,
  deduplication, mixed type groups, case normalization, and read-only listing.
  Also ran the real `python plotting/figures.py F07` command in memcompgen
  and confirmed that only F07 changed among the nine catalog PDFs. Ruff and
  Git whitespace checks passed. Work remains on codex/portable-plot-data;
  main is unchanged. No training or evaluation ran.


## 2026-10-05 - Copy the plotting data into a portable agent handoff

- The researcher requested a new folder to give to agents elsewhere. Created
  `outputs/handoffs/mem_param_plotting_handoff_2026-10-05/` and its matching
  ZIP. Preserved relative paths for all 24 source CSVs and 17 derived CSVs,
  six experiment manifests and source descriptions, nine current figure
  designs with short and descriptive PDF names, current SVGs, and F09 LaTeX.
  Included the six local plotting scripts and two relevant documentation files.
- Added a standalone README, bundle-specific AGENTS.md, a Matplotlib 3.10.8
  requirement, SOURCE_STATE.json, and CHECKSUMS.sha256. The source-state file
  records codex/portable-plot-data, the full source commit, the local Python
  version, copied paths, and the presence of uncommitted plotting changes.
  The guide explains figure IDs, metric boundaries, seed aggregation, colors,
  setup commands, and how to verify the transferred files. It records that
  tau_gen and tau_mem definitions were not finalized in this session.
- The bundle contains 95 files, totaling 172,075,519 bytes. The ZIP contains
  21,820,658 bytes. Its SHA-256 is
  `89008f00566142119bc93f43b277f588cdd2eab096ba9e568cdf38c9d196079d`.
  Current plotting inputs, source provenance, and original outputs are unchanged.
- Verified all copied files byte for byte against the originals, checked the
  source manifest hashes and byte counts, and tested ZIP integrity. Extracted
  the archive into an independent temporary directory and executed
  `python plotting/figures.py --all` there with PYTHONPATH removed, using
  local memcompgen Python 3.10.19, Matplotlib 3.10.8, and installed pdflatex.
  All nine figures rebuilt successfully, including the LaTeX alternative;
  all 41 extracted CSV hashes remained unchanged. This checks a transferred
  bundle in the existing local environment, not a fresh dependency installation.
- Updated the repository map. The handoff and ZIP are ignored generated
  outputs. No training, evaluation, Git commit, or push was performed.


## 2026-10-06 - Audit the proposed paper figure plan and its evaluation

- Read the two supplied texts and independently recomputed their quoted
  numerical claims from all 216 portable runs. Saved the audit script,
  per-seed and summary CSVs, onset sensitivities, quality jumps, calibration
  rates, weighting comparisons, and review in
  `outputs/figure_plan_sanity_2026-10-06/`. All 24 source CSV hashes remain
  unchanged. Existing figures and the shared handoff archive were not changed.
- Reproduced the evaluation's DINOv2 counterexamples, SSCD onset table,
  FID-jump corrections, best-quality values/times, grid example, weighting
  example, and calibration-rate range. The audit adds a within-SSCD x/v
  ranking reversal across N and quantifies onset-cutoff sensitivity. The eps/eps
  point estimate is lowest in all 14 exported settings under the proposed
  primary representations, with no claim of pairwise significance.
- Verified that unit-v repeated controls have distinct source paths/config
  hashes and identical exported FID/SSCD trajectories. They remain three seeds.
  The explicit weighted-v trajectories closely match the grid's v/x and v/eps
  trajectories. Recorded a descriptive row/column/interaction decomposition;
  it is not a significance test or a causal conclusion.
- Flagged SSCD versus raw-pixel rho=0.3 and SE versus SD as convention changes.
  Checked the local conversion formulas, exporter calibration fields, original
  SSCD paper, and official AISTATS template dimensions. Newly named plotting
  scripts and original theory figures are absent, so their implementations
  and completion claims remain unverified. Ran the audit in local memcompgen
  and passed Ruff. No training, evaluation, commit, or push was performed.


## 2026-10-06 - Recreate the paired diagonal trajectory figure as F10

- The researcher supplied a historical 2x3 figure and requested a new version
  from current records, retaining its coloring. They explicitly selected
  raw-pixel f_mem at rho=0.3. Added F10 and the DT paired-diagonal template in
  `plotting/plot_diagonal_trajectories.py`, registered in the figure catalog.
- Reused the existing strict quality loader and seed-summary function. Joined
  raw-pixel memorization to the same 36 diagonal CelebA32 U-Net runs, checking
  sample counts, finite values, duplicates, and matching checkpoint schedules.
  Source inputs are only runs.csv and evaluation_metrics.csv under plot_data.
- Matched the reference's eps/eps, v/v, x/x column order; canonical blue,
  green, red curves; increasingly lighter shades at larger N; and pale blue
  FID / warm memorization backgrounds. The upper row uses log FID; the lower
  row shows f_mem. Both use log tau=step/1e6. Four styles preserve the current
  N=512/1024/2048/4096 mapping. The historical N=8192 is absent from the export.
- Lines show three-seed means and bands show sample SD. All 71 positive
  checkpoints are shown without smoothing. Step zero remains in the saved
  CSVs, marked as not displayed on the log axis. Saved PDF/SVG, a scientific
  caption, 5,184 per-seed values, and 1,728 mean/SD rows. F10.pdf is an exact
  short copy of the descriptive PDF. The canvas is 7.111 by 3.65 inches;
  each axes box is two inches wide and about 1.05 inches high.
- Executed the direct renderer and an instrumented F10 catalog build in local
  memcompgen with MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl. Verified
  every seed value and summary mean/SD against the existing F06 table exactly.
  Checked all six axes, 24 curves, 71 positive points per curve, physical
  dimensions, and text/legend bounds. Catalog builds preserve the new CSVs.
  All prior source and figure-file hashes are unchanged except the updated
  figure index. pdftoppm visual inspection passed. pdfinfo/pdffonts/pdfimages
  confirm a one-page vector PDF with embedded TrueType fonts and no raster
  images. Ruff and Git whitespace checks passed.
- Updated the plotting handoff and repository map with the new ID, template,
  style exception, commands, and artifact roles. Work remains on
  codex/portable-plot-data; main is unchanged. No training or evaluation ran.


## 2026-10-06 - Add the summary row as F11

- The researcher requested the bottom summary row from the supplied backbone
  figure. Added F11, type DS, with F10's two trajectory rows above best FID,
  raw-pixel f_mem at best FID, and memorization onset versus training size.
  Retained raw pixels at rho=0.3, canonical target colors, four training sizes,
  three seeds, and sample SD. The best checkpoint is selected within each seed;
  exact FID ties use the earlier checkpoint. Reference SSCD values are not used.
- Extracted F10's trajectory drawing into a shared function. The new
  `plotting/plot_diagonal_summary.py` reads the same portable records, computes
  per-seed summaries, and renders the new figure. Registered F11 and DS in the
  catalog. Saved separate PDF/SVG, caption, 36 per-seed records, and 12 condition
  summaries. F10 remains unchanged as a compact option.
- Asked the researcher about onset and used a stated provisional definition
  for the draft: first saved checkpoint with f_mem>=0.05 at every remaining
  saved checkpoint through one million steps. This cutoff is distinct from
  rho=0.3 and has no calibrated false-positive interpretation. All three
  seeds are censored for eps/eps at N=2048/4096 and v/v at N=4096. Open markers
  and arrows show those limits. No censored times enter onset means or SD.
- Executed the direct renderer and F11 catalog build in local memcompgen
  with MPLCONFIGDIR=/private/tmp/mem_param_plotting_mpl. All 36 best-checkpoint
  records and 12 best-FID/f_mem means and sample SD match the prior tables
  exactly. Independently checked onset with a forward search over all future
  saved values; confirmed nine censored runs and blank corresponding summaries.
- Inspected the rendered PDF with pdftoppm. All nine axes are exactly two
  inches wide; the canvas is 7.6066 by 5.8 inches. Checked 71 positive points
  per trajectory and text bounds. pdfinfo/pdffonts/pdfimages confirm one page,
  embedded TrueType fonts, and no raster images. F10 rendered through the
  extracted drawing function is pixel-identical to its previous PDF rendering.
- Catalog rendering preserves all numerical CSVs. All pre-existing source
  and figure hashes remain unchanged except the figure index. Ruff and Git
  whitespace checks passed. Updated the handoff and repository map. Work stays
  on codex/portable-plot-data; main is unchanged. No training, evaluation,
  commit, or push was performed.


## 2026-10-06 - Clarify the timescales and widen F11

- The researcher recalled existing tau_gen/tau_mem definitions and requested
  a wider figure. Re-read the supplied figure plan and its evaluation.
  The plan defines tau_gen as the first saved FID within 10% of the run's
  minimum, and tau_mem as the first sustained f_mem>=0.05 checkpoint.
  F11 already implements that sustained-crossing rule. Corrected the earlier
  description of it as a newly proposed definition. The distinction is the
  metric: the plan used calibrated SSCD at alpha=0.01, while this figure uses
  the explicitly requested raw pixels at rho=0.3. Kept the existing numerical
  cutoff, raw-pixel inputs, and censoring. tau_gen is not plotted in F11 and
  need not coincide with the actual best-FID checkpoint.
- Widened all nine F11 axes from 2 to 2.5 inches, holding height fixed.
  The canvas is now 9.5082 by 5.8 inches. The shared width helper accepts an
  explicit width and retains its two-inch default. F11 owns its width in
  plotting/plot_diagonal_summary.py. Updated its caption, footer, catalog
  note, plotting handoff, and repository map.
- Executed the F11 catalog build in local memcompgen and inspected its PDF
  rendering with pdftoppm. Verified all nine axes widths and the unchanged
  height, checked the shared helper's default, and confirmed the short PDF
  matches the descriptive PDF. All source files, numerical CSVs, and F01-F10
  artifacts retain their hashes. Ruff and Git whitespace checks passed.
  Work remains on codex/portable-plot-data; main is unchanged. No training,
  evaluation, commit, or push was performed.


## 2026-10-06 - Moderate the F11 size reduction

- The researcher found the 6.75-inch reduction too strong. Increased the
  export width to 8.3 inches, roughly halfway back to the original 9.9239.
  Preserved the aspect ratio, giving a 4.09818-inch height. Kept the revised
  panel labels, typography, colors, and layout settings. Updated the catalog
  and dimension notes; at 6.75-inch manuscript width, eight-point export
  labels scale to about 6.5 points.
- Rebuilt F11 in local memcompgen and inspected its pdftoppm render. Checked
  dimensions, aspect ratio, matching short/descriptive PDFs, and unchanged
  source-data, numerical-CSV, and other-figure hashes. Git whitespace check
  passed. Work remains on codex/portable-plot-data; main is unchanged.


## 2026-10-06 - Tighten F11 and move its visual keys into the caption

- The researcher requested a tighter, wider layout with the N and parameter
  indications in the caption. Increased F11's axes width from 2.5 to 2.75
  inches, reduced horizontal gutters, and shortened the canvas from 5.8 to
  4.9 inches. Its full width is now 9.9239 inches.
- Removed the separate top metadata header and N legend. Grouped the four
  N line-style keys with the three parameter color/marker keys in one row
  inside the caption area below the panels. Dataset, metric, sample-SD
  convention, and onset definition also remain visible there. Retained
  panel headings, all numerical values, and the gaps at censored onsets.
- Adjusted row spacing after inspecting the rendered labels. Extended the
  display arrows above the unchanged one-million-step censoring markers so
  the arrows remain legible at the shorter panel height. Updated the full
  caption, plotting handoff, and repository map.
- Executed the F11 catalog renderer in local memcompgen. Checked the nine
  2.75-inch axes, 4.9-inch canvas height, seven caption keys, and clearance
  between row labels, titles, and caption elements. Inspected the final PDF
  through pdftoppm. All source and numerical CSV hashes are unchanged;
  F01-F10 artifacts are unchanged. The short and descriptive F11 PDFs match.
  Ruff and Git whitespace checks passed. Work remains on codex/portable-plot-data;
  main is unchanged. No training, evaluation, commit, or push was performed.


## 2026-10-06 - Move F11's title to the bottom

- Moved the dataset/metric/uncertainty title below the caption legend and
  onset note, as requested. Plot panels, dimensions, and numerical values
  are unchanged. Rebuilt F11 in local memcompgen and visually checked the
  PDF with pdftoppm; all three caption rows fit without overlap.


## 2026-10-06 - Separate the timescales from the main trajectory figure

- The researcher requested moving F11's onset panel and the generation
  window into a separate figure, with the two best-FID panels sharing the
  main figure's bottom row. F11 now has a 3+3+2 layout: six 2.75-inch
  trajectory panels and two 4.3476-inch summary panels on its existing
  9.9239-by-4.9-inch canvas. Removed the onset-specific threshold lines and
  caption note from F11. Preserved its title at the bottom and its CSV files.
- Added F12 and the TS template in plotting/plot_diagonal_timescales.py.
  Its three panels show tau_gen, tau_mem, and delta_tau=tau_mem-tau_gen.
  tau_gen is the first saved FID within 10% of each seed's minimum;
  tau_mem reuses the existing sustained 5% raw-pixel onset at rho=0.3.
  The window is computed within each seed before mean and sample SD.
  Negative values remain visible with a zero line. These are signed milestone
  differences; the quality criterion does not require remaining in the band.
- Nine seeds remain censored. Their onset/window estimates and SD are blank.
  For each censored seed, tau_mem>T and delta_tau>T-tau_gen. F12 explicitly
  shows the average of those lower bounds as a bound on the seed mean, using
  open markers and upward arrows without SD or connecting lines. Among the
  27 observed seed gaps, 25 are negative and two positive.
- Saved new F12 PDF/SVG, caption, 36 per-seed records, and 12 condition
  summaries. Both timescale CSVs use raw training steps; the figure divides
  all y values by one million. F12 has three 2.75-inch panels on a
  9.9239-by-3.1-inch canvas. Catalog builds preserve all numerical CSVs.
- Ran the direct F12 renderer and the F11/F12 catalog commands in local
  memcompgen. Independently checked all 36 first-quality crossings, sustained
  onsets, paired differences, sample SD, and lower bounds against saved values.
  Onsets match the previous F11 per-seed table. Verified panel counts/widths,
  caption and label clearances, and matching short/descriptive PDFs. Inspected
  both PDFs using pdftoppm. F12 has embedded TrueType fonts and no raster images.
- All pre-existing source and numerical CSV hashes are unchanged. F01-F10
  artifacts are unchanged. Updated the catalog, plotting handoff, and repo map.
  Ruff and Git whitespace checks passed. Work remains on codex/portable-plot-data;
  main is unchanged. No training, evaluation, commit, or push was performed.


## 2026-10-06 - Use bold math parameter labels in F11 and F12

- The researcher requested LaTeX math for epsilon, x, and v, preferably bold.
  Added shared bold parameter-pair labels using MathText boldsymbol and the
  serif STIX math font. Applied them to F11's column headings and the F11/F12
  legends. Preserved the canonical colors, layout, and numerical inputs.
- Rebuilt both catalog PDFs in local memcompgen and visually inspected their
  pdftoppm renders. The labels render correctly without overlap. Updated the
  plotting handoff with their style owner. No numerical CSVs were rewritten.


## 2026-10-06 - Stack F12 and reduce empty space

- Changed F12 from a wide 1x3 strip to a compact 3x1 stack at the researcher's
  request. The canvas is 3.2484 by 4.8 inches, with 2.55-inch-wide axes,
  one shared N axis, and one units label. Observed/legend markers increased
  from four to five points; censored markers increased from 4.5 to 5.5 points.
  Adjusted censoring offsets and arrows to keep the larger markers distinct.
- Tightened the quality-time y range without clipping any displayed mean
  or sample-SD bar. Shortened the on-figure note and retained all scientific
  definitions in the caption file. Bold parameter labels and the bottommost
  title remain. Updated the TS catalog description, handoff, and repository map.
- Rebuilt F12 in local memcompgen and inspected the PDF using pdftoppm.
  Checked stacked axes, dimensions, shared x labels, marker sizes, caption
  bounds/clearances, and all observed error-bar limits. Source files, numerical
  CSVs, and F01-F11 retain their hashes. Short/descriptive PDFs match.
  Ruff and Git whitespace checks passed. Work remains on codex/portable-plot-data;
  main is unchanged. No training, evaluation, commit, or push was performed.


## 2026-10-06 - Add the DiT counterpart of F12 as F13

- The researcher requested the same timescale figure for DiT. Registered
  F13 under the existing stacked TS template. Read only the portable
  CelebA32 DiT scalar tables: 12 diagonal runs, four sizes, seed 0, and
  72 matched FID/raw-pixel checkpoints per run. Kept rho=0.3, the 10% best-FID
  tolerance, sustained 5% onset, and paired delta_tau=tau_mem-tau_gen.
- Parameterized the paired-metric loader by experiment and extracted the
  existing per-seed checkpoint selector from the U-Net aggregation function.
  The shared timescale summary leaves SD blank for one seed; the renderer
  draws no error bars and explicitly labels seed 0. All 12 DiT onsets are
  observed and all signed gaps are negative. F13 fits y limits to these
  values while retaining log time axes and a linear gap axis including zero.
- F13 preserves F12's 3.2484-by-4.8-inch layout, five-point markers, canonical
  colors, bold math labels, and bottommost title. Saved PDF/SVG, caption,
  12 per-seed rows, and 12 summary rows under outputs/figures/celeba32_dit_3x3.
  Added the direct --experiment option and documented both build commands.
- Executed the direct DiT renderer and F13 catalog build in local memcompgen.
  Independently recomputed all 12 times/gaps directly from the CSVs and
  checked blank SD, matching checkpoint coverage, and absence of error bars.
  Inspected the PDF with pdftoppm and checked dimensions, data limits, and
  caption clearances. All previous U-Net selected checkpoints and timescale
  summaries match exactly; a temporary F12 rendering is pixel-identical to
  its previous rendering. All pre-existing source, CSV, and figure hashes
  remain unchanged except the catalog index. Catalog rendering preserves
  the new CSVs, and F13's short/descriptive PDFs match.
- Updated the plotting handoff and repo map. Ruff and Git whitespace checks
  passed. Work remains on codex/portable-plot-data; main is unchanged.
  No training, evaluation, commit, or push was performed.


## 2026-10-06 - Fit F11 to the AISTATS paper width

- The researcher requested a smaller F11 with its aspect ratio preserved and
  the reviewed AISTATS figure guidance applied. Set its canvas to 6.75 by
  3.33286 inches, preserving the earlier 9.923913-by-4.9-inch aspect ratio.
  The renderer now owns full-canvas width and aspect ratio directly. Kept
  the 3+3+2 arrangement, parameter colors, shade progression, and bottom title.
- Kept tick labels, panel letters, legend, and footer at eight points instead
  of shrinking the existing PDF. Standardized bold (a)-(h) labels above the
  upper-left corner of each panel. Reduced y-tick density, shortened legend
  spacing, and adjusted row spacing to prevent collisions. These typography
  choices support legibility; the official template does not mandate them.
- Rebuilt the catalog F11 PDF/SVG in local memcompgen and visually inspected
  a pdftoppm rendering. Verified exact width and preserved aspect ratio,
  all eight labels, page bounds and text clearances, embedded TrueType fonts,
  and absence of raster images. Compared every plotted line and band/error-bar
  path against the prior renderer: coordinates, colors, and line styles match.
  Recomputed summaries match both existing numerical CSVs exactly.
- Hash checks confirm all plot_data files, numerical tables, and other
  figures are unchanged. Only F11's graphic/caption artifacts and the output
  index changed. The short PDF matches the descriptive PDF. Ruff and Git
  whitespace checks passed. Updated the handoff and repository map, including
  a LaTeX figure* inclusion example; that manuscript snippet was not compiled.
  Work remains on codex/portable-plot-data; main is unchanged. No training,
  evaluation, commit, or push was performed.


## 2026-10-06 - Give F11's trajectory rows more height and tighten the footer

- The researcher clarified that both trajectory rows should have equal height
  and be taller than the bottom summaries. Set FID/f_mem/summary height ratios
  to 1.25/1.25/1, reclaiming part of the footer margin while retaining the
  8.3-by-4.0982-inch canvas and aspect ratio. Moved the N/parameter legend
  closer to the bottom panels, then restored a small gap at the researcher's
  request. Kept the title below the legend. Updated the plotting handoff.
- Rebuilt F11 in local memcompgen and inspected the PDF through pdftoppm.
  Checked equal trajectory heights and their ratio to the summaries,
  unchanged canvas, clear legend/title
  spacing, identical short/descriptive PDFs, and unchanged input-data,
  numerical-table, and other-figure hashes. Git whitespace check passed.
  Work remains on codex/portable-plot-data; main is unchanged.


## 2026-10-06 - Widen F11 slightly and enlarge both trajectory rows

- Increased canvas width from 8.3 to 8.8 inches while retaining its aspect
  ratio; height is now 4.3451 inches. Increased the equal FID/f_mem row weights
  from 1.25 to 1.4 relative to the bottom summary row. Together these changes
  make the figure about 6% wider and the two trajectory rows about 10% taller.
  Kept typography and the legend/footer positions. Updated dimension notes.
- Rebuilt F11 in local memcompgen, inspected its pdftoppm render, and checked
  equal top-row heights, the 1.4:1 trajectory/summary height ratio, legend/title
  clearance, and matching short/descriptive PDFs. Hash checks confirm unchanged
  input data, numerical tables, and other figures. Git whitespace check passed.
  Work remains on codex/portable-plot-data; main is unchanged.


## 2026-10-06 - Add vertical guides to F13

- Added faint dotted vertical gridlines at N=512/1024/2048/4096 in all three
  F13 panels. The Small-DiT-only style condition leaves F12 unchanged.
- Rebuilt F13 in local memcompgen and visually checked the pdftoppm render.
  Hash checks show only F13's PDF/SVG artifacts changed; all numerical tables,
  input data, and other figures are unchanged. Short/descriptive PDFs match.
  Updated the handoff. Git whitespace check passed.


## 2026-10-06 - Keep F12/F13 styling paired and supply paper captions

- The researcher requested that F13 style edits also apply to F12. Removed
  the DiT-only condition from the vertical guides. Both figures now use the
  same dotted x-grid styling. Recorded the paired-style convention in the
  plotting handoff; seed uncertainty, censoring, and data-fitted y limits
  retain their existing meanings.
- Added reproducible manuscript-ready `.caption.tex` outputs alongside the
  existing detailed text captions. Both define the per-run 10% FID criterion,
  sustained 5% raw-pixel onset at rho=0.3, signed gap, axes, colors, and sample
  counts. Captions distinguish three-seed U-Net SD and lower bounds from
  single-seed DiT values. They do not equate the signed gap with a continuous
  interval of near-best quality without memorization.
- Checked the readout against portable records. DiT has 12 negative gaps;
  U-Net has 25 negative and two positive observed seed gaps, plus nine censored
  onsets. Recorded the DiT N=4096 times and memorization fractions at tau_gen
  in the handoff. U-Net's v/v best-FID target changes with N, so its sharp
  quality-time drop does not compare equal absolute FID. Seed-zero data-bundle
  hashes match across architectures; microbatch sizes differ. The plots do
  not isolate capacity or architectural causes.
- Rebuilt both figures with `python plotting/figures.py --type TS` in local
  memcompgen. Visually inspected both PDFs and a one-page caption compilation
  using the official AISTATS 2027 style. LaTeX produced no overfull-box or
  float warnings. Verified all 12 vertical guides per figure, exact numerical
  agreement with existing CSVs, and matching short/descriptive PDFs. Hashes
  confirm unchanged source data, numerical tables, and other figures. Ruff
  and Git whitespace checks passed. Updated the repository map. Work remains
  on codex/portable-plot-data; main is unchanged. No training or evaluation ran.


## 2026-10-06 - Name the equivalent weights directly in F07

- Added x-equivalent and epsilon-equivalent to F07's weighted-velocity legend
  labels, alongside w=(1-t)^2 and w=t^2. Retained purple/orange styling.
- Rebuilt F07 in local memcompgen and visually checked label fit in the PDF.
  Hash checks confirm that only its PDF/SVG artifacts changed; numerical
  tables, input data, and other figures are unchanged. Short/descriptive PDFs
  match. Updated the handoff and passed the Git whitespace check.


## 2026-10-06 - Centralize figure identity, review, and the supplied matrix design

- The researcher requested a shared plotting workflow for multiple agents and
  chose type-based IDs. Moved the registry to plotting/figure_catalog.json.
  Existing figures now use QM01/QM02, QD01-QD03, DM01, WV01, BT01/BT02, DT01,
  DS01, and TS01/TS02. F01-F13 remain aliases, with byte-identical PDFs under
  both names. Existing artifacts are marked legacy until a recorded rebuild.
- Added explicit variant, data-source, metric, rho, seed/training-size,
  checkpoint-selection, uncertainty, placement, and review fields. All entries
  start draft/unassigned; prior style discussions are not paper approval.
  Added --show and composable list/build filters for type, placement, status,
  and metric. Catalog validation rejects duplicate identities/paths and
  scientific metadata unsupported by the selected renderer.
- Each catalog build preserves a PDF and available companion artifacts,
  source archive, and receipt under outputs/figures/versions/<ID>/<build-id>.
  Identity hashes cover the selected inputs, all plotting Python code,
  specification, and Python/NumPy/Matplotlib environment. The conservative
  code scope can invalidate unrelated approvals. Changed data/code/artifacts
  make an existing approval need review; archives are retained. Repeating the
  same identity reuses its first PDF, including creation metadata. No actual
  approval was recorded. Build records and PDFs remain ignored local artifacts.
- Added BT03 using the supplied two-matrix design for N=512 and 2048. Its
  tracked renderer reuses existing per-seed best-FID calculations from
  plot_data, with diagonal tints and paired FID. The supplied external CSV
  was not needed. The researcher was unsure about SE; used the existing
  sample-SD convention and explained the difference. The plot and caption
  explicitly label SD; the renderer supports a separately registered SE variant.
- Added plotting/AGENTS.md for shared agent conventions and updated the
  handoff/repository map. Ran BT03 and the registry CLI in local memcompgen;
  checked all 54 numerical labels against the existing table and their cell
  bounds, then inspected the PDF with pdftoppm. A temporary F12 alias build
  successfully creates TS01 and is pixel-identical to the original F12.
  Four focused pytest cases passed, plus Ruff and Git whitespace checks.
  Hashes confirm all pre-existing plots, captions, CSVs, and plot_data are
  unchanged; only the generated index changed. Work remains uncommitted on
  codex/portable-plot-data; main is unchanged. No training, evaluation, or push.


## 2026-10-06 - Specify paper figures from the accepted PDF layouts

- The researcher retained manuscript Figures 1-4 and Tables 1-2 from
  AISTATS_2027_Memparam_FM (1).pdf, then explicitly chose to keep the layouts
  while using raw pixels at rho=0.3 and sample SD. Figure 4 remains in the
  main set, superseding the earlier suggestion to move it wholly to appendix.
- Wrote docs/paper_figure_plan.txt with the reference PDF path/hash, six
  retained layouts, metric/checkpoint/seed rules, caption corrections, a
  proposed theory addition, and five supporting appendix groups. Preserved
  the PDF's existing panel order. Kept content decisions distinct from
  approval of catalog artifacts; no catalog review status was changed.
- The PDF's Figure 3 reports SSCD and SE. Figure 4's old caption disagrees
  with its plotted onset threshold, uncertainty, and stacked arrangement.
  Recorded those corrections for implementation within the retained layouts.
- Read the PDF with pdftotext and inspected pdftoppm renders of pages 2, 3,
  6, 7, and 8. Checked portable experiment/seed and metric-setting inventories
  in local memcompgen. This establishes planning context, not a full numerical
  audit. Updated the repository map. No plots or numerical inputs were edited,
  and no training, evaluation, commit, or push was performed for this plan.


## 2026-10-06 - Start a separate paper figure workspace with Figure 2 only

- The researcher requested a fresh output folder and focused generation of
  one item or a selected small group. Set the catalog and shared entrypoint
  to outputs/paper_figures/. Preserved the earlier outputs/figures/ collection.
  Made the weighting renderer accept the catalog destination as well.
- Added a short paper-item review list in the new folder. Its generated index
  lists only built items. Recorded the focused workflow in the plotting agent
  guide, paper specification, handoff, and repository map. Existing F-number
  aliases remain compatibility names, separate from manuscript numbering.
- Confirmed that DS01 already implements Figure 2's eight-panel template and
  TS01 implements Figure 4's stacked timescales, both with raw pixels at
  rho=0.3 and sample SD. Generated only DS01 in the new folder as the first
  review draft, build dfe885cd99aa; retained the existing layout. Its caption
  now refers to manuscript Figure 4 rather than the old F12 alias.
- In local memcompgen, ran the DS01 catalog command, inspected a pdftoppm
  render, and checked all 48 best-FID summary means/SDs against the existing
  raw-pixel summary CSV. Four catalog tests and Ruff passed. Verified that all
  catalog stems use the new folder and that only DS01 has a new build receipt.
  SHA256 checks confirmed all 285 prior output/input files remained unchanged.
  No figures were copied in bulk, no scientific records were edited, and no
  training or evaluation ran. Work remains local on codex/portable-plot-data;
  main is unchanged, with no commit or push.


## 2026-10-06 - Adopt the supplied M1-M5 and A1-A5 names

- The researcher explicitly required the names from the supplied paper plan.
  Replaced primary type-based catalog IDs with M/A names and descriptive
  suffixes for variants. M4 is the robustness table, M5 is theory, and A3 is
  the timescale group using reference Figure 4's accepted layout. Table 1 is
  the companion M1_CONVERSIONS. Planned items without renderers remain listed
  in the specification and review list rather than being declared complete.
- Updated the entrypoint, catalog, index, paper specification, agent guide,
  handoff, and repository map. Older commands remain compatibility aliases,
  but no alias PDFs are created. Current output files use M/A names directly.
- Generated only M2 under its correct name, build 43d9a790863d. A pdftoppm
  pixel comparison confirms the figure is identical to the preceding draft.
  Its caption now refers to A3. Removed the superseded working copies after
  checking their hashes; preserved the earlier archived build and source.
- In memcompgen, four updated catalog tests and Ruff passed. SHA256 checks
  confirm all 285 earlier output/input files remain unchanged. No numerical
  records or scientific calculations changed. Work remains local on
  codex/portable-plot-data; main is unchanged. No commit or push.


## 2026-10-06 - Mark M1 complete and approve M2

- The researcher confirmed that M1 is handmade by the authors and complete.
  Checked it off in the paper specification and workspace review list.
- The researcher accepted the current M2. Confirmed the receipt is current
  and recorded approval of exact build 43d9a790863d in the catalog. M2 covers
  the three diagonal objectives at N=512/1024/2048/4096, with FID and raw-pixel
  memorization trajectories, best FID, and paired memorization at each seed's
  best-FID checkpoint. It uses rho=0.3 and three-seed means/sample SD.
- Updated the review list and handoff and refreshed the generated index.
  No figure was rebuilt and no numerical input or plotted value was changed.


## 2026-10-06 - Adapt the supplied paper's M3 design to raw pixels and M2 width

- The researcher preferred the M3 design already in the supplied paper PDF
  and requested raw-pixel memorization and M2's width. Read the supplied Python
  reference and inspected PDF page 7. Attributed the layout to those supplied
  references. The PDF has wide cells and inline mean/uncertainty, unlike the
  earlier square-cell local adaptation.
- Updated only the M3 renderer: 8.8-inch width from M2's width constant,
  2.5-inch height, labels above/left, colored diagonal borders, gray FID, and
  neutral off-diagonals. Extracted PDF vector colors to match its 8% target
  tints and canonical palette. Retained shared bold STIX parameter symbols.
- Used the existing per-seed best-FID selection from plot_data, paired raw
  pixels at rho=0.3, and sample SD for both displayed metrics. Generated M3
  build 9ad16db1854a with PDF/SVG and corrected text caption. It remains draft.
- In local memcompgen, independently joined source CSVs and checked all 18
  cells (72 displayed means/SDs), label containment, and exact equal PDF
  widths of 633.6 points. Inspected the pdftoppm preview and passed Ruff and
  whitespace checks. All pre-existing files except the generated index are
  byte-identical, including approved M2, its archived build, and all inputs.
  The conservative all-plot-code fingerprint makes M2's source-state badge
  stale; its approved build ID and artifact were not changed or reapproved.
  Updated the specification and review list. No training, evaluation, commit,
  or push; work remains on codex/portable-plot-data and main is unchanged.


## 2026-10-06 - Add requested M3-SSCD and M3-DIT variants

- Reused M3's two-matrix template, paper colors, shared bold STIX labels,
  N=512/2048, and 8.8-by-2.5-inch canvas. Added the exact requested names
  M3-SSCD and M3-DIT to the catalog and commands; both remain draft with
  manuscript placement unassigned. The headline M3 remains raw-pixel U-Net.
- M3-SSCD uses CelebA32 U-Net SSCD at held-out-calibrated alpha=0.01,
  with each run's exported calibrated rho. It shows mean +/- sample SD over
  seeds 0/1/2. Build b31554d75753.
- M3-DIT uses CelebA32 Small-DiT raw pixels at fixed rho=0.3, seed 0 only.
  Both f_mem and FID appear without an across-seed uncertainty estimate.
  Build bce5c65ba5f4. Both variants select each seed's own best FID and pair
  memorization at that checkpoint; exact FID ties select the earlier step.
- Extended the existing shared run/metric loader and best-FID summary rather
  than copying a plotting pipeline. Catalog validation rejects mixed fixed
  rho/calibrated alpha labels and unsupported seed/uncertainty declarations.
  Kept absent single-seed SD as None. Wrote PDF/SVG, captions, input/code
  fingerprints, and source archives under outputs/paper_figures/.
- Verification in local memcompgen (Python 3.10.19, Matplotlib 3.10.8):
  11 targeted pytest checks passed for calibration selection, checkpoint ties,
  per-seed pairing, sample SD, absent single-seed uncertainty, and catalog
  provenance contracts. Ruff and git diff --check passed. Independently joined
  the source CSVs and checked 54 U-Net and 18 DiT selected checkpoints and
  all 36 matrix cells. Checked accepted run protocols and common checkpoint
  schedules with the existing quality loader. All 36 legacy raw-pixel summaries
  still match the prior exported summaries. Inspected pdftoppm previews and
  confirmed both PDFs are 633.6 by 180 points. Current receipts and artifact
  hashes match. All pre-existing numerical inputs, figure files, and archives
  are byte-identical; only the existing output README and INDEX were updated.
- Documented the variants and reproducible selected-pair command in the
  paper specification, handoff, output README, and repository map. No training,
  evaluation, commit, or push. Branch remains codex/portable-plot-data;
  main and the exact approved M2 artifact/build were not changed.


## 2026-10-07 - Build M4 / Table 2 with common pixels and a separate feature row

- The researcher requested consistent raw-pixel memorization across datasets
  plus a separate row for applicable feature metrics. Kept the supplied PDF's
  Table 2 layout: four dataset/model columns, N=2048, colored x/x, eps/eps,
  v/v triplets, underlined lowest mean, and a gray quality row.
- Added plotting/plot_robustness_table.py and catalog M4. It reuses the
  accepted quality loader, selects the lowest saved FID separately per seed
  (MNIST uses its primary feature-space FD), and pairs both memorization
  readouts at that same checkpoint. Exact ties select the earlier step.
  Raw pixels use fixed rho=0.3 everywhere. Features use SSCD for CelebA and
  the frozen MNIST digit encoder at held-out-calibrated alpha=0.01.
- The caption describes SSCD as image-copy-detection descriptors and explains
  that the MNIST encoder was trained to reconstruct digit images and classify
  their labels. Verified that definition against the training code and asset
  manifest; checked the official SSCD source. The caption distinguishes the
  common pixel threshold from the feature thresholds targeting a 1% held-out
  flagged rate, and states that both representation and threshold change.
- All metrics show mean +/- sample SD over three seeds where available.
  Small-DiT is explicitly seed 0 and has no SD estimate. Printed values come
  directly from the exports. Two DiT feature percentages round to 98.5 and
  91.1 rather than the old PDF's 98.6 and 91.2 because their stored floats lie
  just below the respective one-decimal rounding boundaries; no metric was
  adjusted to match the old document.
- Generated draft build 14a15d9cc20d: M4.tex is a manuscript table fragment,
  M4.pdf previews that exact fragment as Table 2 at 8.8 x 2.4 inches, and
  separate text/LaTeX captions accompany the receipt and archived source.
  The fragment scales to the enclosing text width and leaves table numbering
  to the manuscript. Updated the plan, handoff, repository map, and review list.
- Verification in local memcompgen: 19 targeted pytest checks passed for
  selection, calibration filters, missing/duplicate pairs, catalog scientific
  contracts, and archive semantics. Ruff and git diff --check passed.
  Independently read source CSVs and checked all 30 selected seed checkpoints,
  36 displayed means, and 27 sample SDs, including the actual LaTeX numbers.
  Inspected the pdftoppm preview. pdfTeX 1.40.29 compiled the exact fragment
  and a separate two-column manuscript wrapper at 6.75-inch text width with
  automatic numbering and no overfull boxes. Current receipt/artifact hashes
  match. All pre-existing inputs, figures, and archives remain byte-identical;
  only the output README and INDEX changed. M2's approved artifact and all
  M3 versions were preserved. No training, evaluation, commit, or push;
  branch codex/portable-plot-data and main unchanged.


## 2026-10-07 - Restore M4's original four-column manuscript fit

- The researcher preferred the original Table 2's width and supplied both
  LaTeX versions. Removed the extra metric-label column and the fixed-width
  resizebox. Four makecells now stack raw-pixel memorization, feature
  memorization, and gray paired quality within each dataset/model column.
  The caption specifies that order and retains the metric explanations.
- Restored pale header/body backgrounds, native small type, and tabular* at
  the enclosing line width. The supplied PDF's table rules span 486 points
  (6.75 inches); the new preview uses that text width, with 0.12-inch margins,
  instead of an 8.8-inch canvas designed for the earlier figures.
- Generated draft build fc8f3ed80518. Inspected the PDF render and compiled
  the exact table fragment in a two-column wrapper at 6.75-inch text width.
  Both compilations passed without overfull boxes. Compared all 36 formatted
  entries, including every SD and underline, to the preceding M4: identical
  after accounting for the new column-major ordering. No numerical logic
  changed. Ruff and git diff --check passed in local memcompgen.
- Updated the paper specification, handoff, and review list. Prior M4 builds
  remain archived. Numerical inputs and other figures were preserved; only
  current M4 outputs and the output README/INDEX changed. No training,
  evaluation, commit, or push; branch codex/portable-plot-data, main unchanged.

## 2026-10-07 - Remove M4 font and spacing overrides

- Followed the researcher's request to retain the supplied original Table 2
  styling. M4 now inherits manuscript fonts and tablehead/tablerow/qgray colors.
  Restored red/blue/green!60!black triplets, text-mode numbers, original small
  type, arraystretch=1.55, default column padding, and plain makecell breaks.
  Removed shared font overrides and the custom caption styling. Thin spaces
  around slashes are needed to fit the added SDs at the original text width.
- Generated draft fa972b2246c8 with the existing M4 command in memcompgen.
  pdfTeX compiled one page without overfull boxes; inspected the rendered PDF.
  Compared against fc8f3ed80518: all 36 means, 27 SDs, and eight underlines
  are unchanged. Numerical inputs and pre-existing other figures/archives
  remain byte-identical. Ruff and git diff --check passed; receipt is current.
- Documented M4's explicit exception to the shared figure typography.
  No training, evaluation, commit, or push. Branch codex/portable-plot-data;
  main unchanged.

## 2026-10-07 - Use the original compact M4 numbers and add a metric column

- The supplied original shows means only. Displaying every mean with its SD
  caused the width pressure in the preceding versions. Restored means-only
  triplets and ordinary slash spaces, then added a left-hand Metric column
  for raw pixels, features, and best quality. Kept the original small type,
  row spacing, default column padding, colors, and manuscript text width.
- Shortened the caption, stated that displayed values are means, and retained
  per-seed best-quality selection, metric definitions, thresholds, and seed
  coverage. Updated the M4 catalog contract to uncertainty=none. The loader
  still computes sample SD; the previous SD table remains archived.
- Built 515a0307ff37 in memcompgen. pdfTeX compiled at 6.75-inch text width
  without overfull boxes; inspected the PDF. All 30 selected checkpoints,
  computed means/SDs, 36 displayed means, and eight underlines match the
  prior results. Numerical inputs and all other figures/archives are unchanged.
  Sixteen targeted robustness/catalog tests, Ruff, and git diff --check passed.
- No training, evaluation, commit, or push. Branch codex/portable-plot-data;
  main unchanged.

## 2026-10-07 - Record explicit approval of M4

- The researcher approved the compact robustness table and asked to check it
  off. Marked exact build 515a0307ff37 approved in the catalog, paper checklist,
  handoff, and output review list; refreshed the linked index without rendering.
- In memcompgen, validated the catalog and confirmed approved/current status.
  All M4 artifact hashes remain unchanged. No plot code, inputs, or other
  approvals changed. No commit or push; branch codex/portable-plot-data,
  main unchanged.

## 2026-10-07 - Name main timescales M5 and the theory addition M6

- The researcher explicitly assigned M5 to tau_gen, tau_mem, and their signed
  gap, and moved the planned theory/computation item to M6. Changed the U-Net
  timescale catalog entry from A3 to M5 with main-text placement. F12/TS01
  aliases now resolve to M5. A3 remains the extended appendix group, including
  A3_DIT. M6 remains planned, with no invented renderer or completion status.
- Updated the checklist, handoff, repository map, plotting guide, output list,
  CLI naming, and the future M2 caption reference. Generated only M5 from the
  existing unchanged timescale renderer: draft build 2f7c2a183a0e, PDF/SVG and
  captions. No evaluation or CSV export was run.
- In memcompgen, all eight catalog tests, Ruff, and git diff --check passed.
  Checked M5's current receipt, main-text placement, aliases, and PDF preview.
  Hashes confirm that every pre-existing numerical input, figure, and archive
  remains unchanged, except the output README and index. M2/M4 retain their
  exact approved-build IDs. The conservative all-code hash marks older source
  receipts stale after these naming edits; their approved artifacts are intact.
- No training, evaluation, commit, or push. Branch codex/portable-plot-data;
  main unchanged.

## 2026-10-07 - Crop workshop Figure 4 and use its horizontal timescale layout

- Extracted page 8 of the supplied workshop PDF and cropped Figure 4's vector
  artwork, including all labels. Saved PDF/PNG and source hash/crop coordinates
  under outputs/paper_figures/references/. This reference supplies no numerical
  inputs to the current plots.
- Replaced the shared M5/A3_DIT stack with three horizontal panels on a
  6.75-by-1.95-inch canvas. Used linear time axes with 10^6 multipliers,
  matching limits for tau_gen/tau_mem, small markers, thin lines, full axes
  frames, and short mathematical labels. Removed large headings and footer
  text; the separate captions supply definitions and color/marker keys.
- Preserved raw pixels at rho=0.3, sustained 5% onset, per-seed pairing,
  sample SD, negative gaps, and explicit censored lower bounds. The DiT gap
  panel fits its observed negative range and retains the zero reference.
- Generated M5 build 1632be7bf6a4 and A3_DIT build 68d2ee0590c5, plus PNG
  display previews named by build ID. Inspected both rendered PDFs and the
  workshop crop. The earlier stacked M5 remains archived.
- In memcompgen, all 48 per-seed times and 24 summaries (including SDs and
  lower bounds) match the preceding code exactly. All numerical inputs,
  pre-existing other figures, and archives remain byte-identical. Current
  receipts/captions, eight catalog tests, Ruff, and git diff --check passed.
- Updated the current plan, handoff, repository map, and output review list.
  No training, evaluation, CSV export, commit, or push. Branch remains
  codex/portable-plot-data; main unchanged.


## 2026-10-07 - Retain original workshop M5 and begin A2/A5 comparisons

- The researcher selected the original workshop Figure 4 for M5 and requested
  a revised caption plus work on representation/threshold agreement. M5 now
  copies the vector crop exactly. The caption states CelebA32 grayscale U-Net,
  seven training sizes, raw pixels at rho=0.5, first 1% onset, mean +/- SE,
  and assignment of unobserved onsets to the final checkpoint. It explains
  the resulting plateaus and capped gaps without claiming observed onsets or
  a continuous generation window. Build 686001f79577 remains caption-review
  draft; selection of the artwork does not approve a new caption silently.
- Added the portable source crop/provenance under plotting/references/ and
  a minimal reproduction renderer. Catalog receipts hash/archive both source
  assets. The previous current-data M5 remains in its original archive and
  its renderer is registered as A3_UNET, alongside A3_DIT. No historical
  numerical values were extracted, interpolated, or mixed into current data.
- Generated initial A2 (16680788af83) and A5 (f9bcf9a596cc) drafts using the
  same 36 current CelebA32 U-Net diagonal runs at their own best-FID step.
  A2 compares pixels, SSCD, LPIPS, and DINOv2 calibrated at alpha=0.01.
  A5 compares fixed raw-pixel rho=0.3/0.5/0.7. Four N panels use mean +/-
  sample SD, canonical target colors, and shared bold STIX target labels.
  Both use one shared renderer and write PDFs/SVGs, captions, and receipts.
- Scientific readout: the fixed-rho mean ordering eps/eps < v/v < x/x
  persists across all four N, but representation rankings can change.
  At N=1024 calibrated DINOv2 gives x/x=53.6% versus eps/eps=61.3%.
  This is an aggregate-rate comparison, not per-image agreement or a
  significance test. The scalar exports cannot supply per-image overlap.
  Neither draft tests the first-1% versus sustained-5% onset definition;
  that belongs to A3. Calibrated-alpha sweeps and other datasets are planned.
- Verification in local memcompgen: 11 targeted tests passed for checkpoint
  pairing, earliest ties, missing/duplicate readouts, and catalog archives.
  Independent CSV joins matched all 36 selected checkpoints and 252 metric
  readouts; independent arithmetic matched 84 means and sample SDs. Ruff and
  git diff --check passed. Inspected final PDF renders; the three caption
  fragments compiled with pdfTeX 1.40.29 without overfull boxes.
- M5 is byte-identical to the selected source crop. Verified current receipts,
  archived artifacts, and archived reference assets. All pre-existing input
  data, other figures, and archives are unchanged; only M5 current aliases,
  README/INDEX and the reference provenance note were updated. M2/M4 exact
  approved artifacts and approval IDs remain unchanged. The obsolete current
  M5 SVG was removed from its alias; its archived copy remains intact.
- Updated the plan, local handoff, plotting guide, repository map, and output
  review list. No training, evaluation, CSV export, commit, or push. Branch
  remains codex/portable-plot-data; main unchanged.


## 2026-10-07 - Interpret A2/A5 and draft two caption options for each

- Re-read the current 252 selected memorization readouts and verified the
  metric definition r=d1/d2 with the strict flag r<rho. Checked held-out
  quantile calibration in the implementation; alpha targets a held-out
  flagged rate and does not guarantee agreement on generated-image flags.
- Summarized the broad decline with training-set size, the stable mean
  ordering across fixed pixel rho values, and representation-dependent
  rankings. Pixels, SSCD, and LPIPS give the lowest mean to eps/eps at every
  displayed N; DINOv2 gives x/x below eps/eps at N=1024 and close v/v and
  eps/eps means at N=4096. These are descriptive seed means, not claims of
  universal ordering, per-image agreement, or statistical significance.
- Added short and fuller appendix caption alternatives for each figure under
  outputs/paper_figures/, as .caption.short.txt/.tex and
  .caption.appendix.txt/.tex. They state dataset/model, four N panels,
  threshold/calibration rule, per-seed best-FID pairing, 10,000 samples,
  target color/marker key, and three-seed sample SD. Fuller captions include
  numerical examples and limits of interpretation. Each TeX file identifies
  its existing figure build in a comment.
- Checked the printed claims against the selected source values and checked
  LaTeX delimiters and replacements. No numerical inputs, plot code,
  generated figures, default captions, catalog entries, or archived builds
  changed. No training, evaluation, commit, or push; branch remains
  codex/portable-plot-data and main unchanged.


## 2026-10-07 - Add A5 trajectory threshold-sensitivity companion

- The researcher requested both the existing best-FID summary and a version
  following workshop Appendix A.3 Figure 7. Registered A5_TRAJECTORIES and
  extended the shared sensitivity renderer with a 3x3 grid: epsilon/epsilon,
  v/v, x/x rows and rho=0.3/0.5/0.7 columns. Used the current CelebA32 U-Net
  four-size, three-seed data exclusively from plot_data/.
- Reused the existing fixed-rho loader and M2's target colors and N shades/
  line styles. All 72 checkpoints contribute to aggregation, including zeros;
  the 71 positive steps are displayed on log axes. Zero mean values break
  lines; no positive values are fabricated. Small markers preserve isolated
  positive observations. Mean +/- sample SD bands are clipped only for
  display. The concise caption explains the grid and log-axis handling.
- Generated draft c4fd8a7b639e: PDF/SVG, text/LaTeX captions, build receipt,
  source archive, and a preview named by build ID. Inspected the final PDF
  render. The caption compiled with pdfTeX 1.40.29. Existing A5 best-FID
  outputs and all other figure artifacts/captions remain unchanged.
- Verification in local memcompgen: independent CSV joins matched all 7776
  source values (36 runs x 72 checkpoints x three rho values) and all 2592
  mean/sample SD summaries. Confirmed rho monotonicity and identical step
  schedules; the smallest positive plotted mean is 1/30000. Twelve targeted
  tests passed, including zero-valued seed inclusion before log display.
  Ruff and git diff --check passed. Current receipt and archive hashes match.
- Preservation hashes confirm that all pre-existing plot_data inputs, other
  figures, captions, and version archives remain byte-identical; only the
  output README and INDEX were updated. Updated the figure plan, handoff,
  and repository map with the focused reproducible command. No training,
  evaluation, CSV export, commit, or push. Branch codex/portable-plot-data;
  main unchanged.

## 2026-10-07 - Shorten the A5 trajectory caption

- Replaced the detailed A5_TRAJECTORIES caption with a concise account of
  the panel layout, threshold effect, lower late-training eps/eps rates,
  training-size key, three-seed sample SD, and omitted zeros on log axes.
  Updated the existing caption generator and saved both plain-text and
  LaTeX versions through the catalog build, 8a701abbeb2f.
- The focused memcompgen build succeeded. Checked the late-training claim
  against every saved step from 100k onward for all four sizes and three
  thresholds. Caption text matches the generator and receipt; input hashes
  are unchanged. Previous builds remain archived. No numerical inputs,
  plot layout, training, or evaluation changed.

## 2026-10-07 - Joint FID/memorization full grids for U-Net and DiT

- Inspected workshop appendix Figure 5 on PDF page 11. Reused its dotted
  FID/left-axis and solid F_mem/right-axis convention in full prediction/loss
  3x3 grids. The researcher explicitly chose all four training sizes overlaid
  in each cell. A1 is CelebA32 grayscale U-Net; added A1_DIT for Small-DiT.
- Extended the existing grid template and aggregation to support DiT seed 0
  without estimating sample SD. Both figures use raw pixels at rho=0.3,
  prediction colors, size shades/markers, identical linear metric ranges,
  and logarithmic training steps. U-Net retains three-seed mean/sample SD.
  All 71 positive steps are displayed without smoothing; step zero remains
  in the source records and the 72-checkpoint aggregation.
- Executed `python plotting/figures.py A1 A1_DIT` in local memcompgen,
  producing builds a54071c96364 and 626e51c96721. Saved PDF/SVG, concise
  text/LaTeX captions, receipts, source archives, and previews named by build.
  Inspected both Poppler renders. Both captions compiled at manuscript width
  with pdfTeX, without overfull boxes or undefined commands.
- Independently joined the original CSV records for all 108 U-Net and 36 DiT
  runs: checked all 20736 source values, 10368 means/SDs, sample counts, image
  format, and matching FID/F_mem checkpoint schedules. Artifact receipt hashes
  match. Sixteen targeted tests passed, including zero-valued seeds, missing
  single-seed SD, and mismatched checkpoints. Ruff and git diff --check passed.
- Preservation hashes confirm all 254 pre-existing plot-data and paper-output
  files remain unchanged except README/INDEX. Updated the figure plan,
  handoff, map, and review list. No numerical exports, training, evaluation,
  commit, or push. Branch remains codex/portable-plot-data; main unchanged.

## 2026-10-07 - Increase line distinction in A1 and A1_DIT

- Strengthened the four size shades within each prediction-target hue and
  increased line width from 1.35 to 1.65 points. Enlarged size markers from
  3.1 to 4.3 points and staggered their placement across saved checkpoints,
  so coincident trajectories no longer place every marker on top of another.
  FID retains dotted lines and hollow markers; F_mem retains solid lines
  and filled markers with white edges. No data coordinates changed.
- Rebuilt both through the catalog in memcompgen: A1 651f57677848 and
  A1_DIT a8e822a66262. Inspected both Poppler previews and saved them beside
  the outputs. Receipt hashes match. Numerical input hashes, captions, other
  figures, and all prior archives are unchanged. Ruff and git diff --check
  passed. This was a visual-only adjustment; no new numerical tests needed.

## 2026-10-07 - Explicit four-shade palettes for the joint grids

- Replaced blended tints with four selected shades within each red, blue,
  and green family, ordered darkest N=512 to lightest N=4096. Both metrics
  retain the same shade for a given size. The size legend now shows all
  three actual target shades instead of gray markers. Larger staggered
  markers and dotted FID/solid F_mem conventions remain in place.
- Updated the concise captions to state the dark-to-light size ordering.
  Rebuilt A1 b97087025b26 and A1_DIT 66d374819b1f in memcompgen and inspected
  both Poppler previews. Updated the review list and handoff with the palette
  owner GRID_SIZE_COLORS. Artifact receipts match; numerical input hashes,
  other figures, and prior archives remain unchanged. Ruff and git diff
  --check passed. No numerical computation logic or evaluation changed.

## 2026-10-07 - Remove F_mem markers from both joint grids

- Removed markers only from the solid F_mem curves in A1 and A1_DIT;
  dotted FID retains its larger, staggered markers. Updated both captions
  to identify markers as FID markers. Rebuilt in memcompgen: A1 05543ae29401
  and A1_DIT c4c62229008d. Inspected both Poppler previews. Input hashes
  match the previous builds; the numerical calculations are unchanged.

## 2026-10-07 - Remove remaining FID markers from the joint grids

- Removed FID markers in A1 and A1_DIT, leaving dotted FID and solid F_mem
  with four size shades. Replaced marker legend keys with color swatches and
  updated the captions. Rebuilt both in memcompgen: A1 38310d9ad2fa and
  A1_DIT 47f139447af6. Inspected both Poppler previews. Numerical inputs
  match the previous builds; artifact receipts verified. Prior builds remain
  archived. No numerical calculations or evaluation records changed.

## 2026-10-07 - Use clear line samples in the bottom legend

- Lengthened and thinned the three colored legend samples for each training
  size in both joint grids so they read as lines rather than small blocks.
  Rebuilt in memcompgen: A1 25d0461836dd and A1_DIT c193bb3f5dcb. Inspected
  both Poppler previews; the legend fits without overlap. Artifact receipts
  and unchanged numerical input hashes verified. Captions and curves retain
  the previous content. Prior builds remain archived.

## 2026-10-07 - Audit theory notebook and draft linear/RFN paper text

- The researcher opened the figure/final-rewrite session and selected the
  supplied theory notebook and pasted theory draft for review. They fixed the
  scope to linear prediction plus frozen random-feature networks, with no RMT,
  and requested the unified predictor B_q*phi(z): identity or nonlinear features.
- Independently verified the linear velocity equation, initialization effect,
  gap identity, general-t slopes, midpoint cubic epsilon term, and random-init
  correction. The linear result measures fitting of sample second moments;
  at the Gaussian midpoint, zero velocity is already population-optimal.
- Reconstructed exact linear curves in local memcompgen without running SGD.
  The 50-dataset epsilon mean retains a quadratic residual: its first five
  plotted log-log values are nonpositive. The cubic diagnostic needs an
  explicitly labelled zero-mean correction or a more accurate ensemble mean.
- Derived and independently reviewed the finite-feature expected initial-gap
  slope, expressed as the variance of per-example feature--target moments.
  Deterministic enumeration/Gaussian quadrature agreed within 7e-17 and the
  Stein identity within 1.2e-16. Nonlinear features can give a linear epsilon
  term; neither cubic delay nor a shared limiting velocity transfers generally.
- Added proposed main/appendix TeX snippets and a detailed audit under
  notebooks/theory/. Both compile in a temporary two-column article wrapper;
  the main text occupies one page without a figure. No overfull equations or
  unresolved references remain after reruns. Actual manuscript layout remains
  unchecked. Updated the repository map and M6 planning status.
- Existing notebook and figures remain unchanged. Their cleared execution
  record does not verify saved SGD results. The RFN source has no computed
  exact-trajectory overlay; numerical moments and final M6 rendering remain
  outstanding. No training, model evaluation, cluster action, commit, or push.
  Branch codex/portable-plot-data; main unchanged.

## 2026-10-07 - Combine selected linear and RFN initialization panels

- The researcher requested the rightmost A_init_sweep panel beside one RFN
  panel. Explained that the RFN panels differ only in zero versus Gaussian
  random readout initialization. Selected its random-initialization panel to
  pair the common entry-variance rule 1/p, with p=d for linear features.
  This rule does not match initial velocity functions or feature variances.
- Split the two existing simulation/plotting cells in the theory notebook.
  Named result dictionaries preserve each study's arrays and time grids, so
  replotting no longer depends on globals overwritten by the other study.
  Added a separate combined M6 cell with shared axes, canonical paper style,
  target colors, seed counts, median curves, and 10--90% intervals. Retained
  the linear panel's exact zero-initialization mean references and labelled
  them separately. Saves M6.pdf/M6.png to the existing OUT directory.
- Updated notebook instructions, the theory audit, M6 plan, repository map,
  and plotting handoff. The user's selected gap-only layout supersedes the
  earlier proposed four-panel layout. The log-log figure is not in M6.
- Verification in local memcompgen: all code cells compile; nbformat validates
  with the supplied notebook's existing missing-cell-ID warning. AST checks
  confirm both simulation bodies are unchanged. Both original renderers and
  the new combined renderer execute with synthetic fixtures. Checked selected
  medians, exact-reference arrays, original time grids, common axes, and the
  three intervals per panel. Inspected the labelled synthetic layout preview.
  All six supplied PDF/PNG files remain byte-identical. No training ran and no
  empirical M6 render is claimed. The researcher runs both simulation cells
  once to populate the new dictionaries, then reuses the M6 plotting cell.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Match M6 paper styling and place legends inside panels

- The researcher requested the same visual style as the main/appendix figures
  and explicitly placed legends inside the plots. Inspected current A1/A2
  previews and the M2/A1/A5 renderers before changing the notebook's M6 cell.
- Used the common 8.8-inch width, DejaVu Sans/STIX typography, bold target
  symbols, compact titles with panel letters, thin axes/ticks, light dotted
  major grid, and lighter uncertainty bands. Removed the large initialization
  heading and separate seed-count title rows; these belong in the caption.
  Target legends sit inside both panels. The linear panel also contains the
  solid/dashed reference and interval key. There is no bottom legend.
- Only the M6 Markdown and rendering cells changed in the notebook. The
  simulation code, selected runs, percentiles, references, and time grids are
  unchanged. Updated the M6 style plan. No training or model evaluation ran.
- Verification in local memcompgen: notebook code compiles; synthetic-fixture
  renders pass exact checks of every line coordinate and band vertex against
  the previous renderer. Font, grid, spine width, and in-axes legend bounds
  pass. Visually inspected the labelled layout preview. All eight current
  theory figure files, including the researcher's M6, remain byte-identical;
  rerunning the M6 cell applies the new style to the live simulation arrays.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Save notebook M6 beside the other paper figures

- The researcher requested the shared M/A figure destination. Changed only
  the notebook's M6 export path and its instructions to use
  outputs/paper_figures/M6.pdf and M6.png, resolved from the repository root.
- Copied the researcher's current M6 PDF/PNG into that folder. Source and
  destination hashes match. Added M6 to the folder README and index as a
  notebook draft without a catalog receipt, and updated the destination notes.
- Local memcompgen checks: every notebook code cell compiles; only the M6
  Markdown/rendering cells differ. Checked root resolution from both the repo
  and notebook directory, and checked both save paths with a recording sink.
  No training or rendering was needed. Existing source figures remain intact.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Stack M6 and remove the empty early RFN time range

- The researcher requested stacked panels and noted the empty start of RFN.
  M6 now places linear above RFN at 4.4 by 5.4 inches, with the shared paper
  style and legends inside. The vertical limits remain common; each log-time
  axis covers its own plotted points with a small margin. No values or times
  were rescaled, removed, or interpolated.
- Recomputed only the RFN step-size pilot in local memcompgen. Its largest
  feature-moment eigenvalue is 3.8656426, so the first recorded time is
  0.3/lambda_max = 0.0776068. The previous shared axis began near 0.003.
  That empty interval has no recorded RFN points; it does not show an absence
  of learning. Resolving earlier times requires smaller updates and a new run.
- Changed only M6 Markdown/rendering cells in the notebook and corresponding
  notes. All code cells compile; nbformat validates with the existing missing
  cell-ID warning. A synthetic-fixture render preserves every curve coordinate
  and interval vertex, has stacked panels with common y limits and separate
  x limits, and keeps all legends inside. Inspected the labelled preview.
- Verification command: MPLCONFIGDIR=/tmp/theory_combined_mpl conda run
  --no-capture-output -n memcompgen python /tmp/check_theory_stacked.py.
  All existing empirical figure files remain byte-identical. The researcher
  must rerun only the M6 cell in the populated kernel to export the new layout
  to outputs/paper_figures/M6.pdf and M6.png. No training or model evaluation
  ran. No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Enlarge M6 text relative to the stacked panels

- The researcher found the text too small relative to the panels. Reduced M6
  from 4.4 by 5.4 to 4.2 by 4.8 inches. Increased labels and titles to 12 pt,
  ticks to 10.5 pt, and legends to 10.5--11 pt. Preserved the researcher's
  14-point panel letters. Put target keys in one row inside each panel;
  the linear reference key sits below them. Both time axes remain labelled.
- Changed only the two M6 notebook cells and the figure size/style note.
  Preserved all other cells, simulation code, existing outputs, and figure
  files. The M6 cell still exports to outputs/paper_figures/M6.pdf and M6.png.
- Local memcompgen check: /tmp/check_theory_readability.py compiled all code
  cells and rendered synthetic arrays to /tmp only. Curve coordinates, band
  vertices, and axis limits match the previous renderer exactly. All legends
  fit inside the axes. Inspected the preview. No training or evaluation ran;
  rerun M6 with the live result dictionaries to update empirical exports.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Remove M6 legends and shorten vertical labels

- Removed all legends from both M6 panels at the researcher's request. Both
  vertical labels now show only Delta G_q(tau). Removed unused legend imports
  from the M6 cell and moved the color/style key into the caption notes.
- Preserved the compact stacked layout, enlarged text, curves, intervals,
  axis limits, and export paths. Only M6 Markdown/rendering cells changed in
  the notebook; existing saved notebook outputs remain until the next rerun.
- Local memcompgen runtime check with /tmp/check_theory_no_legends.py:
  all code compiles; a synthetic render has no legends and the requested
  vertical labels. Curve/band coordinates and axis limits match the prior
  renderer. All other cells and existing empirical exports are unchanged.
  No training ran. Rerun only M6 with live result arrays to update the exports.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Build A4 matched velocity-weighting comparison

- The researcher selected A4 and requested an inspection and paper-style
  revision. Showed the earlier diagonal-versus-weighted plot. Offered the
  scientific choice between diagonal and matched-row comparisons; with no
  reply during the inspection, proceeded with the existing A4 plan and kept
  the diagonal variant separate. No original figure export was overwritten.
- Registered A4 as the matched velocity-output comparison in the existing
  weighting renderer. Explicit native-v/v weights (1-t)^2 and t^2 are paired
  with matrix v/x and v/eps. One matrix v/v baseline is retained per N/seed;
  all six repeated unit controls match exactly and are not extra seeds.
  A4_DIAGONAL retains its original selection and legacy aliases.
- Verified matched exported protocol fields and saved data-bundle hashes.
  Across paired checkpoints, maximum absolute FID/raw-pixel f_mem differences
  are 1.3912404/0.00690001 for x weighting and 0.3332987/0.00610000 for epsilon
  weighting. This is close numerical agreement, not identical trajectories.
  No cause for the small differences was established; the portable bundle
  does not independently audit complete frozen evaluation manifests.
- Styled the 2x2 figure at 8.8 by 4.1 inches: N=512/2048 columns, log FID
  above linear f_mem, shared DejaVu Sans/STIX typography, thin axes, light
  SD bands, no markers, panel letters, and legends inside the lower panels.
  All 71 positive checkpoints appear on log training-step axes; step zero
  stays in the source data. Moved explanatory details into TXT/LaTeX captions.
- Ran `MPLCONFIGDIR=/tmp/theory_combined_mpl conda run --no-capture-output
  -n memcompgen python plotting/figures.py A4`. Draft build fb007ce22e62:
  outputs/paper_figures/A4.pdf, A4.svg, captions, receipt and source archive.
  Inspected the Poppler preview; saved previews/A4__fb007ce22e62.png. The
  LaTeX caption compiles as a one-page fragment without overfull warnings.
- Verification in local memcompgen: figure-catalog tests 10 passed; Ruff
  passes for both changed Python files. /tmp/check_a4.py independently checks
  30 distinct runs, 4320 source values, 1420 plotted means/SD intervals,
  exact old diagonal selection/summaries, legend bounds, artifact hashes,
  and unchanged source CSVs and existing figure exports. No training or
  model evaluation ran. Updated the plan, handoff, repo map and output README;
  preserved the manually listed M6 notebook export in the refreshed index.
  No approval inferred. No commit or push; codex/portable-plot-data, main unchanged.

## 2026-10-07 - Restore the F07 comparison and move A4's legend above the panels

- The researcher requested external legends with x-equivalent and
  epsilon-equivalent names, then clarified that A4 should resemble old F07.
  Inspected F07.pdf and restored its five conditions: x/x, eps/eps, v/v,
  native-v/v with (1-t)^2, and native-v/v with t^2. This replaces the earlier
  matched-row selection for the current A4, not its archived versions.
- Used a shared two-column legend above the panels, retaining the weight
  formulas. The 8.8 by 4.9-inch canvas preserves readable panel sizes and
  current paper fonts, thin axes, light bands, and panel letters. No large
  explanatory heading or markers. Log time displays 71 positive checkpoints;
  raw-pixel rho=0.3 and three-seed sample SD remain unchanged.
- Current draft is 601dad7c7933 in outputs/paper_figures/A4.pdf/.svg, with
  updated TXT/LaTeX captions and previews/A4__601dad7c7933.png. Intermediate
  matched draft c5e7ad733499 and original matched draft fb007ce22e62 remain
  archived. Old F07 exports and other figures are unchanged.
- Local memcompgen runtime check /tmp/check_a4_f07.py independently verifies
  all 4320 source values and 1420 mean/SD intervals for 30 distinct runs.
  The original diagonal selection and summaries are identical. The single
  legend fits above the panels; receipt hashes pass and source CSVs are
  unchanged. Ruff passes. Inspected the Poppler preview; the LaTeX caption
  compiles without overfull warnings. No training or evaluation ran.
- Updated the current plan, handoff, repo map, README and index to reflect
  the clarified comparison. Retained M6's manual notebook-export index row.
  No commit or push; branch codex/portable-plot-data and main unchanged.

## 2026-10-07 - Package main and appendix figure includes for Overleaf

- The researcher requested two LaTeX files with figure includes and captions,
  using a figures/ asset folder in Overleaf. Added results_figures.tex and
  appendix_figures.tex to outputs/paper_figures/. Both are document fragments
  with descriptive labels and no document class, section, or counter changes.
- Main results include M2--M6. M4 remains the exact native approved table,
  with its existing tab:robustness label and non-overriding color fallbacks.
  M6 uses one column; M2/M3/M5 span the text width. M1 remains the existing
  setup figure. Captions state metrics, thresholds, seed counts, and visual
  keys; M6 uses the notebook's FAST=False counts of 40 linear and 20 RFN seeds.
- Appendix includes A1, A1_DIT, A2, A3_DIT, A4, A5 and A5_TRAJECTORIES.
  It uses the latest F07-style A4 and current caption fragments, including
  A2/A5's short options. M3-SSCD and M3-DIT are complete commented optional
  blocks, preserving their unassigned status. No missing A3_UNET reference
  or excluded notebook diagnostic/log-log figure was inserted.
- Created overleaf_figures.zip with both fragments, 13 unchanged PDFs under
  figures/, and README upload/preamble instructions. The M4 preview PDF is
  not needed because its table is embedded. Verified all ZIP asset hashes,
  12 unique active labels, and the exact embedded M4 table body.
- Validation: pdfTeX 1.40.29 compiled a two-column results wrapper and a
  one-column appendix at 6.75-inch text width (9 pages), plus both optional
  matrices (1 page). No missing files, oversized floats, or overfull boxes.
  Rendered with Poppler and inspected every figure/table page. This checks
  the fragments, not final placement in the unavailable Overleaf manuscript.
  Initial compile from the repo root could not resolve fragment paths; the
  intended project-root layout compiled successfully after setting its cwd.
- Packaging checks ran in local memcompgen; no data, PDF export, caption
  source file, plotting code, or experiment was changed. Updated the output
  README and repo map. No commit or push; codex/portable-plot-data, main unchanged.

## 2026-10-07 - Reconcile the supplied results fragment with current figure exports

- Added outputs/paper_figures/results_figures_reconciled.tex as a separate
  alternative. Preserved the supplied active float order: M2 trajectories,
  M3 prediction/loss matrix, M5 timescales, and the native M4 robustness table.
  Preserved all five supplied active section/figure/table labels and the
  appendix paragraph. Retained M6 from the complete main-figure fragment at
  the end, with a note to move it beside the theory discussion as needed.
- Clarified that repository M identifiers do not specify manuscript figure
  numbers: the timescale plot previously called Figure 4 is asset M5; M4 is
  the table. Replaced legacy PDF paths with figures/M2.pdf, M3.pdf and M5.pdf,
  and removed trim settings intended for earlier exports.
- Corrected M2's caption to cover all eight panels and M3's caption to specify
  raw-pixel rho=0.3 and sample SD. Kept M5's distinct historical rho=0.5,
  first-1% onset, standard errors, and capped unobserved onsets explicit.
  The M4 table is byte-identical to both the supplied and canonical tables.
- Validation: local memcompgen checked all six labels are unique and all
  referenced assets exist. pdfTeX 1.40.29 compiled a two-column wrapper twice
  at 6.75-inch text width, producing four pages. Poppler rendered all pages;
  visual inspection found no clipping or overlapping content. No errors,
  missing assets, oversized floats, or overfull boxes; two underfull caption
  lines remain. Final float placement depends on the Overleaf manuscript.
- SHA-256 checks preserved all 75 pre-existing top-level paper-figure files
  and the supplied attachment, including both earlier LaTeX fragments, the
  ZIP, PDFs, and caption sources. No plotting, training, or evaluation ran.
  Updated the repo map. No commit or push; codex/portable-plot-data, main unchanged.

## 2026-11-19 - Make plain technical English the first communication priority

- The researcher requested priority 1 for plain, simple technical English.
- Added the first communication section to root AGENTS.md. It covers replies,
  notes, documentation, and code comments. It requires a check before sending.
- Checked the added text and Git whitespace. No code or experiment changed.
- This rule applies to this repository. No account settings changed.


## 2026-11-19 - Preserve the local paper work for rebuttal

- The researcher approved a GitHub push, a fixed paper-file snapshot, and
  a separate codex/rebuttal branch for new work.
- Backed up all 40 changed files and 342 paper files under
  outputs/handoffs/rebuttal_backup_20261119T115801Z/. Verified ZIP contents
  against the recorded SHA-256 hashes before staging any files.
- Reviewed the pending code, notebook, figures, tests, and documentation.
  Preserved notebook bytes and saved outputs. No experiment or figure ran.
- Local memcompgen: 28 focused plotting tests passed in 11.36 seconds.
  Ruff passed for all 19 pending Python files. All 19 files parsed, all ten
  notebook code cells compiled, and all 24 plot-data CSV hashes and sizes
  matched their manifests. No credential markers were found in pending text.
- Notebook schema validation completed with an existing missing-cell-ID
  warning. No notebook normalization or execution was performed.
- All 47 current paper artifacts matched their build-record hashes. All
  13 current build archives exist. This does not validate saved SGD results
  or change any figure approval.
- Fetched origin before committing. Prepared the existing plotting handoff
  and repository map for the fixed snapshot and separate Berzelius checkout.
- SSH authentication failed before any remote command ran. The cluster
  checkout, jobs, and environment remain unverified. No compute was launched.


## 2026-11-19 - Save the fixed paper-file snapshot

- Created paper_snapshots/2026-11-19/ from the verified local backup.
  The archive preserves all 342 paper files, including captions,
  Overleaf fragments and package, previews, references, and build versions.
- Recorded source code commit 8c43c8662d7ade6468e8df5730f5bae5ad7096a1, the unchanged theory
  notebook hash, each file size and SHA-256, and the archive hash.
- Verified every ZIP member against the manifest in local memcompgen.
  SHA256SUMS covers the ZIP and manifest. The ZIP is 14433932 bytes.
- The snapshot describes the local workspace, not the submitted manuscript.
  Existing review states, numerical inputs, and source output files remain
  unchanged. No notebook, figure, training, or evaluation was rerun.


## 2026-11-19 - Prepare the separate Berzelius rebuttal checkout

- Pushed preservation commits 8c43c86 and 35c2d8a to
  codex/portable-plot-data. Created and pushed codex/rebuttal at 35c2d8a.
  GitHub confirmed both branches. Main remains at 0cb30dd.
- After the researcher established SSH login, inspected the cluster checkout,
  its instructions and latest journals, and the scheduler queue. No jobs were
  running or pending for x_moalj at the check.
- The old mem_param_clean checkout remains on codex/reproducible-runners at
  e5bf94b with additional evaluation work. Backed up all 71 pending files,
  their hashes, Git status, and the worktree patch without changing them.
  Backup: mem_param_rebuttal/outputs/rebuttal_setup/legacy_checkout_backup/.
  Every saved source hash and the old commit were checked again after setup.
- Cloned GitHub codex/rebuttal into the new sibling directory
  /proj/rep-learning-robotics/users/x_moalj/project_mem_par/mem_param_rebuttal.
  It initially matched 35c2d8a exactly. All 24 plot-data CSV hashes and sizes,
  and all 342 archived paper-file hashes, matched on Berzelius.
- Loaded Mambaforge and activated mem_param_env. Its research libraries are
  Python 3.10.20, PyTorch 2.5.1.post303, NumPy 2.2.6, Matplotlib 3.10.8,
  nbformat 5.10.4, ipykernel 7.2.0, and JupyterLab 4.5.6.
- Pytest was missing. Created a checkout-local .venv with system site packages
  and installed pytest 9.1.1, iniconfig 2.3.1, and pluggy 1.7.0 there.
  The shared Conda environment was not changed. All 28 focused plotting tests
  passed in 4.97 seconds with PYTHONPATH pointing to the new checkout.
- Registered the mem_param_rebuttal kernel under .venv/share/jupyter/.
  Its interpreter and PYTHONPATH point to the new checkout. Environment
  records are under outputs/rebuttal_setup/. Updated the plotting handoff
  with activation, imports, tests, paths, and the separate old-code boundary.
- No training, checkpoint evaluation, paper plot, or experiment was run.
  No scheduler job was submitted, resumed, cancelled, or changed.

- Copied the old-checkout source backup and environment report to the laptop
  under outputs/handoffs/berzelius_rebuttal_setup_2026-11-19/. Verified all
  71 copied source-file hashes. These additional cluster changes remain
  separate from the GitHub source and the new rebuttal checkout.

- A fresh mem_param_rebuttal notebook kernel started in notebooks/theory/,
  imported mem_param_src from the new checkout, and used .venv/bin/python.
  The check passed in 44.78 seconds and the kernel was shut down. An initial
  30-second import timeout was too short; the repeated check allowed 150
  seconds. No source change was needed. Kernel evidence is saved in
  outputs/rebuttal_setup/kernel_check.json on Berzelius.


## 2026-11-19 - Combine Berzelius source with local plotting and analysis

- The researcher approved using the full Berzelius source as the base and
  adding the local plotting and analysis files. This covers the repository,
  including older studies, not only the 234 accepted production evaluations.
- Backed up the complete source and Git history before merging. Berzelius
  has 403 source files, including ten ignored patch backups. The laptop has
  370 source files. Both backups exist on both machines under
  consolidation_2026-11-19/, with file hashes and Git-history bundles.
  Full file restoration and cloned-history fsck checks passed. Transfer
  checksums passed on both machines. No experiment tensors were archived.
- Created mem_param_consolidation as a separate source-only test checkout.
  Commit 5a93b16 preserves all 71 pending Berzelius files. The original
  mem_param_clean checkout and its artifact directories were left in place.
- Combined the local branch at 08c07a3. Resolved five conflicts: both journals,
  the plotting handoff, the repository map, and the plotting grid. Preserved
  both journal histories and the Berzelius run-status table. Kept the local
  plotting grid and all other local plotting, theory, and test additions.
- SHA-256 checks passed for all 290 Berzelius source/config/job files,
  including ignored source patch copies. All 36 local plotting/analysis/test
  changes match their original file hashes. Every original source path
  remains present. All 24 plot-data CSV hashes and sizes also match.
- Cluster check: Mambaforge mem_param_env with the isolated pytest 9.1.1
  interpreter, Python 3.10.20, and CPU-only fixtures. All 191 tests and 59
  subtests passed in 80.72 seconds. Three existing torchvision/LPIPS warnings
  remain. Tests wrote only new temporary outputs in the test checkout.
- One existing blank line at the end of tests/test_evaluation_noise.py was
  reported when preserving the old source. It was left unchanged. The merge
  introduces no whitespace-check errors relative to that preserved source.
- Test reports and source checks are under
  mem_param_consolidation/outputs/consolidation_tests/. Updated the handoff
  to keep mem_param_clean as the main cluster directory and preserve all
  existing artifact paths. No model, data, result, or paper export was changed.
- No scheduler job was submitted, resumed, cancelled, or changed.


## 2026-11-19 - Verify the combined source on the laptop

- Fetched the combined source commit f49b778 directly from the isolated
  Berzelius test checkout. Tested it in a separate local Git worktree.
- All 284 active Berzelius source/config/job files match the source backup
  byte for byte. Ignored patch backups remain in the verified archives and
  original directories. No training, sampler, metric, config, or job source
  was changed during integration.
- Local memcompgen: Python 3.10.19, PyTorch 2.9.1, NumPy 2.2.6,
  Matplotlib 3.10.8, pytest 9.1.1, Pillow 11.3.0, and Ruff 0.14.8.
  The full suite passed 191 tests and 59 subtests in 15.22 seconds.
- The first local test run had one failure in a figure-bundle fixture.
  macOS gives tempfile a /var path, while saved summaries resolve it to
  /private/var. Confirmed that path alias mismatch. The failed test and then
  the full suite passed with TMPDIR set to a real /private/tmp directory.
  Research code and test code remained unchanged. Updated the handoff with
  the reproducible command. Existing Pillow/torchvision warnings remain.
- All 99 shell job files passed bash syntax checks; none was executed.
  Fifty-four changed Python files pass Ruff. The unchanged Berzelius
  plotting/weighting/quick_plot.py has one existing E402 import-order warning.
  It was preserved rather than edited during the source merge.
- Local test reports are saved under outputs/handoffs/
  consolidation_2026-11-19/validation/. Cluster reports remain under
  mem_param_consolidation/outputs/consolidation_tests/.
- The publication step updates codex/rebuttal only. Main and historical
  branches remain available. Model, data, noise, reference, generated-result,
  and historical frozen-job directories are outside this source update.
