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
Rename the active `denominator_epsilon` field to `safe_div_denom_eps`. Apply a
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
`[0.01, 0.99]`. Set `safe_div_denom_eps = 0.01`. Compare exact division with
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
`training.use_safe_division`. Inference reads `sampling.use_safe_division`. Do
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
`safe_div_denom_eps = 0.01`. The Flowers sanity run uses `v` prediction and `v`
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
This changes only `sampling.use_safe_division` to `True`. Use the saved data
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
first test: fixed Flowers102 replay with safe_div_denom_eps = 0.05
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

- Set training.use_safe_division and sampling.use_safe_division to False in
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
