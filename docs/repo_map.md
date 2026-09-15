# Repository Map

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

Current skeleton:

```text
mem_param_src/     reusable first-party package
external/          baseline and third-party code, if retained
notebooks/         reviewed notebooks, probes, and paper analysis
jobs/berzelius/    Slurm launchers for Berzelius, if retained
docs/              migration notes and project intent
TODO.txt           single operational backlog of unfinished work
RUN_JOURNAL.md    submitted compute runs and chronological status checks
```

This tree is intentionally small. Add a new folder only when a reviewed file
class needs it.

Use `TODO.txt` for unfinished work. Use `MIGRATION_JOURNAL.md` for decisions,
checks, and completed work. Use `RUN_JOURNAL.md` for every clean-repository
compute submission and its append-only status history. These files must not
become competing backlogs.
Use `docs/config_reference.md` for every active configuration field, its
meaning, current default, implementation status, and legacy runner guidance.

The user-requested `CRITICAL_EPS_EPS_SAMPLING_AUDIT_2026-09-09.txt` preserves
the current sampling audit and Berzelius handoff, including artifact hashes
and an unexecuted fixed-checkpoint GPU command. It records an open diagnosis;
`TODO.txt` remains the backlog and `MIGRATION_JOURNAL.md` the completed-work log.

Known file classes from the old repo:

- model and sampler code
- losses and metrics
- synthetic data loaders
- image data loaders
- trainers
- checkpoint replay and evaluation code
- artifact, logging, and tracking code
- visualization and figure-generation code
- Berzelius job files
- experiment sweep definitions
- tests
- notebooks
- baseline and third-party code
- generated outputs, which stay untracked

Probable destinations:

- Library code goes under `mem_param_src/`.
- Dataset definitions and loader logic live under `mem_param_src/datasets/`.
- Generic metric math stays directly under `mem_param_src/metrics/`.
- Image preprocessing and metric adapters live under
  `mem_param_src/metrics/image/`.
- Dataset-specific synthetic metrics live under
  `mem_param_src/metrics/synthetic/`.
- Berzelius launch files go under `jobs/berzelius/`.
- `RUN_JOURNAL.md` records every submitted job and each later status check.
- Reviewed notebooks go under `notebooks/`.
- Baselines and third-party code go under `external/` only if retained.
- Generated outputs do not enter the clean repo by default.

Boundary rules:

- Berzelius smoke jobs: `jobs/berzelius/eps_recovery_smoke/submit.sh` submits
  rows from `variants.tsv` through `run_array.sbatch`. The shared experiment
  config is `configs/runners/celeba32_eps_recovery_smoke.py`. Scripts call the
  existing image runner; they do not own model, loss, or sampler logic.
  `inspect_results.py` reads saved raw tensors and hashes without model calls.

- One-million-step checkpoint evaluation:
  `jobs/berzelius/eps_recovery_1m/eval/submit.sh` launches six epsilon-only
  tasks at the shared 100k, 390k, and 695k checkpoints. Each source uses its
  as-launched sampler. `prepare_fid_reference.py` first persists one canonical
  10,000-image CelebA validation tensor and verifies its saved 1,024-image
  training-artifact prefix. `evaluate_checkpoints.py` writes append-only metric
  rows under `outputs/evals/`; `combine_results.py` makes one comparison CSV
  and plot. The evaluator reads source checkpoints and the shared frozen FID
  reference without modifying source runs. `submit_references.sh` adds the saved
  `v/v` and `x/x` references to the same three-point 10k/10k comparison.
  See `jobs/berzelius/eps_recovery_1m/eval/README.md` for the data flow.

- Sampling: `mem_param_src/sampling.py` owns direct epsilon-to-velocity
  conversion and Heun's optional final Euler step. `mem_param_src/model.py`
  retains training losses/conversions and forwards `model.sample` arguments.
  `sampling.eps_to_v` and `sampling.heun_final_euler` select the two behaviors.
  Monitoring and saved-checkpoint evaluation pass these controls explicitly.
  Older saved evaluations retain historical settings unless overridden.
  See the runnable API shape and comparison controls in the README's
  "Direct epsilon conversion and final Euler" section.

- Image monitoring: `runners/image_train.py` reads `--f-mem-monitor` and
  `--f-mem-samples`. `runners/training.py` calls `runners/image_monitoring.py`
  at preview checkpoints. It samples once for grids and optional f_mem, using
  `metrics/image/projection.py` and shared `metrics/nearest_neighbors.py`.
  JSON records and the fixed projection stay in the run's `monitoring/` folder.
  Optional full generated tensors go to `monitoring/generated_samples/`.
  Post-training evaluation stores its optional checkpoint tensors in
  `evaluations/<name>/generated_samples/`.
  `runners/image_plotting.py` draws fixed training examples above generated
  samples in `monitoring/comparison_grids/`. It reads saved loss and f_mem
  records to refresh `monitoring/plots/` during training.
  `runners/tracking.py` sends scalar records and grids to W&B when enabled.

- `mem_param_src/` is library code only.
- Job files can call live entrypoints, but must not own study logic.
- Notebooks are for reviewed analysis and probes, not core implementation.
- Plot code must be separated from metric computation.
- Image metric modules must reuse the generic nearest-neighbor implementation;
  they must not carry a second copy of `d1`, `d2`, `r`, or `f_mem` math.
- Checkpoint evaluation must read saved artifacts, not rebuild datasets.
