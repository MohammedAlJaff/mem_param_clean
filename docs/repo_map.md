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
```

This tree is intentionally small. Add a new folder only when a reviewed file
class needs it.

Use `TODO.txt` for unfinished work. Use `MIGRATION_JOURNAL.md` for decisions,
checks, and completed work so the two files do not become competing backlogs.
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
- Reviewed notebooks go under `notebooks/`.
- Baselines and third-party code go under `external/` only if retained.
- Generated outputs do not enter the clean repo by default.

Boundary rules:

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
