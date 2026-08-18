# Repository Map

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

- `mem_param_src/` is library code only.
- Job files can call live entrypoints, but must not own study logic.
- Notebooks are for reviewed analysis and probes, not core implementation.
- Plot code must be separated from metric computation.
- Image metric modules must reuse the generic nearest-neighbor implementation;
  they must not carry a second copy of `d1`, `d2`, `r`, or `f_mem` math.
- Checkpoint evaluation must read saved artifacts, not rebuild datasets.
