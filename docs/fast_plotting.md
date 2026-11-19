# Fast plotting workflow

## Purpose

Use this workflow when metrics already exist and the next task is figure design.
It keeps plotting separate from sampling, encoding, nearest-neighbor search, and
metric computation. A normal figure edit should take seconds.

## Data flow

```text
accepted evaluation outputs
        |
        v
one immutable long-form aggregate CSV
        |  prepare once
        v
small per-metric CSV tables plus INDEX.json
        |  read only the requested metrics
        v
one scratch PNG
        |  collect after review
        v
outputs/figures/<experiment>/
```

The evaluation output is the scientific record. The per-metric tables are a
plotting cache. A figure is a view of those tables.

## Required aggregate schema

Write one row for each experiment condition, training step, metric, and
threshold. Keep identifiers as columns. Do not encode conditions only in file
names.

The weighting experiment uses these columns:

- `n_train`: training-set size.
- `weighting`: stable machine-readable condition name.
- `weighting_label`: display label.
- `metric`: stable metric name.
- `threshold_kind` and `threshold`: optional memorization threshold.
- `step`: optimizer step.
- `seed_0`, `seed_1`, and `seed_2`: individual seed values.
- `mean` and `std`: seed mean and sample standard deviation.
- `minimum` and `maximum`: range checks and optional plots.

New experiment blocks can add condition columns such as prediction target, loss
target, model, dataset, resolution, or capacity. Keep the metric and step
columns consistent across blocks.

## One-time preparation

Run the table builder only after the aggregate passes its completeness checks.
For the weighting experiment, use:

```bash
/home/x_moalj/.conda/envs/mem_param_env/bin/python \
  plotting/weighting/prepare_tables.py
```

This command writes one CSV per metric and an `INDEX.json` file. The index
records the source path, source hash, row count, table hashes, and table row
counts. Rebuild the tables only when the accepted aggregate changes.

## Fast figure iteration

Render one figure from the cached tables:

```bash
/home/x_moalj/.conda/envs/mem_param_env/bin/python \
  plotting/weighting/plot_memorization.py \
  --space dinov2 --alpha 0.01 --n-train 2048 \
  --output /tmp/weighting_dinov2_n2048.png
```

Expected output:

```text
Wrote /tmp/weighting_dinov2_n2048.png in <seconds> seconds
```

Use these small entrypoints:

- `plot_metric.py`: one selected metric trajectory.
- `plot_quality.py`: both quality trajectories.
- `plot_memorization.py`: one memorization space and threshold.
- `plot_losses.py`: all held-out-loss trajectories.
- `plot_overview.py`: the primary multi-panel trajectory figure.
- `plot_final.py`: one final-checkpoint comparison with individual seeds.

Write experimental style changes to `/tmp` or `quick_figures/`. When a figure
is approved, collect it under a flat, descriptive directory:

```bash
/home/x_moalj/.conda/envs/mem_param_env/bin/python \
  plotting/weighting/collect_figures.py
```

The current destination is
`outputs/figures/celeba32_vv_weighting_experiment/`.

## Performance contract

A quick plotting command must:

- read only the required CSV table or tables;
- write only the requested figure;
- avoid checkpoints, models, sample tensors, and feature tensors;
- avoid sampling, encoding, neighbor search, and metric computation;
- avoid regenerating unrelated plots;
- complete in less than three seconds on the login node for the current tables.

If a plot takes minutes, stop it and inspect file access. It is probably using
an evaluation path instead of the plotting cache.

## PNG and PDF policy

Use PNG for rapid inspection, notebooks, messages, and the flat figure
directory. Generate PDF only for a selected paper figure. Do not create both
formats during every style edit.

## Adding another experiment block

Use the same pattern:

1. Validate all run, seed, and checkpoint identities.
2. Write one immutable long-form aggregate CSV.
3. Split it into per-metric tables once.
4. Add small scripts that render one figure type per command.
5. Keep experiment colors, labels, and panel definitions in one module.
6. Test table completeness, seed aggregation, thresholds, and output opening.
7. Record the aggregate hash and figure command in the experiment README.

Plot scripts may change labels, layout, colors, and scale. They must not change
scientific values. Any data transformation belongs in the validated aggregate
builder and requires a new aggregate hash.
