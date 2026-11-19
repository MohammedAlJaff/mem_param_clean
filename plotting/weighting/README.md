# Fast loss-weighting plots

This directory contains small plotting commands for the CelebA32 native-velocity
loss-weighting experiment. Build the metric tables once. Then render only the
figure that you want to inspect or edit.

Quick plots do not change accepted evaluation artifacts.
The general workflow and performance contract are in
`docs/fast_plotting.md`.


## One-time table preparation

Run this command from the repository root:

```bash
/home/x_moalj/.conda/envs/mem_param_env/bin/python \
  plotting/weighting/prepare_tables.py
```

It splits the accepted 10,368-row aggregate into one CSV per metric. Every CSV
contains both train sizes, all three weightings, all three seed values, the
mean, the sample standard deviation, every threshold, and all 72 checkpoints.
Pass `--aggregate-csv` to use the same workflow for a later experiment.

## Small figure commands

Each command writes only one PNG by default.

- `plot_metric.py`: one selected metric trajectory.
- `plot_quality.py`: Inception FID and DINOv2 distance.
- `plot_memorization.py`: one memorization space and threshold.
- `plot_losses.py`: all three held-out-loss trajectories.
- `plot_overview.py`: the primary nine-panel trajectory figure.
- `plot_final.py`: one final-checkpoint metric with all seed points.

For example:

```bash
python plotting/weighting/plot_memorization.py \
  --space dinov2 --alpha 0.01 --n-train 2048
```

Raw pixels can instead use a fixed threshold, such as
`--space raw_pixels --rho 0.5`.

The default output directory is `outputs/evals/.../aggregate/quick_figures/`.
Use `--output /tmp/test.png` while changing styles. Use `--table-dir` for a
later experiment. These scripts do not scan checkpoints, load samples, encode
images, or regenerate unrelated plots.

## Easy figure directory

Collect the current PNG figures into one flat directory with descriptive names:

```bash
python plotting/weighting/collect_figures.py
```

The destination is `outputs/figures/celeba32_vv_weighting_experiment/`.
