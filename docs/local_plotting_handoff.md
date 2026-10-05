# Local plotting handoff

## Goal

Use the accepted scalar training and evaluation records on a normal laptop
without copying checkpoints, generated sample banks, feature matrices, datasets,
or the cluster output tree.

The portable source of truth is `plot_data/`. Rendered figures remain scratch
outputs under `outputs/figures/` and are intentionally ignored by Git.

## Pull and environment

From the existing local clone:

```bash
git fetch origin
git switch codex/reproducible-runners
git pull --ff-only origin codex/reproducible-runners
```

The current CelebA32 plotting command needs Python and Matplotlib. It does not
need PyTorch, CUDA, the datasets, or any project checkpoint:

```bash
python -m pip install matplotlib
python plotting/celeba32_3x3/plot_f_mem_grid.py
```

Expected outputs:

```text
outputs/figures/celeba32_unet_prediction_loss_3x3/
  all_n_train__fmem_raw_pixels__rho_0p3__seed_mean_sd.png
  all_n_train__fmem_raw_pixels__rho_0p3__seed_mean_sd.csv
```

The figure is a 3x3 prediction-target by loss-target grid. Every panel overlays
`N_train={512,1024,2048,4096}`. Lines are three-seed means. Bands are plus or
minus one sample standard deviation. The plotted metric is raw-pixel `f_mem`
at fixed `rho=0.3` across all 72 saved checkpoints.

## Portable data layout

```text
plot_data/
  INDEX.json
  README.md
  celeba32_unet_3x3/
  celeba32_dit_3x3/
  celeba32_vv_weighting/
  celeba64_rgb_diagonal/
  mnist32_diagonal/
  mnist64_diagonal/
```

Every experiment directory has the same files:

- `runs.csv`: one row per accepted run and all experimental conditions.
- `training_metrics.csv`: training loss and saved monitoring trajectories.
- `evaluation_metrics.csv`: accepted checkpoint-level scientific metrics.
- `evaluation_diagnostics.csv`: raw-range, runtime, and threshold diagnostics.
- `manifest.json`: row counts, byte sizes, and SHA-256 hashes.
- `README.md`: the local schema reminder.

Join any metric table to `runs.csv` with `run_id`. Do not infer experimental
conditions by parsing filenames when the corresponding column exists in
`runs.csv`.

The `source_run_path` and `evaluation_result_path` columns are provenance
records from the cluster. They are not expected to exist on the laptop and are
not required for plotting.

## Current coverage

The committed bundle contains 216 accepted evaluations:

| Experiment | Accepted runs | Evaluation rows |
| --- | ---: | ---: |
| CelebA32 U-Net 3x3 | 108 | 186,624 |
| CelebA32 Small-DiT 3x3 | 36 | 62,208 |
| CelebA32 v/v weighting | 18 | 31,104 |
| CelebA64 RGB diagonal | 18 | 32,400 |
| MNIST32 diagonal | 18 | 5,760 |
| MNIST64 diagonal | 18 | 5,760 |

The final 18 seed-1 Small-DiT evaluations were still running when this bundle
was exported. Therefore `celeba32_dit_3x3` currently contains the 36 accepted
seed-0 runs only. Do not treat the missing seed-1 cells as zeroes and do not
silently mix incomplete seed coverage into aggregate plots.

## Rules for local Codex

1. Read `plot_data/README.md`, `plot_data/INDEX.json`, and the relevant
   experiment's `runs.csv` before writing a plot.
2. Use only `plot_data/` for numerical inputs. Do not require
   `outputs/evals/`, checkpoints, tensors, models, or cluster paths.
3. Preserve individual seeds until the requested aggregation is explicit.
   When showing a three-seed mean, use sample standard deviation.
4. Keep prediction space, loss space, training size, model, dataset, metric
   space, threshold kind, `alpha`, and `rho` explicit in filtering code.
5. Write scratch PNGs and their compact plotted CSV tables under
   `outputs/figures/<experiment>/`.
6. Do not edit exported CSVs manually. The exporter writes deterministic rows,
   so regenerated scientific-data changes remain reviewable in normal Git
   diffs.
7. Keep one plotting entrypoint focused on one figure family. A style edit
   should take seconds and must not trigger evaluation work.

## Exporter scope

`plotting/export_plot_data.py` is the cluster-side regeneration command. Its
input roots intentionally point at retained accepted evaluation outputs. A
laptop without those outputs should consume the committed `plot_data/`
bundle, not rerun the exporter.

The exporter includes only accepted, publication-ready runs. It exports numeric
training history, monitoring records, evaluation metrics, and scalar
diagnostics. It excludes checkpoints and all large reproducible tensors.

## Suggested first prompt for local Codex

> Read `docs/local_plotting_handoff.md` and `plot_data/README.md`. Work only
> from the committed `plot_data/` CSVs. First reproduce the existing CelebA32
> U-Net 3x3 f_mem figure. Then help me iterate on one requested figure at a time.
> Preserve seed-level values, show exactly which metric and threshold are used,
> save figures under `outputs/figures/`, and never depend on cluster outputs
> or checkpoints.

## Validation performed before push

- Exported all six experiment directories successfully.
- Verified 216 accepted run rows and the recorded per-experiment row counts.
- Verified every exported file is below 40 MB.
- Reran the exporter and obtained byte-identical hashes for all 38 files.
- Passed Ruff and Python compilation for both plotting scripts.
- Regenerated the 3x3 figure from only `plot_data/` in 1.84 seconds.
- Passed Git whitespace checks.
