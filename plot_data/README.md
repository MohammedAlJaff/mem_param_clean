# Plot data

Each subdirectory is one experiment with the same small relational layout:

- runs.csv: one row per accepted run and all condition labels.
- training_metrics.csv: saved training loss and monitoring trajectories.
- evaluation_metrics.csv: all accepted checkpoint-level evaluation metrics.
- evaluation_diagnostics.csv: raw-range, runtime, and calibration diagnostics.
- manifest.json: row counts and SHA-256 hashes.

Join each metric table to runs.csv with run_id.
No checkpoints or large tensors are stored here.

From the repository root, reproduce the current CelebA32 U-Net figure with:

```bash
python plotting/celeba32_3x3/plot_f_mem_grid.py
```

The PNG and its compact plotted table are written under
`outputs/figures/celeba32_unet_prediction_loss_3x3/`.
