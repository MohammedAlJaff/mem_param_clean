"""Image comparison grids and training curves from saved monitoring records."""

import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torchvision.utils import make_grid

from mem_param_src.runners.artifacts import read_json


def save_training_comparison_grid(train_images, samples, path, step):
    """Show fixed training examples above generated samples at one checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = min(8, max(len(train_images), len(samples)))
    rows = [math.ceil(len(images) / nrow) for images in (train_images, samples)]
    figure = Figure(figsize=(nrow * 0.8, sum(rows) * 0.8 + 1.2), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, gridspec_kw={"height_ratios": rows})
    for axis, images, label in zip(
        axes,
        (train_images, samples),
        ("Training examples (fixed)", f"Generated samples | step {int(step):,}"),
    ):
        # Both panels use the same fixed display range; raw samples stay unchanged.
        grid = make_grid(
            images.detach().cpu(), nrow=nrow, normalize=True, value_range=(-1.0, 1.0)
        )
        axis.imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
        axis.set_title(label, fontsize=11, loc="left")
        axis.set_axis_off()
    figure.savefig(path, dpi=160)
    return path


def save_monitor_curves(run_dir):
    """Refresh loss and projected f_mem figures from records already on disk."""
    run_dir = Path(run_dir)
    plot_dir = run_dir / "monitoring" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    history = read_json(run_dir / "train_history.json")
    metrics = [
        read_json(path)
        for path in sorted((run_dir / "monitoring").glob("f_mem_step_*.json"))
    ]
    paths = {}
    for key, records, title, ylabel in (
        ("loss", history, "Training loss", "Batch loss"),
        ("f_mem", metrics, "Projected-pixel memorization", "f_mem"),
    ):
        if not records:
            continue
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [r["step"] for r in records],
            [r[key] for r in records],
            marker="o" if key == "f_mem" else None,
            markersize=3,
            linewidth=1.3,
        )
        if key == "f_mem":
            axis.set_ylim(-0.02, 1.02)
            record = records[-1]
            title += f" | rho={record['rho']:g}, n={record['n_samples']:,}, N={record['N']:,}"
        axis.set(title=title, xlabel="Training step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        paths[key] = plot_dir / f"{key}.png"
        figure.savefig(paths[key], dpi=160)
    return paths
