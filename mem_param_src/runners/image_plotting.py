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
    legacy_metrics = [
        read_json(path)
        for path in sorted((run_dir / "monitoring").glob("f_mem_step_*.json"))
    ]
    metric_groups = {}
    if legacy_metrics:
        metric_groups[""] = legacy_metrics
    profile_root = run_dir / "monitoring" / "f_mem"
    if profile_root.is_dir():
        for profile_dir in sorted(path for path in profile_root.iterdir() if path.is_dir()):
            records = [
                read_json(path) for path in sorted(profile_dir.glob("step_*.json"))
            ]
            if records:
                metric_groups[profile_dir.name] = records

    paths = {}
    if history:
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [record["step"] for record in history],
            [record["loss"] for record in history],
            markersize=3,
            linewidth=1.3,
        )
        axis.set(title="Training loss", xlabel="Training step", ylabel="Batch loss")
        axis.grid(alpha=0.2)
        paths["loss"] = plot_dir / "loss.png"
        figure.savefig(paths["loss"], dpi=160)

    if metric_groups:
        figure = Figure(figsize=(7, 4), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        for profile, records in metric_groups.items():
            axis.plot(
                [record["step"] for record in records],
                [record["f_mem"] for record in records],
                marker="o",
                markersize=3,
                linewidth=1.3,
                label=profile or None,
            )
        last_profile = next(reversed(metric_groups))
        record = metric_groups[last_profile][-1]
        title = (
            "Projected-pixel memorization"
            f" | rho={record['rho']:g}, n={record['n_samples']:,}, N={record['N']:,}"
        )
        axis.set(
            title=title,
            xlabel="Training step",
            ylabel="f_mem",
            ylim=(-0.02, 1.02),
        )
        axis.grid(alpha=0.2)
        if any(metric_groups):
            axis.legend()
        paths["f_mem"] = plot_dir / "f_mem.png"
        figure.savefig(paths["f_mem"], dpi=160)
    return paths
