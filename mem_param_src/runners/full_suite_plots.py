"""Write diagnostic plots for a full checkpoint metric suite."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    tensor_sha256,
    write_json,
)

SAMPLE_GRID_STEPS = (0, 10_000, 100_000, 1_000_000)


def _table_rows(root):
    """Load combined metric rows with numeric fields converted on demand."""
    path = root / "combined" / "checkpoint_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_series(path, series, *, title, ylabel):
    """Write one line plot for a mapping of labels to ordered points."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for label, points in sorted(series.items()):
        points = sorted(points)
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker=".",
            linewidth=1.2,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if series:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _quality_plots(root, rows, plot_root):
    names = {
        "inception_fid",
        "dinov2_fd",
        "mnist_fd_jointae32_v1",
        "mnist_class_fd_jointae32_v1",
    }
    series = {}
    for row in rows:
        if row["metric"] in names:
            series.setdefault(row["metric"], []).append(
                (int(row["step"]), float(row["value"]))
            )
    path = plot_root / "quality_trajectories.png"
    _plot_series(
        path,
        series,
        title="Quality trajectories",
        ylabel="Distance",
    )
    return [path]


def _memorization_plots(root, rows, plot_root):
    del root
    grouped = {}
    for row in rows:
        if not row["metric"].startswith("f_mem"):
            continue
        space = row["space"] or row["metric"].removeprefix("f_mem_")
        if row["alpha"]:
            threshold = f"alpha={float(row['alpha']):g}"
        else:
            threshold = f"rho={float(row['rho']):g}"
        grouped.setdefault(space, {}).setdefault(threshold, []).append(
            (int(row["step"]), float(row["value"]))
        )
    paths = []
    for space, series in sorted(grouped.items()):
        path = plot_root / f"memorization_{space}.png"
        _plot_series(
            path,
            series,
            title=f"Memorization trajectory: {space}",
            ylabel="f_mem",
        )
        paths.append(path)
    return paths


def _loss_plot(rows, plot_root):
    series = {}
    for row in rows:
        if row["metric"].startswith("heldout_loss_"):
            label = row["metric"].removeprefix("heldout_loss_")
            series.setdefault(label, []).append((int(row["step"]), float(row["value"])))
    path = plot_root / "heldout_loss_trajectories.png"
    _plot_series(path, series, title="Held-out losses", ylabel="Loss")
    return [path]


def _diagnostic_plots(records, plot_root):
    raw_series = {
        "raw_min": [],
        "raw_max": [],
        "raw_rms": [],
        "outside_minus1_1_fraction": [],
    }
    runtime_series = {"sampling_seconds": [], "checkpoint_total_seconds": []}
    for record in records:
        step = int(record["step"])
        for key, points in raw_series.items():
            points.append((step, float(record["raw_sample_statistics"][key])))
        runtimes = record.get("stage_runtimes", {})
        for key, points in runtime_series.items():
            if key in runtimes:
                points.append((step, float(runtimes[key])))
    raw_path = plot_root / "raw_range_diagnostics.png"
    runtime_path = plot_root / "runtime_diagnostics.png"
    _plot_series(
        raw_path,
        raw_series,
        title="Raw sample diagnostics",
        ylabel="Value",
    )
    _plot_series(
        runtime_path,
        runtime_series,
        title="Checkpoint runtimes",
        ylabel="Seconds",
    )
    return [raw_path, runtime_path]


def _show_image(axis, image):
    """Show one model-range image on an axis."""
    image = image.detach().cpu().float().clamp(-1, 1).add(1).div(2)
    if image.shape[0] == 1:
        axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
    else:
        axis.imshow(image.permute(1, 2, 0).numpy())
    axis.axis("off")


def _sample_grids(root, checkpoints, plot_root):
    paths = []
    available = {int(step) for step, _ in checkpoints}
    for step in SAMPLE_GRID_STEPS:
        if step not in available:
            continue
        sample_path = root / "checkpoints" / f"step_{step:06d}" / "samples.pt"
        samples = load_torch(sample_path)
        count = min(64, len(samples))
        rows = (count + 7) // 8
        figure, axes = plt.subplots(rows, 8, figsize=(8, rows))
        flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for index, axis in enumerate(flat_axes):
            if index < count:
                _show_image(axis, samples[index])
            else:
                axis.axis("off")
        figure.suptitle(f"Samples at step {step:,}")
        figure.tight_layout()
        path = plot_root / f"samples_step_{step:06d}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _neighbor_spaces(protocol):
    spaces = list(protocol.get("memorization_spaces", ()))
    if "f_mem_raw_pixels" in protocol["metrics"] and "raw_pixels" not in spaces:
        spaces.insert(0, "raw_pixels")
    return spaces


def _neighbor_panels(root, bundle, protocol, checkpoints, plot_root):
    final_step = int(checkpoints[-1][0])
    step_root = root / "checkpoints" / f"step_{final_step:06d}"
    samples = load_torch(step_root / "samples.pt")
    training = bundle["train"].contiguous()
    paths = []
    for space in _neighbor_spaces(protocol):
        artifact_name = (
            "mnist_jointae.pt" if space == "mnist_jointae32" else f"{space}.pt"
        )
        neighbor_path = step_root / "nearest_neighbors" / artifact_name
        if not neighbor_path.is_file():
            raise FileNotFoundError(
                f"Missing nearest-neighbor artifact: {neighbor_path}"
            )
        neighbors = load_torch(neighbor_path)
        order = torch.argsort(neighbors["r"])[: min(8, len(samples))]
        figure, axes = plt.subplots(2, len(order), figsize=(2 * len(order), 4))
        if len(order) == 1:
            axes = axes.reshape(2, 1)
        for column, query_index in enumerate(order.tolist()):
            _show_image(axes[0, column], samples[query_index])
            train_index = int(neighbors["nn_idx"][query_index])
            _show_image(axes[1, column], training[train_index])
            axes[0, column].set_title(
                f"r={float(neighbors['r'][query_index]):.3f}",
                fontsize=8,
            )
        axes[0, 0].set_ylabel("Generated")
        axes[1, 0].set_ylabel("Nearest train")
        figure.suptitle(f"Final nearest neighbors: {space}")
        figure.tight_layout()
        path = plot_root / f"nearest_neighbors_{space}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def write_full_suite_plots(root, bundle, protocol, checkpoints):
    """Write all requested diagnostic plots with one strict cache identity."""
    root = Path(root)
    plot_root = root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "combined" / "summary.json"
    available_steps = {int(step) for step, _ in checkpoints}
    sample_steps = sorted(
        {int(step) for step in SAMPLE_GRID_STEPS if int(step) in available_steps}
        | {int(checkpoints[-1][0])}
    )
    sample_hashes = {
        str(step): tensor_sha256(
            load_torch(root / "checkpoints" / f"step_{step:06d}" / "samples.pt")
        )
        for step in sample_steps
    }
    identity = {
        "summary_sha256": file_sha256(summary_path),
        "sample_tensor_sha256s": sample_hashes,
        "training_tensor_sha256": tensor_sha256(bundle["train"]),
        "protocol": protocol,
        "sample_grid_steps": list(SAMPLE_GRID_STEPS),
    }
    manifest_path = plot_root / "plot_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("identity") == identity:
            files = manifest.get("files", {})
            if files and all(
                (root / name).is_file() and file_sha256(root / name) == expected
                for name, expected in files.items()
            ):
                return plot_root

    summary = read_json(summary_path)
    rows = _table_rows(root)
    paths = []
    paths.extend(_quality_plots(root, rows, plot_root))
    paths.extend(_memorization_plots(root, rows, plot_root))
    paths.extend(_loss_plot(rows, plot_root))
    paths.extend(_diagnostic_plots(summary["checkpoints"], plot_root))
    paths.extend(_sample_grids(root, checkpoints, plot_root))
    paths.extend(_neighbor_panels(root, bundle, protocol, checkpoints, plot_root))
    write_json(
        manifest_path,
        {
            "identity": identity,
            "files": {str(path.relative_to(root)): file_sha256(path) for path in paths},
        },
    )
    return plot_root
