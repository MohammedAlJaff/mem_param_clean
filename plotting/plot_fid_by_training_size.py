"""Plot accepted DiT, MNIST, and CelebA64 quality trajectories by training size."""

import csv
import math
import statistics
from pathlib import Path

from matplotlib.lines import Line2D

from celeba32_3x3.plot_f_mem_grid import (
    N_TRAIN_LINESTYLES, N_TRAIN_MARKERS, TARGET_COLORS, TARGET_LABELS, TARGETS,
    plt, set_panel_width, target_color,
)
from paper_style import TARGET_PAIR_LABELS

EXPERIMENTS = {
    "celeba32_dit_3x3": {
        "title": "CelebA32 Small-DiT", "dataset": "celeba32_grey",
        "model_arch": "dit", "image_size": 32, "image_channels": 1,
        "sizes": (512, 1024, 2048, 4096), "seeds": (0,), "matrix": True,
        "metric": "inception_fid", "metric_label": "Inception FID",
    },
    "mnist32_diagonal": {
        "title": "MNIST32 U-Net", "dataset": "mnist",
        "model_arch": "unet", "image_size": 32, "image_channels": 1,
        "sizes": (512, 2048), "seeds": (0, 1, 2), "matrix": False,
        "metric": "mnist_fd_jointae32_v1", "metric_label": "JointAE FD",
    },
    "mnist64_diagonal": {
        "title": "MNIST64 U-Net", "dataset": "mnist",
        "model_arch": "unet", "image_size": 64, "image_channels": 1,
        "sizes": (512, 2048), "seeds": (0, 1, 2), "matrix": False,
        "metric": "mnist_fd_jointae32_v1", "metric_label": "JointAE FD",
    },
    "celeba64_rgb_diagonal": {
        "title": "CelebA64 RGB U-Net", "dataset": "celeba32_grey",
        "model_arch": "unet", "image_size": 64, "image_channels": 3,
        "sizes": (512, 2048), "seeds": (0, 1, 2), "matrix": False,
        "metric": "inception_fid", "metric_label": "Inception FID",
    },
}


def load_records(experiment, spec):
    """Join quality metrics to run conditions and require the declared coverage."""
    root = Path("plot_data") / experiment
    with (root / "runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    cells = [(p, loss) for p in TARGETS for loss in TARGETS if spec["matrix"] or p == loss]
    expected = {(n, p, loss, seed) for n in spec["sizes"] for p, loss in cells for seed in spec["seeds"]}
    conditions, by_id = {}, {}
    fixed = {
        "dataset": spec["dataset"], "model_arch": spec["model_arch"],
        "image_size": str(spec["image_size"]), "image_channels": str(spec["image_channels"]),
        "loss_weighting": "unit", "evaluation_tier": "paper",
        "training_steps": "1000000", "source_final_step": "1000000",
        "evaluation_sample_count": "10000", "source_state": "trained",
    }
    for run in runs:
        if any(run[field] != value for field, value in fixed.items()):
            raise ValueError(f"Unexpected run settings: {run['run_id']}")
        key = (int(run["n_train"]), run["prediction_space"], run["loss_space"], int(run["seed"]))
        if key in conditions or run["run_id"] in by_id:
            raise ValueError(f"Duplicate run: {run['run_id']}")
        conditions[key] = run["run_id"]
        by_id[run["run_id"]] = run
    if set(conditions) != expected:
        raise ValueError(f"Unexpected size/target/seed coverage for {experiment}")

    per_run = {run_id: {} for run_id in by_id}
    with (root / "evaluation_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] != spec["metric"]:
                continue
            if row["space"] or row["alpha"] or row["rho"] or row["sample_count"] != "10000":
                raise ValueError(f"Unexpected quality-metric settings: {row['run_id']}")
            values = per_run[row["run_id"]]
            step, value = int(row["step"]), float(row["value"])
            if step in values or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Duplicate or nonpositive log-scale value: {row['run_id']}, {step}")
            values[step] = value
    common_steps = sorted(next(iter(per_run.values())))
    for run_id, values in per_run.items():
        if len(values) != int(by_id[run_id]["evaluation_checkpoint_count"]):
            raise ValueError(f"Incomplete trajectory: {run_id}")
        if sorted(values) != common_steps or min(values) != 0 or max(values) != 1_000_000:
            raise ValueError(f"Unmatched checkpoint schedule: {run_id}")

    records = []
    for (n, prediction, loss, seed), run_id in sorted(conditions.items()):
        for step, value in sorted(per_run[run_id].items()):
            records.append({
                "experiment": experiment, "n_train": n,
                "prediction_space": prediction, "loss_space": loss,
                "seed": seed, "run_id": run_id, "metric": spec["metric"],
                "step": step, "value": value, "sample_count": 10000,
            })
    return records


def summarize(records):
    """Compute arithmetic means and sample SD; single-seed SD remains absent."""
    groups = {}
    fields = ("experiment", "n_train", "prediction_space", "loss_space", "metric", "step")
    for row in records:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row["value"])
    rows = []
    for key, values in groups.items():
        rows.append({
            **dict(zip(fields, key)), "n_seeds": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else "",
        })
    return rows


def plot_trajectories(spec, records, summary, output):
    """Use target colors, with shades and markers for one or two sizes."""
    nrows = 3 if spec["matrix"] else 1
    figure, axes = plt.subplots(nrows, 3, figsize=(12, 7 if nrows == 3 else 3.8),
                                sharex=True, sharey=True, squeeze=False)
    bounds = [r["value"] for r in records]
    for row in summary:
        if row["std"] != "":
            bounds.extend((max(min(bounds), row["mean"] - row["std"]), row["mean"] + row["std"]))
    y_min, y_max = min(bounds) / 1.2, max(bounds) * 1.2
    cells = [(p, loss) for p in TARGETS for loss in TARGETS if spec["matrix"] or p == loss]
    for axis, (prediction, loss) in zip(axes.flat, cells):
        for n_train in spec["sizes"]:
            color = target_color(prediction, n_train, spec["sizes"])
            linestyle = "-" if len(spec["sizes"]) <= 2 else N_TRAIN_LINESTYLES[n_train]
            marker = N_TRAIN_MARKERS[n_train] if len(spec["sizes"]) <= 2 else None
            values = [r for r in summary if (r["n_train"], r["prediction_space"], r["loss_space"]) == (n_train, prediction, loss)]
            values.sort(key=lambda row: row["step"])
            steps = [r["step"] for r in values]
            if len(spec["seeds"]) > 1:
                for seed in spec["seeds"]:
                    seed_rows = [r for r in records if (r["n_train"], r["prediction_space"], r["loss_space"], r["seed"]) == (n_train, prediction, loss, seed)]
                    seed_rows.sort(key=lambda row: row["step"])
                    axis.plot(steps, [r["value"] for r in seed_rows], color=color,
                              linestyle=linestyle, alpha=0.24, linewidth=0.8)
                # Clip only the displayed lower band at the positive log-axis floor.
                axis.fill_between(steps, [max(y_min, r["mean"] - r["std"]) for r in values],
                                  [r["mean"] + r["std"] for r in values], color=color, alpha=0.10)
            axis.plot(steps, [r["mean"] for r in values], color=color, linestyle=linestyle, linewidth=2,
                      marker=marker, markevery=max(1, len(steps) // 10), markersize=4,
                      label=rf"$N_{{\mathrm{{train}}}}={n_train:,}$")
        axis.set_yscale("log")
        axis.set_ylim(y_min, y_max)
        axis.set_xscale("symlog", linthresh=1000)
        axis.set_xticks((0, 1000, 10000, 100000, 1000000), ("0", "1k", "10k", "100k", "1M"))
        axis.set_xlim(0, 1_000_000)
        axis.grid(alpha=0.2)
        if not spec["matrix"]:
            axis.set_title(TARGET_PAIR_LABELS[prediction],
                           color=TARGET_COLORS[prediction])
    if spec["matrix"]:
        for j, target in enumerate(TARGETS):
            axes[0, j].set_title(f"Loss: {TARGET_LABELS[target]}")
            axes[j, 0].set_ylabel(f"Prediction: {TARGET_LABELS[target]}", color=TARGET_COLORS[target])
    for axis in axes[-1]:
        axis.set_xlabel("Training step")
    figure.supylabel(f"{spec['metric_label']} (log scale)", x=0.01)
    figure.suptitle(f"{spec['title']}: quality trajectories by training size", y=0.985,
                   fontsize=11)
    caption = ("Seed 0 only; no across-seed uncertainty estimate."
               if len(spec["seeds"]) == 1 else
               "Three-seed means ± sample SD; faint lines show individual seeds.")
    figure.text(0.5, 0.944 if nrows == 3 else 0.905, caption, ha="center", fontsize=10)
    handles = []
    for n in spec["sizes"]:
        shade = ""
        if len(spec["sizes"]) == 2:
            shade = " (lighter)" if n == min(spec["sizes"]) else " (darker)"
        handles.append(Line2D(
            [], [], color="#999999" if shade == " (lighter)" else "black",
            linestyle="-" if len(spec["sizes"]) <= 2 else N_TRAIN_LINESTYLES[n],
            marker=N_TRAIN_MARKERS[n] if len(spec["sizes"]) <= 2 else None,
            label=rf"$N_{{\mathrm{{train}}}}={n:,}$" + shade,
        ))
    figure.legend(handles=handles, loc="upper center", ncol=len(spec["sizes"]), frameon=False,
                  bbox_to_anchor=(0.5, 0.922 if nrows == 3 else 0.875))
    figure.subplots_adjust(left=0.13 if nrows == 3 else 0.09, right=0.98,
                           top=0.85 if nrows == 3 else 0.75,
                           bottom=0.07 if nrows == 3 else 0.13, hspace=0.13, wspace=0.09)
    set_panel_width(figure, axes[0, 0])
    for extension in ("pdf", "svg"):
        figure.savefig(output.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def main():
    for experiment, spec in EXPERIMENTS.items():
        records = load_records(experiment, spec)
        summary = summarize(records)
        folder = Path("outputs/figures") / experiment
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"all_n_train__{spec['metric']}__log_scale"
        for suffix, rows in (("per_seed", records), ("mean_sd", summary)):
            with output.with_name(f"{output.name}__{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        plot_trajectories(spec, records, summary, output)
        print(f"Wrote {output.resolve()}.[pdf,svg] and CSVs: {len(records)} seed values, {len(summary)} summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
