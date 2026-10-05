"""Plot the CelebA32 prediction/loss matrix as f_mem trajectories."""

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PLOT_DATA_ROOT = Path("plot_data/celeba32_unet_3x3")
DEFAULT_FIGURE_ROOT = Path("outputs/figures/celeba32_unet_prediction_loss_3x3")
TARGETS = ("x", "eps", "v")
N_TRAINS = (512, 1024, 2048, 4096)
TARGET_LABELS = {
    "x": r"$x$",
    "eps": r"$\epsilon$",
    "v": r"$v$",
}
N_TRAIN_COLORS = {
    512: "#0072B2",
    1024: "#009E73",
    2048: "#E69F00",
    4096: "#D55E00",
}
EXPECTED_CHECKPOINT_COUNT = 72
TABLE_FIELDS = (
    "n_train",
    "prediction_target",
    "loss_target",
    "rho",
    "step",
    "seed_0",
    "seed_1",
    "seed_2",
    "mean",
    "std",
)


def _selected_runs(plot_data_root, n_trains):
    """Load and validate the requested three-seed U-Net matrices."""
    path = Path(plot_data_root) / "runs.csv"
    runs = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            n_train = int(row["n_train"])
            if n_train not in n_trains:
                continue
            key = (
                n_train,
                row["prediction_space"],
                row["loss_space"],
                int(row["seed"]),
            )
            if key in runs:
                raise ValueError(f"Duplicate run for {key}: {row['run_id']}")
            runs[key] = row["run_id"]

    expected = {
        (n_train, prediction, loss, seed)
        for n_train in n_trains
        for prediction in TARGETS
        for loss in TARGETS
        for seed in range(3)
    }
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(f"Incomplete U-Net matrix; missing={missing}, extra={extra}")
    return runs


def _load_series(plot_data_root, runs, rho):
    """Load every requested raw-pixel f_mem trajectory in one table pass."""
    run_keys = {run_id: key for key, run_id in runs.items()}
    series_by_key = {key: {} for key in runs}
    path = Path(plot_data_root) / "evaluation_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = run_keys.get(row["run_id"])
            if key is None or row["metric"] != "f_mem_raw_pixels":
                continue
            # An empty alpha identifies a fixed raw-pixel rho threshold.
            if row["alpha"] or not row["rho"]:
                continue
            if not math.isclose(float(row["rho"]), rho, abs_tol=1e-12):
                continue
            step = int(row["step"])
            value = float(row["value"])
            series = series_by_key[key]
            if step in series:
                raise ValueError(f"Duplicate step {step} for {row['run_id']}")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Invalid f_mem={value} at step {step} for {row['run_id']}"
                )
            series[step] = value

    for key, series in series_by_key.items():
        if len(series) != EXPECTED_CHECKPOINT_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CHECKPOINT_COUNT} rho={rho:g} rows for {key}, "
                f"found {len(series)}"
            )
        if min(series) != 0 or max(series) != 1_000_000:
            raise ValueError(
                f"Unexpected checkpoint range for {key}: {min(series)} to {max(series)}"
            )
    return series_by_key


def _aggregate(plot_data_root, n_trains, rho):
    """Aggregate three seeds for every requested prediction/loss cell."""
    runs = _selected_runs(plot_data_root, n_trains)
    per_run = _load_series(plot_data_root, runs, rho)

    rows = []
    common_steps = None
    for n_train in n_trains:
        for prediction in TARGETS:
            for loss in TARGETS:
                seed_series = [
                    per_run[(n_train, prediction, loss, seed)] for seed in range(3)
                ]
                steps = sorted(seed_series[0])
                if any(sorted(series) != steps for series in seed_series[1:]):
                    raise ValueError(
                        f"Seed checkpoint grids differ for "
                        f"N={n_train}, {prediction}/{loss}"
                    )
                if common_steps is None:
                    common_steps = steps
                elif steps != common_steps:
                    raise ValueError(
                        f"Cell checkpoint grid differs for "
                        f"N={n_train}, {prediction}/{loss}"
                    )
                for step in steps:
                    values = [series[step] for series in seed_series]
                    rows.append(
                        {
                            "n_train": n_train,
                            "prediction_target": prediction,
                            "loss_target": loss,
                            "rho": rho,
                            "step": step,
                            "seed_0": values[0],
                            "seed_1": values[1],
                            "seed_2": values[2],
                            "mean": statistics.fmean(values),
                            "std": statistics.stdev(values),
                        }
                    )
    return rows


def _write_table(path, rows):
    """Write the compact plotting table beside the scratch figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _cell_rows(rows, n_train, prediction, loss):
    """Select one ordered training-size and prediction/loss trajectory."""
    return sorted(
        (
            row
            for row in rows
            if row["n_train"] == n_train
            and row["prediction_target"] == prediction
            and row["loss_target"] == loss
        ),
        key=lambda row: row["step"],
    )


def _plot_grid(path, rows, n_trains, rho, dpi):
    """Draw all selected training sizes in the shared-scale 3x3 grid."""
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(10.5, 8.0),
        sharex=True,
        sharey=True,
    )
    upper_values = []

    for row_index, prediction in enumerate(TARGETS):
        for column_index, loss in enumerate(TARGETS):
            axis = axes[row_index, column_index]
            for n_train in n_trains:
                selected = _cell_rows(rows, n_train, prediction, loss)
                if not selected:
                    raise ValueError(
                        f"No trajectory for N={n_train}, {prediction}/{loss}"
                    )
                steps = [row["step"] for row in selected]
                means = [row["mean"] for row in selected]
                lower = [max(0.0, row["mean"] - row["std"]) for row in selected]
                upper = [min(1.0, row["mean"] + row["std"]) for row in selected]
                upper_values.extend(upper)
                color = N_TRAIN_COLORS[n_train]

                axis.plot(
                    steps,
                    means,
                    color=color,
                    linewidth=1.65,
                    label=rf"$N_{{\mathrm{{train}}}}={n_train:,}$",
                )
                axis.fill_between(
                    steps,
                    lower,
                    upper,
                    color=color,
                    alpha=0.10,
                )

            axis.grid(alpha=0.22)
            axis.set_xscale("symlog", linthresh=1_000)
            if row_index == 0:
                axis.set_title(f"Loss target {TARGET_LABELS[loss]}")
            if column_index == 0:
                axis.set_ylabel(
                    f"Predict {TARGET_LABELS[prediction]}\n"
                    r"$f_{\mathrm{mem}}$"
                )
            if row_index == 2:
                axis.set_xlabel("Training step")

    y_max = max(0.05, min(1.0, max(upper_values) * 1.08))
    for axis in axes.flat:
        axis.set_ylim(0.0, y_max)
        axis.set_xticks((0, 1_000, 10_000, 100_000, 1_000_000))
        axis.set_xticklabels(("0", "1k", "10k", "100k", "1M"))

    figure.suptitle(
        "CelebA32 U-Net prediction/loss matrix\n"
        rf"Raw-pixel $f_{{\mathrm{{mem}}}}$ at $\rho={rho:g}$",
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(n_trains),
        frameon=False,
    )
    figure.text(
        0.5,
        0.875,
        "Lines: three-seed means; bands: ±1 sample standard deviation",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.85))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-data-root",
        type=Path,
        default=DEFAULT_PLOT_DATA_ROOT,
    )
    parser.add_argument(
        "--n-train",
        type=int,
        choices=N_TRAINS,
        nargs="+",
        default=N_TRAINS,
        metavar="N",
    )
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-output", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    n_trains = tuple(dict.fromkeys(args.n_train))
    rows = _aggregate(args.plot_data_root, n_trains, args.rho)

    output_path = args.output
    if output_path is None:
        rho_label = f"{args.rho:g}".replace(".", "p")
        if n_trains == N_TRAINS:
            size_label = "all_n_train"
        else:
            size_label = "n" + "_".join(str(value) for value in n_trains)
        output_path = DEFAULT_FIGURE_ROOT / (
            f"{size_label}__fmem_raw_pixels__rho_{rho_label}__seed_mean_sd.png"
        )
    output_path = output_path.expanduser().resolve()
    table_path = args.table_output
    if table_path is None:
        table_path = output_path.with_suffix(".csv")
    table_path = table_path.expanduser().resolve()

    _write_table(table_path, rows)
    _plot_grid(output_path, rows, n_trains, args.rho, args.dpi)
    elapsed = time.perf_counter() - started
    print(
        f"Wrote {output_path} and {table_path} "
        f"from {len(rows)} aggregate rows in {elapsed:.2f} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
