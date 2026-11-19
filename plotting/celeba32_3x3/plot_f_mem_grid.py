"""Plot the CelebA32 prediction/loss matrix as FID and f_mem trajectories."""

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

# Keep direct-script commands able to find the shared plotting style.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_style import TARGET_LABELS as TARGET_LABELS, apply_style

apply_style()

PANEL_WIDTH_INCHES = 2.0
DEFAULT_PLOT_DATA_ROOT = Path("plot_data/celeba32_unet_3x3")
DEFAULT_FIGURE_ROOT = Path("outputs/figures/celeba32_unet_prediction_loss_3x3")
TARGETS = ("x", "eps", "v")
N_TRAINS = (512, 1024, 2048, 4096)
# Shared prediction palette; weighted-velocity variants add their own colors.
TARGET_COLORS = {"x": "#D62728", "eps": "#1F77B4", "v": "#2CA02C"}
N_TRAIN_LINESTYLES = {512: "-", 1024: "--", 2048: "-.", 4096: ":"}
# Markers identify size in dual-axis figures and one/two-size comparisons.
N_TRAIN_MARKERS = {512: "o", 1024: "s", 2048: "^", 4096: "D"}
# Explicit dark-to-light palettes for the joint grids, ordered by N_TRAINS.
GRID_SIZE_COLORS = {
    "x": ("#67000D", "#D62728", "#EF6548", "#FB9272"),
    "eps": ("#08306B", "#1F77B4", "#399BD5", "#82C8E5"),
    "v": ("#00441B", "#2CA02C", "#66B85A", "#AED581"),
}
EXPECTED_CHECKPOINT_COUNT = 72
METRICS = ("f_mem_raw_pixels", "inception_fid")
TABLE_FIELDS = (
    "n_train",
    "prediction_target",
    "loss_target",
    "metric",
    "rho",
    "step",
    "seed_0",
    "seed_1",
    "seed_2",
    "mean",
    "std",
)


def set_panel_width(figure, axis, width=PANEL_WIDTH_INCHES):
    """Apply the common or explicitly chosen axes width, keeping the height."""
    figure.set_figwidth(width / axis.get_position().width)


def size_color(color, n_train, n_trains):
    """Use a lighter condition color for the smaller of two training sizes."""
    if len(n_trains) == 2 and n_train == min(n_trains):
        return to_hex([0.65 * channel + 0.35 for channel in to_rgb(color)])
    return color


def target_color(prediction, n_train, n_trains):
    """Apply the shared size shade to a prediction-target color."""
    return size_color(TARGET_COLORS[prediction], n_train, n_trains)


def _selected_runs(plot_data_root, n_trains, model_arch="unet", seeds=(0, 1, 2)):
    """Load and validate matrix coverage for the requested model and seeds."""
    path = Path(plot_data_root) / "runs.csv"
    runs = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != "celeba32_grey" or row["model_arch"] != model_arch:
                continue
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
        for seed in seeds
    }
    if set(runs) != expected:
        missing = sorted(expected - set(runs))
        extra = sorted(set(runs) - expected)
        raise ValueError(f"Incomplete {model_arch} matrix; missing={missing}, extra={extra}")
    return runs


def _load_series(plot_data_root, runs, rho, metric, alpha=None):
    """Load one metric for the same accepted runs and checkpoint grid."""
    if metric not in ("inception_fid", "f_mem_raw_pixels", "f_mem_sscd"):
        raise ValueError(f"Unsupported metric: {metric}")
    if metric == "f_mem_sscd" and (alpha is None or rho is not None):
        raise ValueError("SSCD requires calibrated alpha, not a fixed rho.")
    run_keys = {run_id: key for key, run_id in runs.items()}
    series_by_key = {key: {} for key in runs}
    path = Path(plot_data_root) / "evaluation_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = run_keys.get(row["run_id"])
            if key is None or row["metric"] != metric:
                continue
            if metric == "f_mem_raw_pixels":
                # An empty alpha identifies a fixed raw-pixel rho threshold.
                if row["space"] != "raw_pixels" or row["alpha"] or not row["rho"]:
                    continue
                if not math.isclose(float(row["rho"]), rho, abs_tol=1e-12):
                    continue
            elif metric == "f_mem_sscd":
                # SSCD exports have blank space and a run-specific calibrated rho.
                if row["space"] or not row["alpha"] or not row["rho"]:
                    continue
                if not math.isclose(float(row["alpha"]), alpha, abs_tol=1e-12):
                    continue
            elif row["space"] or row["alpha"] or row["rho"]:
                # Inception FID has no memorization space or threshold fields.
                continue
            if row["sample_count"] != "10000":
                raise ValueError(f"Unexpected evaluation sample count: {row['run_id']}")
            step = int(row["step"])
            value = float(row["value"])
            series = series_by_key[key]
            if step in series:
                raise ValueError(f"Duplicate step {step} for {row['run_id']}")
            if (
                not math.isfinite(value)
                or value < 0.0
                or (metric.startswith("f_mem_") and value > 1.0)
            ):
                raise ValueError(
                    f"Invalid {metric}={value} at step {step} for {row['run_id']}"
                )
            series[step] = value

    for key, series in series_by_key.items():
        if len(series) != EXPECTED_CHECKPOINT_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_CHECKPOINT_COUNT} {metric} rows for {key}, "
                f"found {len(series)}"
            )
        if min(series) != 0 or max(series) != 1_000_000:
            raise ValueError(
                f"Unexpected checkpoint range for {key}: {min(series)} to {max(series)}"
            )
    return series_by_key


def _aggregate(plot_data_root, n_trains, rho, metric, model_arch="unet", seeds=(0, 1, 2)):
    """Aggregate a metric in every cell; one seed has no sample SD."""
    runs = _selected_runs(plot_data_root, n_trains, model_arch, seeds)
    per_run = _load_series(plot_data_root, runs, rho, metric)

    rows = []
    common_steps = None
    for n_train in n_trains:
        for prediction in TARGETS:
            for loss in TARGETS:
                seed_series = [
                    per_run[(n_train, prediction, loss, seed)] for seed in seeds
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
                            "metric": metric,
                            "rho": rho if metric == "f_mem_raw_pixels" else "",
                            "step": step,
                            **{f"seed_{seed}": value for seed, value in zip(seeds, values)},
                            "mean": statistics.fmean(values),
                            "std": statistics.stdev(values) if len(values) > 1 else None,
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


def _cell_rows(rows, n_train, prediction, loss, metric):
    """Select one ordered training-size and prediction/loss trajectory."""
    return sorted(
        (
            row
            for row in rows
            if row["n_train"] == n_train
            and row["prediction_target"] == prediction
            and row["loss_target"] == loss
            and row["metric"] == metric
        ),
        key=lambda row: row["step"],
    )


def _plot_grid(path, rows, n_trains, rho, dpi, model_label="CelebA32 U-Net"):
    """Overlay training sizes with dotted FID and solid raw-pixel f_mem."""
    figure, axes = plt.subplots(3, 3, figsize=(8.8, 6.6), sharex=True, sharey=True)
    for row_index, prediction in enumerate(TARGETS):
        for column_index, loss in enumerate(TARGETS):
            axis = axes[row_index, column_index]
            f_mem_axis = axis.twinx()
            for n_train in n_trains:
                size_index = N_TRAINS.index(n_train)
                color = GRID_SIZE_COLORS[prediction][size_index]
                metric_rows = {
                    metric: _cell_rows(rows, n_train, prediction, loss, metric)
                    for metric in METRICS
                }
                if [row["step"] for row in metric_rows[METRICS[0]]] != [
                        row["step"] for row in metric_rows[METRICS[1]]]:
                    raise ValueError("FID and f_mem checkpoint grids differ")
                for metric, metric_axis, linestyle in (
                    ("inception_fid", axis, ":"),
                    ("f_mem_raw_pixels", f_mem_axis, "-"),
                ):
                    # Omit step zero only from the logarithmic time display.
                    selected = [r for r in metric_rows[metric] if r["step"] > 0]
                    if not selected:
                        raise ValueError(f"Missing {metric}: N={n_train}, {prediction}/{loss}")
                    steps = [row["step"] for row in selected]
                    means = [row["mean"] for row in selected]
                    if metric == "inception_fid" and max(means) > 600:
                        raise ValueError("FID exceeds the shared U-Net/DiT display range.")
                    if selected[0]["std"] is not None:
                        lower = [max(0, row["mean"] - row["std"]) for row in selected]
                        upper = [row["mean"] + row["std"] for row in selected]
                        if metric == "f_mem_raw_pixels":
                            upper = [min(1, value) for value in upper]
                        elif max(upper) > 600:
                            raise ValueError("FID uncertainty exceeds the shared display range.")
                        metric_axis.fill_between(
                            steps, lower, upper, color=color,
                            alpha=0.06 if metric == "inception_fid" else 0.10, linewidth=0,
                        )
                    metric_axis.plot(
                        steps, means, color=color, linestyle=linestyle, linewidth=1.65,
                        dash_capstyle="round",
                    )

            axis.set_xscale("log")
            axis.set_xlim(250, 1_000_000)
            axis.set_xticks((1_000, 10_000, 100_000, 1_000_000))
            axis.set_ylim(0, 600)
            axis.set_yticks((0, 200, 400, 600))
            f_mem_axis.set_ylim(0, 1.03)
            f_mem_axis.set_yticks((0, 0.25, 0.5, 0.75, 1))
            axis.grid(which="major", color="0.87", linewidth=0.5, linestyle=":")
            axis.set_axisbelow(True)
            axis.tick_params(which="both", labelsize=8, length=2.5, width=0.6)
            f_mem_axis.tick_params(axis="y", labelright=column_index == 2,
                                   labelsize=8, length=2.5, width=0.6)
            f_mem_axis.text(0.035, 0.95, f"({chr(97 + 3 * row_index + column_index)})",
                            transform=axis.transAxes, ha="left", va="top", fontsize=9)
            for panel in (axis, f_mem_axis):
                for spine in panel.spines.values():
                    spine.set_linewidth(0.6)
            if row_index == 0:
                axis.set_title(f"Loss target {TARGET_LABELS[loss]}", pad=7)
            if column_index == 0:
                axis.set_ylabel(f"Predict {TARGET_LABELS[prediction]}\nFID",
                                color=TARGET_COLORS[prediction], fontsize=10)
            if column_index == 2:
                f_mem_axis.set_ylabel(r"$F_{\mathrm{mem}}$", fontsize=10)
            if row_index == 2:
                axis.set_xlabel(r"Training steps, $\tau$", fontsize=9)

    figure.legend(
        handles=[tuple(Line2D([], [], color=GRID_SIZE_COLORS[target][N_TRAINS.index(n)],
                              linestyle="-", linewidth=1.8)
                       for target in TARGETS) for n in n_trains],
        labels=[rf"$N={n}$" for n in n_trains],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, handlelength=6,
        loc="lower center", bbox_to_anchor=(0.5, 0.080), ncol=4,
        frameon=False, columnspacing=1.5, handletextpad=0.4,
    )
    figure.legend(
        handles=[Line2D([], [], color="0.25", linestyle=":", label="FID (left)"),
                 Line2D([], [], color="0.25", linestyle="-", label=r"$F_{\mathrm{mem}}$ (right)")],
        loc="lower center", bbox_to_anchor=(0.5, 0.044), ncol=2,
        frameon=False, handlelength=2.5, columnspacing=2,
    )
    seed_note = "seed 0" if all(row["std"] is None for row in rows) else "mean ± SD, 3 seeds"
    figure.text(0.5, 0.015, f"{model_label} · raw pixels, " + rf"$\rho={rho:g}$ · {seed_note}",
                ha="center", va="center", fontsize=8, color="0.35")
    figure.subplots_adjust(left=0.105, right=0.915, bottom=0.205, top=0.94,
                           hspace=0.16, wspace=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi)
    if path.suffix == ".pdf":
        figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def grid_caption(model_label, single_seed, latex=False):
    """Keep the shared reading instructions concise and coverage explicit."""
    uncertainty = ("Curves show seed 0; no across-seed uncertainty is available. "
                   if single_seed else "Curves and bands show mean +/- sample SD over three seeds. ")
    text = (
        f"Joint quality and memorization trajectories for {model_label} on CelebA32 grayscale. "
        "Rows vary the prediction target and columns the loss target, ordered x, epsilon, v. "
        "Dotted lines show Inception FID (left axis); solid lines show raw-pixel F_mem "
        "at rho=0.3 (right axis). Dark-to-light shades identify "
        "N=512, 1024, 2048, and 4096. "
        f"{uncertainty}All positive saved checkpoints are shown on a logarithmic training-step axis."
    )
    if latex:
        for plain, math_text in (
            ("x, epsilon, v", ", ".join(TARGET_LABELS[t] for t in TARGETS)),
            ("F_mem", r"$F_{\mathrm{mem}}$"), ("rho=0.3", r"$\rho=0.3$"),
            ("N=512, 1024, 2048, and 4096", r"$N=512,1024,2048,$ and $4096$"),
            ("+/-", r"$\pm$"),
        ):
            text = text.replace(plain, math_text)
    return text


def render(output, experiment="celeba32_unet_3x3"):
    """Build one current-data appendix grid without writing numerical exports."""
    model_arch, seeds, model_label = {
        "celeba32_unet_3x3": ("unet", (0, 1, 2), "U-Net"),
        "celeba32_dit_3x3": ("dit", (0,), "Small-DiT"),
    }[experiment]
    root = DEFAULT_PLOT_DATA_ROOT.parent / experiment
    rows = [row for metric in METRICS
            for row in _aggregate(root, N_TRAINS, 0.3, metric, model_arch, seeds)]
    _plot_grid(output.with_suffix(".pdf"), rows, N_TRAINS, 0.3, 180,
               model_label=f"CelebA32 {model_label}")
    for suffix, latex in ((".caption.txt", False), (".caption.tex", True)):
        output.with_suffix(suffix).write_text(
            grid_caption(model_label, len(seeds) == 1, latex) + "\n", encoding="utf-8",
        )


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
    rows = [
        row
        for metric in METRICS
        for row in _aggregate(args.plot_data_root, n_trains, args.rho, metric)
    ]

    output_path = args.output
    if output_path is None:
        rho_label = f"{args.rho:g}".replace(".", "p")
        if n_trains == N_TRAINS:
            size_label = "all_n_train"
        else:
            size_label = "n" + "_".join(str(value) for value in n_trains)
        output_path = DEFAULT_FIGURE_ROOT / (
            f"{size_label}__fid__fmem_raw_pixels__rho_{rho_label}__seed_mean_sd.pdf"
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
