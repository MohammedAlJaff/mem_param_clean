"""Aggregate and plot the complete v/v loss-weighting evaluation."""

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001

from mem_param_src.runners.artifacts import file_sha256, write_json


RUN_PATTERN = re.compile(r"^n(512|2048)_(xeq|unit|epseq)_s([012])$")
N_TRAINS = (512, 2048)
SEEDS = (0, 1, 2)
WEIGHTINGS = ("xeq", "unit", "epseq")
WEIGHTING_LABELS = {
    "xeq": r"$x$-equivalent: $(1-t)^2$",
    "unit": r"unit: $1$",
    "epseq": r"$\epsilon$-equivalent: $t^2$",
}
WEIGHTING_COLORS = {
    "xeq": "#0072B2",
    "unit": "#666666",
    "epseq": "#D55E00",
}
ALPHAS = (0.005, 0.01, 0.02, 0.05)
FIXED_RHOS = (0.3, 0.5, 0.7)
CALIBRATED_F_MEM = (
    ("f_mem_raw_pixels", "Raw pixels"),
    ("f_mem_sscd", "SSCD"),
    ("f_mem_lpips", "LPIPS"),
    ("f_mem_dinov2", "DINOv2"),
)
EXPECTED_ACCEPTANCE = {
    "accepted": True,
    "checkpoint_count": 72,
    "evaluation_tier": "paper",
    "publication_ready": True,
    "sample_count": 10_000,
}
AGGREGATE_FIELDS = (
    "n_train",
    "weighting",
    "weighting_label",
    "metric",
    "threshold_kind",
    "threshold",
    "step",
    "seed_0",
    "seed_1",
    "seed_2",
    "mean",
    "std",
    "minimum",
    "maximum",
)


def _utc_timestamp():
    """Return a stable UTC timestamp for the plot manifest."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_identity(name):
    """Parse one frozen result directory name."""
    match = RUN_PATTERN.fullmatch(name)
    if match is None:
        return None
    return {
        "n_train": int(match.group(1)),
        "weighting": match.group(2),
        "seed": int(match.group(3)),
    }


def _threshold_identity(row):
    """Return the configured threshold identity for one metric row."""
    if row["alpha"]:
        return "alpha", f"{float(row['alpha']):g}"
    if row["metric"] == "f_mem_raw_pixels" and row["rho"]:
        return "rho", f"{float(row['rho']):g}"
    return "", ""


def _series_key(metric, threshold_kind="", threshold=""):
    """Return one stable aggregate-series key."""
    return metric, threshold_kind, threshold


def _load_observations(evaluation_root):
    """Load and validate all 18 accepted result tables."""
    expected_runs = {
        (n_train, weighting, seed)
        for n_train in N_TRAINS
        for weighting in WEIGHTINGS
        for seed in SEEDS
    }
    found_runs = set()
    observations = defaultdict(dict)
    input_files = {}
    common_steps = None
    raw_row_count = 0

    for run_root in sorted(Path(evaluation_root).iterdir()):
        identity = _run_identity(run_root.name)
        if identity is None:
            continue
        run_key = (
            identity["n_train"],
            identity["weighting"],
            identity["seed"],
        )
        if run_key in found_runs:
            raise ValueError(f"Duplicate result directory for {run_key}")
        found_runs.add(run_key)

        accepted_path = run_root / "ACCEPTED.json"
        table_path = run_root / "combined" / "checkpoint_metrics.csv"
        if not accepted_path.is_file() or not table_path.is_file():
            raise FileNotFoundError(f"Incomplete accepted result: {run_root}")
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (accepted.get(key), expected)
            for key, expected in EXPECTED_ACCEPTANCE.items()
            if accepted.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Acceptance mismatch for {run_root.name}: {mismatches}")

        with table_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        steps = sorted({int(row["step"]) for row in rows})
        if len(steps) != EXPECTED_ACCEPTANCE["checkpoint_count"]:
            raise ValueError(f"Expected 72 steps in {table_path}, found {len(steps)}")
        if common_steps is None:
            common_steps = steps
        elif steps != common_steps:
            raise ValueError(f"Checkpoint step grid differs in {table_path}")

        seen = set()
        for row in rows:
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric in {table_path}: {row}")
            if row["metric"].startswith("f_mem") and not 0 <= value <= 1:
                raise ValueError(f"f_mem outside [0, 1] in {table_path}: {row}")
            threshold_kind, threshold = _threshold_identity(row)
            series = _series_key(row["metric"], threshold_kind, threshold)
            observation_key = (
                identity["n_train"],
                identity["weighting"],
                series,
                int(row["step"]),
            )
            row_key = (series, int(row["step"]))
            if row_key in seen:
                raise ValueError(f"Duplicate metric row in {table_path}: {row_key}")
            seen.add(row_key)
            observations[observation_key][identity["seed"]] = value
        raw_row_count += len(rows)
        input_files[str(table_path.relative_to(evaluation_root))] = file_sha256(
            table_path
        )
        input_files[str(accepted_path.relative_to(evaluation_root))] = file_sha256(
            accepted_path
        )

    if found_runs != expected_runs:
        missing = sorted(expected_runs - found_runs)
        extra = sorted(found_runs - expected_runs)
        raise ValueError(f"Result matrix mismatch. Missing={missing}, extra={extra}")
    return observations, common_steps, input_files, raw_row_count


def _aggregate_observations(observations):
    """Aggregate the three seed values for every metric point."""
    rows = []
    for (n_train, weighting, series, step), seed_values in sorted(observations.items()):
        if set(seed_values) != set(SEEDS):
            raise ValueError(
                f"Incomplete seeds for N={n_train}, {weighting}, {series}, "
                f"step={step}: {sorted(seed_values)}"
            )
        values = [seed_values[seed] for seed in SEEDS]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        metric, threshold_kind, threshold = series
        rows.append(
            {
                "n_train": n_train,
                "weighting": weighting,
                "weighting_label": WEIGHTING_LABELS[weighting],
                "metric": metric,
                "threshold_kind": threshold_kind,
                "threshold": threshold,
                "step": step,
                "seed_0": values[0],
                "seed_1": values[1],
                "seed_2": values[2],
                "mean": mean,
                "std": std,
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return rows


def _atomic_csv(path, rows):
    """Write the aggregate table atomically."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select_series(rows, n_train, series):
    """Select one aggregate series and validate all three weightings."""
    metric, threshold_kind, threshold = series
    selected = defaultdict(list)
    for row in rows:
        if row["n_train"] != n_train:
            continue
        if row["metric"] != metric:
            continue
        if row["threshold_kind"] != threshold_kind:
            continue
        if row["threshold"] != threshold:
            continue
        selected[row["weighting"]].append(row)
    if set(selected) != set(WEIGHTINGS):
        raise ValueError(f"Missing weighting in N={n_train}, series={series}")
    for weighting in WEIGHTINGS:
        selected[weighting].sort(key=lambda row: row["step"])
    return selected


def _plot_panel(axis, rows, n_train, specification):
    """Plot three seed-aggregated weighting curves on one panel."""
    series = specification["series"]
    selected = _select_series(rows, n_train, series)
    all_means = [
        row["mean"]
        for weighting in WEIGHTINGS
        for row in selected[weighting]
        if row["mean"] > 0
    ]
    positive_floor = min(all_means) * 0.25 if all_means else 1e-12
    for weighting in WEIGHTINGS:
        points = selected[weighting]
        steps = [row["step"] for row in points]
        means = [row["mean"] for row in points]
        lower = [row["mean"] - row["std"] for row in points]
        upper = [row["mean"] + row["std"] for row in points]
        if specification["kind"] == "f_mem":
            lower = [max(0, value) for value in lower]
            upper = [min(1, value) for value in upper]
        elif specification.get("log_y"):
            lower = [max(positive_floor, value) for value in lower]
        axis.plot(
            steps,
            means,
            color=WEIGHTING_COLORS[weighting],
            linewidth=1.8,
            label=WEIGHTING_LABELS[weighting],
        )
        axis.fill_between(
            steps,
            lower,
            upper,
            color=WEIGHTING_COLORS[weighting],
            alpha=0.18,
            linewidth=0,
        )
    axis.set_title(specification["title"])
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(specification["ylabel"])
    axis.set_xscale("symlog", linthresh=1_000)
    axis.set_xticks((0, 1_000, 10_000, 100_000, 1_000_000))
    axis.set_xticklabels(("0", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"))
    if specification.get("log_y"):
        axis.set_yscale("log")
    if specification["kind"] == "f_mem":
        axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)


def _write_plot(root, rows, n_train, name, specifications, *, columns):
    """Write one multi-panel PNG and PDF with a shared legend."""
    count = len(specifications)
    panel_rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(5.2 * columns, 4.0 * panel_rows),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for axis, specification in zip(flat_axes, specifications):
        _plot_panel(axis, rows, n_train, specification)
    for axis in flat_axes[count:]:
        axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    figure.suptitle(f"CelebA32 v/v loss weighting, N={n_train:,}", y=0.992)
    figure.text(
        0.5, 0.965, "Mean across three seeds; shaded band is ±1 SD", ha="center"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{name}.{suffix}"
        figure.savefig(path, dpi=180)
        paths.append(path)
    plt.close(figure)
    return paths


def _spec(metric, title, ylabel, *, kind="value", log_y=False, alpha=None, rho=None):
    """Build one plot-panel specification."""
    threshold_kind = ""
    threshold = ""
    if alpha is not None:
        threshold_kind = "alpha"
        threshold = f"{alpha:g}"
    elif rho is not None:
        threshold_kind = "rho"
        threshold = f"{rho:g}"
    return {
        "series": _series_key(metric, threshold_kind, threshold),
        "title": title,
        "ylabel": ylabel,
        "kind": kind,
        "log_y": log_y,
    }


def _plot_specification_groups():
    """Return the complete and primary plot groups."""
    quality = [
        _spec("inception_fid", "Inception FID", "FID", log_y=True),
        _spec("dinov2_fd", "DINOv2 Fréchet distance", "Distance", log_y=True),
    ]
    heldout = [
        _spec(
            "heldout_loss_random_time",
            "Held-out loss: random time",
            "Loss",
            log_y=True,
        ),
        _spec(
            "heldout_loss_fixed_t_0.5",
            "Held-out loss: t=0.5",
            "Loss",
            log_y=True,
        ),
        _spec(
            "heldout_loss_fixed_t_0.75",
            "Held-out loss: t=0.75",
            "Loss",
            log_y=True,
        ),
    ]
    calibrated = {}
    for metric, label in CALIBRATED_F_MEM:
        calibrated[metric] = [
            _spec(
                metric,
                f"{label}: alpha={alpha:g}",
                "f_mem",
                kind="f_mem",
                alpha=alpha,
            )
            for alpha in ALPHAS
        ]
    raw_fixed = [
        _spec(
            "f_mem_raw_pixels",
            f"Raw pixels: rho={rho:g}",
            "f_mem",
            kind="f_mem",
            rho=rho,
        )
        for rho in FIXED_RHOS
    ]
    primary = (
        quality
        + [
            _spec(
                metric,
                f"{label} f_mem: alpha=0.01",
                "f_mem",
                kind="f_mem",
                alpha=0.01,
            )
            for metric, label in CALIBRATED_F_MEM
        ]
        + heldout
    )
    return quality, heldout, calibrated, raw_fixed, primary


def write_weighting_trajectory_plots(evaluation_root, output_root=None):
    """Write all seed-aggregated weighting trajectories and their manifest."""
    evaluation_root = Path(evaluation_root).expanduser().resolve()
    if output_root is None:
        output_root = evaluation_root / "aggregate" / "seed_trajectories"
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    observations, steps, input_files, raw_row_count = _load_observations(
        evaluation_root
    )
    aggregate_rows = _aggregate_observations(observations)
    table_path = output_root / "seed_aggregated_metrics.csv"
    _atomic_csv(table_path, aggregate_rows)

    quality, heldout, calibrated, raw_fixed, primary = _plot_specification_groups()
    plot_paths = []
    for n_train in N_TRAINS:
        n_root = output_root / f"n{n_train}"
        n_root.mkdir(parents=True, exist_ok=True)
        plot_paths.extend(
            _write_plot(
                n_root,
                aggregate_rows,
                n_train,
                "primary_overview",
                primary,
                columns=3,
            )
        )
        plot_paths.extend(
            _write_plot(
                n_root,
                aggregate_rows,
                n_train,
                "quality",
                quality,
                columns=2,
            )
        )
        plot_paths.extend(
            _write_plot(
                n_root,
                aggregate_rows,
                n_train,
                "heldout_losses",
                heldout,
                columns=3,
            )
        )
        for metric, _ in CALIBRATED_F_MEM:
            plot_paths.extend(
                _write_plot(
                    n_root,
                    aggregate_rows,
                    n_train,
                    f"memorization_{metric.removeprefix('f_mem_')}_calibrated",
                    calibrated[metric],
                    columns=2,
                )
            )
        plot_paths.extend(
            _write_plot(
                n_root,
                aggregate_rows,
                n_train,
                "memorization_raw_pixels_fixed_rho",
                raw_fixed,
                columns=3,
            )
        )

    output_files = [table_path] + plot_paths
    manifest = {
        "schema_version": 1,
        "created_at": _utc_timestamp(),
        "evaluation_root": str(evaluation_root),
        "output_root": str(output_root),
        "n_train_values": list(N_TRAINS),
        "weightings": {
            weighting: WEIGHTING_LABELS[weighting] for weighting in WEIGHTINGS
        },
        "seeds": list(SEEDS),
        "checkpoint_count": len(steps),
        "checkpoint_steps": steps,
        "raw_metric_rows": raw_row_count,
        "aggregate_rows": len(aggregate_rows),
        "aggregation": {
            "line": "arithmetic mean across seeds",
            "band": "plus or minus one sample standard deviation",
            "sample_standard_deviation_ddof": 1,
            "smoothing": False,
            "primary_calibrated_f_mem_alpha": 0.01,
            "x_axis": "symmetric log with linear threshold 1000",
        },
        "input_files": input_files,
        "output_files": {
            str(path.relative_to(output_root)): file_sha256(path)
            for path in output_files
        },
    }
    write_json(output_root / "plot_manifest.json", manifest)
    return manifest


def build_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-root")
    return parser


def main(argv=None):
    """Write weighting trajectories from one completed evaluation matrix."""
    args = build_parser().parse_args(argv)
    manifest = write_weighting_trajectory_plots(
        args.evaluation_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
