"""Render one loss-weighting figure from cached metric tables."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mem_param_src.runners.weighting_trajectory_plots import (
    CALIBRATED_F_MEM,
    WEIGHTING_COLORS,
    WEIGHTING_LABELS,
    WEIGHTINGS,
    _plot_panel,
    _plot_specification_groups,
    _select_series,
    _spec,
)

DEFAULT_ROOT = Path(
    "outputs/evals/celeba32_vv_weighting_full_10000/20261003T134718Z_2109890/aggregate"
)
DEFAULT_TABLE_DIR = DEFAULT_ROOT / "seed_trajectories" / "plot_tables"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "quick_figures"
SPACE_TO_METRIC = {
    "raw_pixels": "f_mem_raw_pixels",
    "sscd": "f_mem_sscd",
    "lpips": "f_mem_lpips",
    "dinov2": "f_mem_dinov2",
}


def _load_metrics(table_dir, metrics):
    rows = []
    for metric in sorted(set(metrics)):
        path = Path(table_dir) / f"{metric}.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["n_train"] = int(row["n_train"])
                row["step"] = int(row["step"])
                for key in ("seed_0", "seed_1", "seed_2", "mean", "std"):
                    row[key] = float(row[key])
                rows.append(row)
    return rows


def _specifications(args):
    quality, heldout, _, _, primary = _plot_specification_groups()
    if args.figure == "quality":
        return quality
    if args.figure == "losses":
        return heldout
    if args.figure == "overview":
        return primary
    metric = args.metric
    if args.figure == "memorization":
        metric = SPACE_TO_METRIC[args.space]
    if metric.startswith("f_mem"):
        label = dict(CALIBRATED_F_MEM).get(metric, metric)
        if args.rho is not None:
            return [
                _spec(
                    metric,
                    f"{label} f_mem: rho={args.rho:g}",
                    "f_mem",
                    kind="f_mem",
                    rho=args.rho,
                )
            ]
        return [
            _spec(
                metric,
                f"{label} f_mem: alpha={args.alpha:g}",
                "f_mem",
                kind="f_mem",
                alpha=args.alpha,
            )
        ]
    return [_spec(metric, metric.replace("_", " ").title(), metric, log_y=True)]


def _save_trajectory(rows, specifications, args, output_path):
    count = len(specifications)
    columns = min(3, count)
    panel_rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(5.0 * columns, 3.8 * panel_rows),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for axis, specification in zip(flat_axes, specifications):
        _plot_panel(axis, rows, args.n_train, specification)
    for axis in flat_axes[count:]:
        axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(f"CelebA32 v/v weighting, N={args.n_train:,}", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def _save_final(rows, specification, args, output_path):
    selected = _select_series(rows, args.n_train, specification["series"])
    final_rows = [selected[weighting][-1] for weighting in WEIGHTINGS]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    positions = list(range(len(WEIGHTINGS)))
    colors = [WEIGHTING_COLORS[key] for key in WEIGHTINGS]
    for position, row, color in zip(positions, final_rows, colors):
        axis.errorbar(
            position,
            row["mean"],
            yerr=row["std"],
            fmt="none",
            ecolor=color,
            capsize=5,
        )
        axis.scatter(
            [position] * 3,
            [row[f"seed_{seed}"] for seed in range(3)],
            color=color,
            alpha=0.55,
        )
        axis.scatter(position, row["mean"], color=color, marker="D", s=65, zorder=3)
    axis.set_xticks(positions, [WEIGHTING_LABELS[key] for key in WEIGHTINGS])
    axis.set_ylabel(specification["ylabel"])
    axis.set_title(f"{specification['title']} at step {final_rows[0]['step']:,}")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=args.dpi)
    plt.close(figure)


def main(argv=None, *, default_figure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=("metric", "quality", "memorization", "losses", "overview", "final"),
        default=default_figure or "metric",
    )
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-train", type=int, choices=(512, 2048), default=512)
    parser.add_argument("--metric", default="inception_fid")
    parser.add_argument("--space", choices=tuple(SPACE_TO_METRIC), default="dinov2")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    if default_figure is not None:
        args.figure = default_figure

    started = time.perf_counter()
    specifications = _specifications(args)
    metrics = [specification["series"][0] for specification in specifications]
    rows = _load_metrics(args.table_dir, metrics)
    if args.output is None:
        detail = ""
        if args.figure == "memorization":
            detail = f"_{args.space}"
        elif args.figure in {"metric", "final"}:
            detail = f"_{args.metric}"
        name = f"n{args.n_train}_{args.figure}{detail}.png"
        args.output = DEFAULT_OUTPUT_DIR / name
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.figure == "final":
        _save_final(rows, specifications[0], args, args.output)
    else:
        _save_trajectory(rows, specifications, args, args.output)
    elapsed = time.perf_counter() - started
    print(f"Wrote {args.output} in {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
