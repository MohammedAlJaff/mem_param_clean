"""Compare memorization across representations, thresholds, and training."""

import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from celeba32_3x3.plot_f_mem_grid import (
    N_TRAINS, N_TRAIN_LINESTYLES, TARGET_COLORS, TARGETS,
    _load_series, _selected_runs,
)
from paper_style import TARGET_PAIR_LABELS, apply_style
from plot_diagonal_trajectories import COLUMN_TARGETS, curve_color
from plot_fid_by_training_size import load_records
from plot_robustness_table import dataset_settings


EXPERIMENT = "celeba32_unet_3x3"
RAW = "f_mem_raw_pixels"
METRICS = (RAW, "f_mem_sscd", "f_mem_lpips", "f_mem_dinov2")
RHOS = (0.3, 0.5, 0.7)
ALPHA = 0.01
TRAJECTORY_FLOOR = 1e-5


def select_readouts(quality, evaluation):
    """Pair every threshold/representation with the same run and selected step."""
    best = {}
    for row in quality:
        if row["prediction_space"] != row["loss_space"]:
            continue
        old = best.get(row["run_id"])
        if old is None or (row["value"], row["step"]) < (old["value"], old["step"]):
            best[row["run_id"]] = row
    selected = {run_id: {} for run_id in best}
    for row in evaluation:
        run_id, metric = row["run_id"], row["metric"]
        if run_id not in best or metric not in METRICS:
            continue
        if int(row["step"]) != best[run_id]["step"]:
            continue
        if not row["alpha"] and metric == RAW and float(row["rho"]) in RHOS:
            key = (metric, "rho", float(row["rho"]))
        elif row["alpha"] and float(row["alpha"]) == ALPHA:
            key = (metric, "alpha", ALPHA)
        else:
            continue
        value = float(row["value"])
        expected_space = "raw_pixels" if metric == RAW else ""
        if (row["space"] != expected_space or not row["rho"]
                or row["sample_count"] != "10000"
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid memorization record: {run_id}, {key}")
        if key in selected[run_id]:
            raise ValueError(f"Duplicate memorization readout: {run_id}, {key}")
        selected[run_id][key] = value
    expected = {(RAW, "rho", rho) for rho in RHOS} | {(m, "alpha", ALPHA) for m in METRICS}
    for run_id, values in selected.items():
        if set(values) != expected:
            raise ValueError(f"Incomplete paired readouts at best FID: {run_id}")
        if [values[(RAW, "rho", r)] for r in RHOS] != sorted(values[(RAW, "rho", r)] for r in RHOS):
            raise ValueError(f"Nonmonotonic fixed-rho sweep: {run_id}")
    return [{**row, "readouts": selected[run_id]} for run_id, row in best.items()]


def load_readouts():
    quality = load_records(EXPERIMENT, dataset_settings(EXPERIMENT))
    with (Path("plot_data") / EXPERIMENT / "evaluation_metrics.csv").open(newline="") as handle:
        rows = select_readouts(quality, csv.DictReader(handle))
    for n in N_TRAINS:
        for target in TARGETS:
            seeds = sorted(r["seed"] for r in rows
                           if r["n_train"] == n and r["prediction_space"] == target)
            if seeds != [0, 1, 2]:
                raise ValueError(f"Unexpected seeds for N={n}, target={target}")
    return rows


def caption(mode, latex=False):
    comparison = (
        "Representation comparison on CelebA32 grayscale U-Net. Pixels, SSCD "
        "image-copy-detection descriptors, LPIPS perceptual features, and DINOv2 "
        "visual features each use their exported threshold calibrated to a 1% "
        "flagged rate on held-out images (alpha=0.01). The pixel points here use "
        "calibrated thresholds, not the headline fixed rho=0.3. "
        if mode == "representation" else
        "Raw-pixel threshold sensitivity on CelebA32 grayscale U-Net. "
        "The nearest-neighbour ratio threshold varies over rho=0.3, 0.5, and 0.7; "
        "rho=0.3 is the current headline setting and rho=0.5 is the workshop setting. "
    )
    text = comparison + (
        "Each panel gives F_mem (%) at N=512, 1024, 2048, or 4096. Each seed's "
        "lowest-FID checkpoint is selected once (earlier step for exact ties) "
        "and held fixed across all readouts. Points and error bars show mean +/- "
        "sample SD over three seeds, using 10,000 generated samples per checkpoint. "
        "Red, blue, and green identify x/x, epsilon/epsilon, and v/v. "
        "Lines guide comparisons between settings. These panels compare aggregate "
        "rates and parameterization ordering, not agreement on individual images. "
        "They do not recompute the workshop timescales or test its 1% onset cutoff."
    )
    if latex:
        for plain, math_text in (
            ("F_mem (%)", r"$F_{\mathrm{mem}}$ (\%)"),
            ("N=512, 1024, 2048, or 4096", r"$N=512,1024,2048,$ or $4096$"),
            ("rho=0.3, 0.5, and 0.7", r"$\rho\in\{0.3,0.5,0.7\}$"),
            ("rho=0.3", r"$\rho=0.3$"), ("rho=0.5", r"$\rho=0.5$"),
            ("alpha=0.01", r"$\alpha=0.01$"), ("1%", r"1\%"), ("+/-", r"$\pm$"),
            ("x/x", TARGET_PAIR_LABELS["x"]),
            ("epsilon/epsilon", TARGET_PAIR_LABELS["eps"]), ("v/v", TARGET_PAIR_LABELS["v"]),
        ):
            text = text.replace(plain, math_text)
    return text


def load_trajectories():
    """Keep all seeds and checkpoints when averaging each fixed-rho trajectory."""
    root = Path("plot_data") / EXPERIMENT
    runs = {key: run_id for key, run_id in _selected_runs(root, N_TRAINS).items()
            if key[1] == key[2]}
    series = {rho: _load_series(root, runs, rho, RAW) for rho in RHOS}
    steps = sorted(next(iter(series[RHOS[0]].values())))
    for rho, trajectories in series.items():
        for key, values in trajectories.items():
            if sorted(values) != steps:
                raise ValueError(f"Unmatched checkpoint grid: {rho}, {key}")
            if rho != RHOS[0]:
                previous = series[RHOS[RHOS.index(rho) - 1]][key]
                if any(previous[step] > values[step] for step in steps):
                    raise ValueError(f"Nonmonotonic fixed-rho trajectory: {key}")
    curves = {}
    for rho in RHOS:
        for target in COLUMN_TARGETS:
            for n in N_TRAINS:
                values = np.array([[series[rho][(n, target, target, seed)][step]
                                    for step in steps] for seed in (0, 1, 2)])
                curves[(rho, target, n)] = {
                    "steps": np.array(steps),
                    "mean": values.mean(axis=0),
                    "std": values.std(axis=0, ddof=1),
                }
    return series, curves


def trajectory_caption(latex=False):
    text = (
        "Raw-pixel memorization trajectories for CelebA32 grayscale U-Net. "
        "Rows show epsilon/epsilon, v/v, and x/x; columns show rho=0.3, 0.5, and 0.7. "
        "Increasing rho raises measured memorization, but epsilon/epsilon remains "
        "lower late in training, particularly for larger training sets. Line styles "
        "distinguish N=512, 1024, 2048, and 4096. Curves and bands show mean +/- "
        "sample SD over three seeds. Both axes are logarithmic; zero means are omitted."
    )
    if latex:
        for plain, math_text in (
            ("epsilon/epsilon", TARGET_PAIR_LABELS["eps"]),
            ("v/v", TARGET_PAIR_LABELS["v"]), ("x/x", TARGET_PAIR_LABELS["x"]),
            ("rho=0.3, 0.5, and 0.7", r"$\rho=0.3,0.5,$ and $0.7$"),
            ("Increasing rho", r"Increasing $\rho$"),
            ("N=512, 1024, 2048, and 4096", r"$N=512,1024,2048,$ and $4096$"),
            ("F_mem", r"$F_{\mathrm{mem}}$"), ("+/-", r"$\pm$"),
        ):
            text = text.replace(plain, math_text)
    return text


def render_trajectories(output):
    """Use the workshop's target-by-threshold grid with current data and colors."""
    _, curves = load_trajectories()
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 6.15), sharex=True, sharey=True)
    for i, target in enumerate(COLUMN_TARGETS):
        for j, rho in enumerate(RHOS):
            axis = axes[i, j]
            for n in N_TRAINS:
                curve = curves[(rho, target, n)]
                positive_steps = curve["steps"] > 0
                steps, means, stds = [curve[key][positive_steps]
                                      for key in ("steps", "mean", "std")]
                color = curve_color(target, n)
                # Clip only the displayed SD band; zero seed values remain in the mean.
                axis.fill_between(steps, np.maximum(means - stds, TRAJECTORY_FLOOR),
                                  means + stds, where=means > 0,
                                  color=color, alpha=0.12, linewidth=0)
                # NaNs break the curve at zero rather than fabricating positive counts.
                axis.plot(steps, np.where(means > 0, means, np.nan),
                          color=color, linestyle=N_TRAIN_LINESTYLES[n], linewidth=1.5,
                          marker="o", markersize=1.8)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(250, 1_000_000)
            axis.set_ylim(TRAJECTORY_FLOOR, 1.25)
            axis.set_xticks((1e3, 1e4, 1e5, 1e6))
            axis.set_yticks((1e-4, 1e-3, 1e-2, 1e-1, 1))
            axis.grid(which="major", color="0.87", linewidth=0.5, linestyle=":")
            axis.set_axisbelow(True)
            axis.tick_params(which="both", labelsize=8)
            axis.text(0.04, 0.94, f"({chr(97 + 3 * i + j)})", transform=axis.transAxes,
                      ha="left", va="top", fontsize=9)
            if i == 0:
                axis.set_title(rf"$\rho={rho}$", fontsize=11)
            if j == 0:
                axis.set_ylabel(r"$F_{\mathrm{mem}}$", fontsize=10)
            if i == 2:
                axis.set_xlabel(r"Training steps, $\tau$", fontsize=9)
    handles = [Line2D([], [], color="0.3", linestyle=N_TRAIN_LINESTYLES[n],
                      linewidth=1.6, label=rf"$N={n}$") for n in N_TRAINS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.54, 0.012), columnspacing=2)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.14, top=0.935,
                        wspace=0.13, hspace=0.16)
    for i, target in enumerate(COLUMN_TARGETS):
        position = axes[i, 0].get_position()
        fig.text(0.018, (position.y0 + position.y1) / 2, TARGET_PAIR_LABELS[target],
                 color=TARGET_COLORS[target], fontsize=13, rotation=90,
                 ha="center", va="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix))
    plt.close(fig)
    for suffix, latex in ((".caption.txt", False), (".caption.tex", True)):
        output.with_suffix(suffix).write_text(trajectory_caption(latex) + "\n", encoding="utf-8")


def render(output, mode):
    if mode == "trajectories":
        render_trajectories(output)
        return
    rows = load_readouts()
    if mode == "representation":
        keys = [(metric, "alpha", ALPHA) for metric in METRICS]
        xs, labels = list(range(4)), ["Pixels", "SSCD", "LPIPS", "DINOv2"]
        xlabel = r"Representation ($\alpha=0.01$)"
    else:
        keys = [(RAW, "rho", rho) for rho in RHOS]
        xs, labels = list(RHOS), ["0.3", "0.5", "0.7"]
        xlabel = r"Ratio threshold, $\rho$"
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.55), sharey=True)
    for index, (axis, n) in enumerate(zip(axes, N_TRAINS)):
        for target, marker in zip(TARGETS, ("o", "s", "^")):
            group = [r for r in rows if r["n_train"] == n and r["prediction_space"] == target]
            values = [[100 * row["readouts"][key] for row in group] for key in keys]
            axis.errorbar(xs, [statistics.fmean(v) for v in values],
                          yerr=[statistics.stdev(v) for v in values],
                          color=TARGET_COLORS[target], marker=marker, markersize=4,
                          linewidth=1.2, elinewidth=0.8, capsize=2,
                          label=TARGET_PAIR_LABELS[target])
        axis.set_title(rf"({chr(97 + index)}) $N={n}$", fontsize=9)
        axis.set_xticks(xs, labels, fontsize=8)
        axis.set_xlabel(xlabel, fontsize=8)
        axis.set_ylim(-4, 105)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.grid(axis="x", color="0.87", linestyle=":", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.margins(x=0.09)
    axes[0].set_ylabel(r"$F_{\mathrm{mem}}$ at best FID (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.01), fontsize=9, columnspacing=2.5)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.86, bottom=0.32, wspace=0.17)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix))
    plt.close(fig)
    for suffix, latex in ((".caption.txt", False), (".caption.tex", True)):
        output.with_suffix(suffix).write_text(caption(mode, latex) + "\n", encoding="utf-8")
