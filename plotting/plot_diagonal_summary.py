"""Add two wide best-FID summary panels below the F10 trajectories."""

import csv
import statistics

import numpy as np
from matplotlib.lines import Line2D

from celeba32_3x3.plot_f_mem_grid import (
    DEFAULT_FIGURE_ROOT, N_TRAINS, N_TRAIN_LINESTYLES, TARGET_COLORS,
    plt,
)
from plot_diagonal_trajectories import COLUMN_TARGETS, draw_trajectories, load_paired_records
from plot_fid_by_training_size import summarize
from paper_style import TARGET_PAIR_LABELS


OUTPUT = DEFAULT_FIGURE_ROOT / "diagonal__trajectories_and_summaries__raw_pixels__rho_0p3"
# Moderate reduction from the original width, preserving its aspect ratio.
FIGURE_WIDTH_INCHES = 8.8
FIGURE_ASPECT_RATIO = 9.92391304347826 / 4.9
# tau_mem uses the plan's sustained 5% crossing, applied here to raw pixels.
ONSET_CUTOFF = 0.05
TARGET_MARKERS = {"x": "o", "eps": "s", "v": "^"}


def select_seed_checkpoints(records):
    """Select best FID and sustained onset separately within each seed."""
    runs = {}
    for row in records:
        run = runs.setdefault(row["run_id"], {})
        run.setdefault(row["metric"], {})[row["step"]] = row
    selected = []
    for run_id, metrics in runs.items():
        fid, f_mem = metrics["inception_fid"], metrics["f_mem_raw_pixels"]
        best_step = min(fid, key=lambda step: (fid[step]["value"], step))
        steps = sorted(f_mem)
        # Sustained onset is the first saved step after the final sub-cutoff value.
        index = max((i for i, step in enumerate(steps)
                     if f_mem[step]["value"] < ONSET_CUTOFF), default=-1) + 1
        censored = index == len(steps)
        row = fid[best_step]
        selected.append({
            "n_train": row["n_train"], "prediction_space": row["prediction_space"],
            "loss_space": row["loss_space"], "seed": row["seed"], "run_id": run_id,
            "rho": 0.3, "best_fid_step": best_step, "inception_fid": row["value"],
            "f_mem_raw_pixels": f_mem[best_step]["value"],
            "onset_cutoff": ONSET_CUTOFF, "onset_rule": "sustained_through_last_checkpoint",
            "onset_step": "" if censored else steps[index], "onset_censored": censored,
            "last_evaluated_step": steps[-1], "onset_supporting_checkpoints": len(steps) - index,
        })
    return selected


def summarize_runs(records):
    """Summarize the selected checkpoints across the three U-Net seeds."""
    selected = select_seed_checkpoints(records)
    rows = []
    for n_train in N_TRAINS:
        for target in TARGET_MARKERS:
            seeds = [row for row in selected if row["n_train"] == n_train
                     and row["prediction_space"] == target]
            observed = [row["onset_step"] for row in seeds if not row["onset_censored"]]
            row = {"n_train": n_train, "prediction_space": target, "loss_space": target,
                   "rho": 0.3, "n_seeds": len(seeds), "onset_cutoff": ONSET_CUTOFF,
                   "onset_rule": "sustained_through_last_checkpoint"}
            for metric in ("inception_fid", "f_mem_raw_pixels"):
                values = [seed[metric] for seed in seeds]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            # A censored run is not an observed onset at the evaluation horizon.
            row.update(n_onset_observed=len(observed), n_onset_censored=len(seeds) - len(observed),
                       onset_step_mean=statistics.fmean(observed) if len(observed) == len(seeds) else "",
                       onset_step_std=statistics.stdev(observed) if len(observed) == len(seeds) else "")
            rows.append(row)
    return selected, rows


def draw_summaries(axes, rows):
    for target, marker in TARGET_MARKERS.items():
        group = [row for row in rows if row["prediction_space"] == target]
        color = TARGET_COLORS[target]
        for axis, metric in zip(axes, ("inception_fid", "f_mem_raw_pixels")):
            means = [row[f"{metric}_mean"] for row in group]
            stds = [row[f"{metric}_std"] for row in group]
            axis.errorbar(N_TRAINS, means, yerr=stds, color=color, marker=marker,
                          markersize=3.5, linewidth=1.3, elinewidth=0.8, capsize=2)
    titles = ("Best FID ↓", r"$f_{\mathrm{mem}}$ at best FID")
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=8, pad=4)
        axis.set_xscale("log", base=2)
        axis.set_xticks(N_TRAINS, [str(n) for n in N_TRAINS])
        axis.set_xlim(460, 4600)
        axis.set_xlabel(r"Training examples, $N$", fontsize=8)
        axis.tick_params(axis="both", which="major", labelsize=8, length=2.5, width=0.6)
        axis.tick_params(axis="both", which="minor", length=1.5, width=0.45)
        axis.grid(axis="y", color="#B9BDC5", alpha=0.4, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
    axes[0].set_ylim(0, 40)
    axes[0].set_yticks((0, 20, 40))
    axes[1].set_ylim(-0.03, 1.04)
    axes[1].set_yticks((0, 0.5, 1))


def plot_grid(trajectory_rows, rows, output):
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_WIDTH_INCHES / FIGURE_ASPECT_RATIO))
    grid = figure.add_gridspec(4, 3, height_ratios=(1.4, 1.4, 0.45, 1), hspace=0.50, wspace=0.16)
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = figure.add_subplot(grid[i, j], sharex=axes[0, 0] if i or j else None,
                                            sharey=axes[i, 0] if j else None)
            axes[i, j].tick_params(labelbottom=i == 1, labelleft=j == 0)
    draw_trajectories(axes[:2], trajectory_rows)
    for column, target in enumerate(COLUMN_TARGETS):
        axes[0, column].set_title(TARGET_PAIR_LABELS[target], fontsize=9,
                                  color=TARGET_COLORS[target], pad=4)
        axes[1, column].set_yticks((0, 0.5, 1))
    bottom_grid = grid[3, :].subgridspec(1, 2, wspace=0.10)
    bottom_axes = [figure.add_subplot(bottom_grid[0, j]) for j in range(2)]
    draw_summaries(bottom_axes, rows)
    for index, axis in enumerate([*axes.flat, *bottom_axes]):
        axis.set_title(f"({chr(97 + index)})", loc="left", fontsize=8,
                       fontweight="bold", pad=4)
    # Keep the dataset, uncertainty, and visual keys together in the caption.
    figure.text(0.5, 0.04, r"CelebA32 U-Net · raw pixels, $\rho=0.3$ · three-seed mean ± sample SD",
                ha="center", va="center", fontsize=8)
    size_handles = [Line2D([], [], color="#555555", linestyle=N_TRAIN_LINESTYLES[n],
                           label=rf"$N={n:,}$", linewidth=1.5) for n in N_TRAINS]
    target_handles = [Line2D([], [], color=TARGET_COLORS[t], marker=m, markersize=3.5,
                             label=TARGET_PAIR_LABELS[t])
                      for t, m in TARGET_MARKERS.items()]
    figure.legend(handles=size_handles + target_handles, loc="center", bbox_to_anchor=(0.5, 0.10),
                           ncol=7, frameon=False, fontsize=8, handlelength=2, columnspacing=0.8)
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.23, top=0.925)
    for suffix in (".pdf", ".svg"):
        figure.savefig(output.with_suffix(suffix))
    plt.close(figure)


def render(output=OUTPUT, write_tables=False):
    records = load_paired_records()
    selected, rows = summarize_runs(records)
    trajectory_rows = summarize(records)
    for row in trajectory_rows:
        row.update(tau=row["step"] / 1_000_000, plotted=row["step"] > 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_grid(trajectory_rows, rows, output)
    if write_tables:
        for suffix, table in (("__per_seed.csv", selected), ("__mean_sd.csv", rows)):
            with output.with_name(output.name + suffix).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    caption = (
        "CelebA32 grayscale 32x32 U-Net diagonal trajectories and summaries. "
        "Columns in the first two rows: eps/eps (blue), v/v (green), x/x (red). "
        "(a-c) Inception FID (log). (d-f) Raw-pixel f_mem at fixed rho=0.3. "
        "Four increasingly lighter colors and solid/dashed/dash-dot/dotted lines identify "
        "N=512/1024/2048/4096. In the summary row, red circles identify x/x, blue squares "
        "eps/eps, and green triangles v/v. Both visual keys are grouped in the caption below "
        "the panels. All 71 positive saved checkpoints are plotted without smoothing, "
        "with tau=training_steps/10^6. Bottom: (g) lowest saved FID, (h) paired raw-pixel "
        "f_mem at that checkpoint. These two panels share the full bottom-row width. "
        "Best FID is selected separately in each seed, breaking exact ties by earlier step. "
        "Lines, bands and error bars show three-seed means and sample SD; "
        "trajectory bands are clipped to metric display bounds. Generation and memorization "
        "timescales and their signed gap are shown separately in M5. "
        "Each evaluation uses 10,000 generated samples. "
        "Numerical inputs: plot_data/celeba32_unet_3x3/runs.csv and evaluation_metrics.csv.\n"
    )
    output.with_name(output.name + ".caption.txt").write_text(caption, encoding="utf-8")
    print(f"Wrote {output}.[pdf,svg]; {len(selected)} runs and {len(rows)} condition summaries.")


if __name__ == "__main__":
    render(write_tables=True)
