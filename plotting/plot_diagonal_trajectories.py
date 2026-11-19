"""Render paired CelebA32 U-Net diagonal FID and raw-pixel f_mem trajectories."""

import csv
import math
from pathlib import Path

from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D

from celeba32_3x3.plot_f_mem_grid import (
    DEFAULT_FIGURE_ROOT, N_TRAINS, N_TRAIN_LINESTYLES, TARGET_COLORS,
    plt, set_panel_width,
)
from paper_style import TARGET_PAIR_LABELS
from plot_fid_by_training_size import EXPERIMENTS, load_records, summarize


EXPERIMENT = "celeba32_unet_3x3"
OUTPUT = DEFAULT_FIGURE_ROOT / "diagonal__fid__fmem_raw_pixels__rho_0p3__seed_mean_sd"
# Match the supplied reference's column order and four-size shade progression.
COLUMN_TARGETS = ("eps", "v", "x")
SIZE_WHITE_MIX = {512: 0.0, 1024: 0.18, 2048: 0.36, 4096: 0.54}
ROW_BACKGROUNDS = ("#F4F7FF", "#FFF8EF")


def load_paired_records(experiment=EXPERIMENT):
    """Join FID and raw-pixel f_mem for the declared U-Net or DiT diagonal runs."""
    settings = ({**EXPERIMENTS["celeba32_dit_3x3"], "title": "CelebA32 U-Net",
                 "model_arch": "unet", "seeds": (0, 1, 2)}
                if experiment == EXPERIMENT else EXPERIMENTS[experiment])
    records = [row for row in load_records(experiment, settings)
               if row["prediction_space"] == row["loss_space"]]
    by_id, quality_steps = {}, {}
    for row in records:
        by_id[row["run_id"]] = row
        quality_steps.setdefault(row["run_id"], set()).add(row["step"])
        row.update(space="", alpha="", rho="", tau=row["step"] / 1_000_000,
                   plotted=row["step"] > 0)

    memorization_steps = {run_id: set() for run_id in by_id}
    path = Path("plot_data") / experiment / "evaluation_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["run_id"] not in by_id or row["metric"] != "f_mem_raw_pixels":
                continue
            if row["space"] != "raw_pixels" or row["alpha"] or row["rho"] != "0.3":
                continue
            step, value = int(row["step"]), float(row["value"])
            steps = memorization_steps[row["run_id"]]
            if step in steps or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Duplicate or invalid f_mem: {row['run_id']}, {step}")
            if row["sample_count"] != "10000":
                raise ValueError(f"Unexpected sample count: {row['run_id']}")
            steps.add(step)
            records.append({**by_id[row["run_id"]], "metric": row["metric"],
                            "step": step, "value": value, "space": row["space"],
                            "alpha": "", "rho": 0.3, "tau": step / 1_000_000,
                            "plotted": step > 0})
    if memorization_steps != quality_steps:
        raise ValueError("FID and raw-pixel memorization checkpoint grids differ.")
    return sorted(records, key=lambda row: (
        row["n_train"], COLUMN_TARGETS.index(row["prediction_space"]),
        row["metric"], row["seed"], row["step"],
    ))


def curve_color(target, n_train):
    """Use the reference's dark-to-light shade progression as N increases."""
    white = SIZE_WHITE_MIX[n_train]
    return to_hex([(1 - white) * channel + white for channel in to_rgb(TARGET_COLORS[target])])


def draw_trajectories(axes, rows):
    """Draw the shared F10/F11 trajectory rows on supplied axes."""
    fid_rows = [row for row in rows if row["metric"] == "inception_fid" and row["plotted"]]
    fid_floor = min(row["mean"] for row in fid_rows) / 2
    fid_min = min(max(fid_floor, row["mean"] - row["std"]) for row in fid_rows) * 0.85
    fid_max = max(row["mean"] + row["std"] for row in fid_rows) * 1.08
    first_tau = min(row["tau"] for row in rows if row["plotted"])
    for column, target in enumerate(COLUMN_TARGETS):
        axes[0, column].set_title(
            TARGET_PAIR_LABELS[target],
            fontsize=10, color=TARGET_COLORS[target], pad=5,
        )
        for i, metric in enumerate(("inception_fid", "f_mem_raw_pixels")):
            axis = axes[i, column]
            axis.set_facecolor(ROW_BACKGROUNDS[i])
            for n_train in N_TRAINS:
                selected = sorted((row for row in rows if row["metric"] == metric
                                   and row["prediction_space"] == target
                                   and row["n_train"] == n_train and row["plotted"]),
                                  key=lambda row: row["step"])
                tau = [row["tau"] for row in selected]
                means = [row["mean"] for row in selected]
                # Display bands respect the positive FID axis and f_mem in [0, 1].
                lower = [max(fid_min if i == 0 else 0, row["mean"] - row["std"])
                         for row in selected]
                upper = [row["mean"] + row["std"] if i == 0 else
                         min(1, row["mean"] + row["std"]) for row in selected]
                color = curve_color(target, n_train)
                axis.fill_between(tau, lower, upper, color=color, alpha=0.12, linewidth=0)
                axis.plot(tau, means, color=color, linestyle=N_TRAIN_LINESTYLES[n_train],
                          linewidth=1.35)
            axis.set_xscale("log")
            axis.set_xlim(first_tau, 1)
            axis.set_xticks((0.001, 0.01, 0.1, 1), ("0.001", "0.01", "0.1", "1"))
            axis.tick_params(axis="both", which="major", labelsize=8, length=2.5, width=0.6)
            axis.tick_params(axis="both", which="minor", length=1.5, width=0.45)
            axis.grid(which="major", color="#B9BDC5", alpha=0.32, linewidth=0.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(fid_min, fid_max)
        axes[1, column].set_ylim(0, 1.02)
        axes[1, column].set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        axes[1, column].set_xlabel(r"$\tau$ (training steps / $10^6$)", fontsize=8)
    axes[0, 0].set_ylabel("FID (log)", fontsize=9)
    axes[1, 0].set_ylabel(r"$f_{\mathrm{mem}}$", fontsize=10)


def plot_grid(rows, output):
    """Place log-FID above f_mem, with a common logarithmic training-time axis."""
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 3.65), sharex=True, sharey="row")
    draw_trajectories(axes, rows)
    figure.text(0.5, 0.98,
                r"CelebA32 U-Net · raw pixels, $\rho=0.3$ · three-seed mean ± SD",
                ha="center", va="top", fontsize=9)
    handles = [Line2D([], [], color=to_hex([0.2 + 0.8 * SIZE_WHITE_MIX[n]] * 3),
                      linestyle=N_TRAIN_LINESTYLES[n], linewidth=1.5,
                      label=rf"$N_{{\mathrm{{train}}}}={n:,}$") for n in N_TRAINS]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                  ncol=4, frameon=False, fontsize=8, handlelength=2.8, columnspacing=1.3)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.85,
                           hspace=0.18, wspace=0.10)
    set_panel_width(figure, axes[0, 0])
    for suffix in (".pdf", ".svg"):
        figure.savefig(output.with_suffix(suffix))
    plt.close(figure)


def render(output=OUTPUT, write_tables=False):
    records = load_paired_records()
    rows = summarize(records)
    for row in rows:
        row.update(rho=0.3 if row["metric"] == "f_mem_raw_pixels" else "",
                   tau=row["step"] / 1_000_000, plotted=row["step"] > 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_grid(rows, output)
    if write_tables:
        for suffix, table in (("__per_seed.csv", records), ("__mean_sd.csv", rows)):
            with output.with_name(output.name + suffix).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    caption = (
        "FID and memorization trajectories for CelebA32 grayscale 32x32 U-Net diagonal conditions. "
        "Columns: eps/eps, v/v, x/x. Top: Inception FID on a logarithmic scale. "
        "Bottom: raw-pixel f_mem at fixed rho=0.3. Lines are means over seeds 0, 1, 2; "
        "bands show one sample standard deviation, clipped to the displayed metric bounds. "
        "Training sizes 512, 1024, 2048, 4096 use solid, dashed, dash-dot, dotted lines, "
        "respectively, with increasingly lighter target colors. Each evaluation uses 10,000 "
        "generated samples. The logarithmic time axis is tau=training_steps/10^6. "
        "All 71 positive saved steps are plotted without smoothing; step zero remains in the CSVs. "
        "N=8192 from the historical reference is not present in the current export. "
        "Numerical inputs: plot_data/celeba32_unet_3x3/runs.csv and evaluation_metrics.csv.\n"
    )
    output.with_name(output.name + '.caption.txt').write_text(caption, encoding="utf-8")
    print(f"Wrote {output}.[pdf,svg] from 36 runs; {len(rows)} checkpoint summaries.")


if __name__ == "__main__":
    render(write_tables=True)
