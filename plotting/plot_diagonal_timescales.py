"""Compare near-best-quality time, memorization onset, and their signed gap."""

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from celeba32_3x3.plot_f_mem_grid import (
    DEFAULT_FIGURE_ROOT, N_TRAINS, TARGET_COLORS, plt,
)
from plot_diagonal_summary import ONSET_CUTOFF, TARGET_MARKERS, select_seed_checkpoints
from plot_diagonal_trajectories import EXPERIMENT, load_paired_records
from paper_style import TARGET_PAIR_LABELS


OUTPUT = DEFAULT_FIGURE_ROOT / "diagonal__timescales__raw_pixels__rho_0p3"
MODEL_LABELS = {"celeba32_unet_3x3": "U-Net", "celeba32_dit_3x3": "Small-DiT"}
FIGURE_SIZE = (6.75, 1.95)
MARKER_SIZE = 2.8
QUALITY_TOLERANCE = 0.10


def summarize_timescales(records):
    """Compute paired seed times before means/SD; keep censoring as bounds."""
    selected = select_seed_checkpoints(records)
    fid_by_run = {}
    for row in records:
        if row["metric"] == "inception_fid":
            fid_by_run.setdefault(row["run_id"], []).append(row)
    seeds = []
    for row in selected:
        threshold = (1 + QUALITY_TOLERANCE) * row["inception_fid"]
        tau_gen = min(point["step"] for point in fid_by_run[row["run_id"]]
                      if point["value"] <= threshold)
        censored = row["onset_censored"]
        tau_mem = row["onset_step"]
        horizon = row["last_evaluated_step"]
        seeds.append({
            **{key: row[key] for key in ("n_train", "prediction_space", "loss_space", "seed", "run_id")},
            "rho": 0.3, "quality_tolerance": QUALITY_TOLERANCE,
            "inception_fid_best": row["inception_fid"], "best_fid_step": row["best_fid_step"],
            "gen_fid_threshold": threshold, "tau_gen_step": tau_gen,
            "onset_cutoff": ONSET_CUTOFF, "onset_rule": row["onset_rule"],
            "tau_mem_step": tau_mem, "censored": censored, "last_evaluated_step": horizon,
            "tau_mem_lower_bound_step": horizon if censored else "",
            "delta_tau_step": "" if censored else tau_mem - tau_gen,
            "delta_tau_lower_bound_step": horizon - tau_gen if censored else "",
        })
    rows = []
    for n_train in N_TRAINS:
        for target in TARGET_MARKERS:
            group = [row for row in seeds if row["n_train"] == n_train
                     and row["prediction_space"] == target]
            n_censored = sum(row["censored"] for row in group)
            row = {"n_train": n_train, "prediction_space": target, "loss_space": target,
                   "rho": 0.3, "quality_tolerance": QUALITY_TOLERANCE,
                   "onset_cutoff": ONSET_CUTOFF, "n_seeds": len(group), "n_censored": n_censored}
            for metric in ("tau_gen", "tau_mem", "delta_tau"):
                values = [seed[f"{metric}_step"] for seed in group]
                complete = all(value != "" for value in values)
                row[f"{metric}_mean_step"] = statistics.fmean(values) if complete else ""
                row[f"{metric}_std_step"] = statistics.stdev(values) if complete and len(values) > 1 else ""
                if metric != "tau_gen":
                    # Averaging individual lower bounds bounds the three-seed mean;
                    # it is not an estimated onset/window and has no SD error bar.
                    bounds = [seed[f"{metric}_lower_bound_step"] if seed["censored"]
                              else seed[f"{metric}_step"] for seed in group]
                    row[f"{metric}_mean_lower_bound_step"] = statistics.fmean(bounds) if n_censored else ""
            rows.append(row)
    return seeds, rows


def plot_grid(rows, output):
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    single_seed = all(row["n_seeds"] == 1 for row in rows)
    censored = any(row["n_censored"] for row in rows)
    metrics = ("tau_gen", "tau_mem", "delta_tau")
    labels = (r"$\tau_{\mathrm{gen}}$", r"$\tau_{\mathrm{mem}}$", r"$\Delta\tau$")
    for panel, axis, metric, label in zip("abc", axes, metrics, labels):
        for target, marker in TARGET_MARKERS.items():
            group = [row for row in rows if row["prediction_space"] == target]
            means = [row[f"{metric}_mean_step"]
                     if row[f"{metric}_mean_step"] != "" else np.nan for row in group]
            stds = [row[f"{metric}_std_step"]
                    if row[f"{metric}_std_step"] != "" else np.nan for row in group]
            color = TARGET_COLORS[target]
            axis.errorbar(N_TRAINS, means, yerr=None if single_seed else stds, color=color, marker=marker,
                          markersize=MARKER_SIZE, linewidth=1, elinewidth=0.6, capsize=1.5)
            if metric == "tau_gen":
                continue
            for row in group:
                if not row["n_censored"]:
                    continue
                position = row["n_train"] * {"x": 1, "eps": 0.95, "v": 1.05}[target]
                bound = row[f"{metric}_mean_lower_bound_step"]
                axis.plot(position, bound, marker=marker, markerfacecolor="white",
                          markeredgecolor=color, markersize=MARKER_SIZE + 0.8, markeredgewidth=0.8)
                axis.annotate("", xy=(position, bound + 120_000), xytext=(position, bound + 25_000),
                              arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8,
                                          "shrinkA": 0, "shrinkB": 0, "mutation_scale": 5})
        axis.set_title(f"({panel})", loc="right", fontsize=8, pad=3)
        axis.set_ylabel(label, fontsize=9, labelpad=2)
        axis.set_xlabel(r"$N_{\mathrm{train}}$", fontsize=8, labelpad=2)
        axis.set_xscale("log", base=2)
        axis.set_xticks(N_TRAINS, [str(n) for n in N_TRAINS])
        axis.set_xlim(460, 4600)
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        axis.tick_params(axis="both", which="major", labelsize=7.5, length=2.5, width=0.5)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(6, 6), useMathText=False)
        axis.yaxis.get_offset_text().set_fontsize(7.5)
        axis.grid(axis="y", color="#B9BDC5", alpha=0.4, linewidth=0.5)
        axis.grid(axis="x", which="major", color="#B9BDC5", alpha=0.55,
                  linewidth=0.6, linestyle=":")
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
        if metric != "delta_tau":
            axis.set_ylim(-35_000, 1_150_000)
            axis.set_yticks((0, 250_000, 500_000, 750_000, 1_000_000))
    axes[2].set_ylim(-1_000_000, 1_150_000)
    axes[2].set_yticks((-1_000_000, -500_000, 0, 500_000, 1_000_000))
    axes[2].axhline(0, color="#333333", linewidth=0.7, linestyle=":")
    # DiT's gaps are all observed: fit their range while retaining the zero reference.
    if single_seed and not censored:
        axes[2].relim()
        axes[2].autoscale(enable=True, axis="y")
        axes[2].margins(y=0.12)
        axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4))
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.30, top=0.86, wspace=0.39)
    for suffix in (".pdf", ".svg"):
        figure.savefig(output.with_suffix(suffix))
    plt.close(figure)


def render(output=None, write_tables=False, experiment=EXPERIMENT):
    if output is None:
        output = OUTPUT if experiment == EXPERIMENT else Path("outputs/figures") / experiment / OUTPUT.name
    seeds, rows = summarize_timescales(load_paired_records(experiment))
    output.parent.mkdir(parents=True, exist_ok=True)
    model = MODEL_LABELS[experiment]
    plot_grid(rows, output)
    if write_tables:
        for suffix, table in (("__per_seed.csv", seeds), ("__mean_sd.csv", rows)):
            with output.with_name(output.name + suffix).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    seed_ids = sorted({row["seed"] for row in seeds})
    single_seed = len(seed_ids) == 1
    uncertainty = ("Observed points are seed 0 values; sample SD is undefined and no error bars are drawn. "
                   if single_seed else "Observed conditions show means and sample SD over three seeds. ")
    censoring = (
        "Censored onset has tau_mem>T and delta_tau>T-tau_gen, where T is the last saved step. "
        "Open markers with upward arrows show lower bounds on the seed mean, obtained by "
        "averaging individual bounds; they are not measured means and carry no SD. "
        "Lines stop at missing estimates. Small horizontal offsets separate overlapping markers. "
        if any(row["censored"] for row in seeds) else
        "Every run has observed onset; no censored markers are needed. The gap panel fits "
        "the observed single-seed range while retaining the zero reference. "
    )
    caption = (
        f"CelebA32 grayscale 32x32 {model} diagonal timescales, seeds "
        f"{'/'.join(map(str, seed_ids))} and N=512/1024/2048/4096. "
        "Three panels run left to right in the workshop figure's compact layout. "
        "Red circles: x/x; blue squares: eps/eps; green triangles: v/v. "
        "(a) tau_gen is the first saved checkpoint with Inception FID <=1.10 times the run's "
        "minimum saved FID. It is within-run quality attainment, not a generalization test; "
        "quality is not required to remain within this band. (b) tau_mem is the first saved "
        "checkpoint with raw-pixel f_mem>=0.05 at all remaining saved checkpoints through "
        "1,000,000 steps. The metric uses fixed rho=0.3; the 5% onset cutoff is not an alpha "
        "calibration level. (c) delta_tau=tau_mem-tau_gen is computed within each seed before "
        "aggregation. Negative values mean memorization onset precedes near-best quality; "
        "they are retained without clipping. All y axes are linear and show training steps "
        "with a 10^6 multiplier; (a,b) share the same limits. " + uncertainty + censoring +
        "Each evaluation uses 10,000 generated samples. "
        f"Inputs: plot_data/{experiment}/runs.csv and evaluation_metrics.csv only.\n"
    )
    output.with_name(output.name + ".caption.txt").write_text(caption, encoding="utf-8")
    paper_uncertainty = (
        r"Points show seed 0; error bars are omitted because between-seed variability cannot be estimated. "
        if single_seed else
        r"Points and error bars show means and sample standard deviations across three seeds. "
    )
    paper_censoring = (
        r"Open markers with upward arrows denote lower bounds when onset is unobserved by "
        r"$T=10^6$ steps: $\tau_{\mathrm{mem}}>T$ and "
        r"$\Delta\tau>T-\tau_{\mathrm{gen}}$. Bounds are averaged across seeds and carry no error bars. "
        if any(row["censored"] for row in seeds) else
        r"All memorization onsets are observed within the training horizon. "
    )
    paper_caption = (
        rf"\caption{{Quality and Memorization Timescales for CelebA32 {model}. "
        r"(a) $\tau_{\mathrm{gen}}$ is the first saved checkpoint with FID at most "
        r"$1.1$ times the run's minimum saved FID. "
        r"(b) $\tau_{\mathrm{mem}}$ is the first saved checkpoint at which raw-pixel "
        r"$f_{\mathrm{mem}}\geq0.05$ and remains so at every later saved checkpoint through "
        r"$10^6$ training steps, using $\rho=0.3$. "
        r"(c) $\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$, computed per seed; "
        r"negative values indicate memorization onset before near-best FID. "
        + paper_uncertainty + paper_censoring +
        f"Red circles, blue squares, and green triangles denote {TARGET_PAIR_LABELS['x']}, "
        f"{TARGET_PAIR_LABELS['eps']}, and {TARGET_PAIR_LABELS['v']}, respectively. "
        r"Dotted vertical lines mark training sizes. "
        r"Panels run left to right. All time axes are linear with a $10^6$ multiplier; "
        r"panels (a,b) share the same limits. "
        r"Each evaluation uses 10,000 generated samples. The quality threshold is specific to each run; "
        r"the signed gap does not establish a continuous interval of near-best quality without memorization.}"
        "\n"
    )
    output.with_name(output.name + ".caption.tex").write_text(paper_caption, encoding="utf-8")
    print(f"Wrote {output}.[pdf,svg]; {len(seeds)} paired seed times and {len(rows)} summaries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=MODEL_LABELS, default=EXPERIMENT)
    args = parser.parse_args()
    render(write_tables=True, experiment=args.experiment)
