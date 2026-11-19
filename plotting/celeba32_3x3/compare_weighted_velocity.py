"""Compare explicit velocity weights with diagonal targets or matched velocity-output runs."""

import csv
import statistics
from pathlib import Path

from plot_f_mem_grid import TARGET_COLORS, _load_series, plt
from paper_style import TARGET_LABELS, TARGET_PAIR_LABELS, apply_style

SIZES = (512, 2048)
RHO = 0.3
OUTPUT_ROOT = Path("outputs/figures/celeba32_weighting_comparison")
CONDITIONS = {
    "x/x": (TARGET_PAIR_LABELS["x"], TARGET_COLORS["x"], "-"),
    "eps/eps": (TARGET_PAIR_LABELS["eps"], TARGET_COLORS["eps"], "-"),
    "v/v": (TARGET_PAIR_LABELS["v"], TARGET_COLORS["v"], "-"),
    "v/v x_equivalent": (f"{TARGET_PAIR_LABELS['v']} ({TARGET_LABELS['x']}-equivalent), "
                         r"$w=(1-t)^2$", "#9467BD", "--"),
    "v/v eps_equivalent": (f"{TARGET_PAIR_LABELS['v']} ({TARGET_LABELS['eps']}-equivalent), "
                           r"$w=t^2$", "#FF7F0E", ":"),
}
MATCHED_CONDITIONS = {
    "v/x": (f"{TARGET_LABELS['v']}/{TARGET_LABELS['x']}", TARGET_COLORS["x"], "-"),
    "v/eps": (f"{TARGET_LABELS['v']}/{TARGET_LABELS['eps']}", TARGET_COLORS["eps"], "-"),
    "v/v": CONDITIONS["v/v"],
    "v/v x_equivalent": (f"{TARGET_LABELS['x']}-equivalent, "
                         r"$w=(1-t)^2$", "#9467BD", "--"),
    "v/v eps_equivalent": (f"{TARGET_LABELS['eps']}-equivalent, "
                           r"$w=t^2$", "#FF7F0E", ":"),
}
MATCH_FIELDS = (
    "dataset", "model_arch", "n_train", "seed", "training_steps",
    "learning_rate", "batch_size", "effective_batch_size", "image_channels",
    "image_size", "evaluation_checkpoint_count", "evaluation_sample_count",
    "evaluation_tier", "source_state", "source_final_step", "source_data_bundle_sha256",
)


def load_comparison(matched=False):
    """Check matching records and retain three seeds for each of five conditions."""
    records = {}
    metrics = {name: {} for name in ("inception_fid", "f_mem_raw_pixels")}
    for experiment in ("celeba32_unet_3x3", "celeba32_vv_weighting"):
        root = Path("plot_data") / experiment
        with (root / "runs.csv").open(newline="", encoding="utf-8") as handle:
            records[experiment] = [
                row for row in csv.DictReader(handle) if int(row["n_train"]) in SIZES
            ]
        run_ids = {row["run_id"]: row["run_id"] for row in records[experiment]}
        for metric in metrics:
            metrics[metric].update(_load_series(root, run_ids, RHO, metric))

    matrix = {
        (int(row["n_train"]), int(row["seed"]), row["prediction_space"], row["loss_space"]): row
        for row in records["celeba32_unet_3x3"]
    }
    weighted = {
        (int(row["n_train"]), int(row["seed"]), row["loss_weighting"]): row
        for row in records["celeba32_vv_weighting"]
    }
    selected = []
    for n_train in SIZES:
        for seed in range(3):
            baseline = matrix[(n_train, seed, "v", "v")]
            for weight in ("unit", "x_equivalent", "eps_equivalent"):
                row = weighted[(n_train, seed, weight)]
                if row["prediction_space"] != "v" or row["loss_space"] != "v":
                    raise ValueError("The weighting controls must predict and target velocity.")
                for field in MATCH_FIELDS:
                    if row[field] != baseline[field]:
                        raise ValueError(f"Unmatched {field}: {row['run_id']}")
                if weight == "unit":
                    # Same-seed control reproductions must not count as extra seeds.
                    for metric in metrics:
                        if metrics[metric][row["run_id"]] != metrics[metric][baseline["run_id"]]:
                            raise ValueError(f"The repeated v/v control differs for {metric}.")
                else:
                    selected.append({**row, "condition": f"v/v {weight}"})
            for target in ("x", "eps", "v"):
                prediction = "v" if matched else target
                row = matrix[(n_train, seed, prediction, target)]
                if row["loss_weighting"] != "unit":
                    raise ValueError("Matrix controls require unit loss weighting.")
                for field in MATCH_FIELDS:
                    if row[field] != baseline[field]:
                        raise ValueError(f"Unmatched {field}: {row['run_id']}")
                selected.append({**row, "condition": f"{prediction}/{target}"})
    return selected, metrics


def summarize(selected, metrics):
    """Retain paired checkpoint values and select best FID separately per seed."""
    trajectories, best_rows = [], []
    for run in selected:
        run_id = run["run_id"]
        fid = metrics["inception_fid"][run_id]
        f_mem = metrics["f_mem_raw_pixels"][run_id]
        if set(fid) != set(f_mem):
            raise ValueError(f"Checkpoint grids differ for {run_id}")
        best_step = min(fid, key=lambda step: (fid[step], step))
        for step in sorted(fid):
            row = {
                "n_train": int(run["n_train"]), "condition": run["condition"],
                "seed": int(run["seed"]), "run_id": run_id, "rho": RHO,
                "step": step, "inception_fid": fid[step], "f_mem_raw_pixels": f_mem[step],
            }
            trajectories.append(row)
            if step == best_step:
                best_rows.append(row)
    summary_rows = []
    for n_train in SIZES:
        for condition in CONDITIONS:
            seeds = [r for r in best_rows if r["n_train"] == n_train and r["condition"] == condition]
            if len(seeds) != 3 or {row["seed"] for row in seeds} != {0, 1, 2}:
                raise ValueError(f"Expected exactly three seeds: {n_train}, {condition}")
            row = {"n_train": n_train, "condition": condition, "rho": RHO, "n_seeds": 3}
            for metric in metrics:
                values = [seed[metric] for seed in seeds]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            summary_rows.append(row)
    return trajectories, best_rows, summary_rows


def plot_comparison(selected, metrics, output=None, matched=False):
    """Separate FID and memorization, with one column per training size."""
    output = Path(output) if output is not None else OUTPUT_ROOT / "diagonal_vs_weighted_velocity"
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_style()
    conditions = MATCHED_CONDITIONS if matched else CONDITIONS
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 4.9), sharex=True, sharey="row")
    for j, n_train in enumerate(SIZES):
        axes[0, j].set_title(rf"$N={n_train:,}$", fontsize=11, pad=7)
        for condition, (label, color, linestyle) in conditions.items():
            runs = [r for r in selected if int(r["n_train"]) == n_train and r["condition"] == condition]
            if sorted(int(r["seed"]) for r in runs) != [0, 1, 2]:
                raise ValueError(f"Expected three distinct seeds: {n_train}, {condition}")
            for i, metric in enumerate(("inception_fid", "f_mem_raw_pixels")):
                series = [metrics[metric][r["run_id"]] for r in runs]
                if any(sorted(values) != sorted(series[0]) for values in series):
                    raise ValueError("Seed checkpoint grids differ.")
                # A log time axis displays every positive checkpoint without smoothing.
                steps = sorted(step for step in series[0] if step > 0)
                mean = [statistics.fmean(values[t] for values in series) for t in steps]
                std = [statistics.stdev(values[t] for values in series) for t in steps]
                lower = [max(5 if i == 0 else 0, m - s) for m, s in zip(mean, std)]
                upper = [m + s for m, s in zip(mean, std)]
                if metric == "f_mem_raw_pixels":
                    upper = [min(1, value) for value in upper]
                # Dashes leave the nearly coincident solid reference curves visible.
                axes[i, j].plot(steps, mean, color=color, linestyle=linestyle,
                                label=label, linewidth=1.5 if linestyle == "-" else 1.3)
                axes[i, j].fill_between(steps, lower, upper, color=color, alpha=0.10, linewidth=0)
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylim(5, 500)
        axes[1, j].set_ylim(0, 1.03)
        axes[1, j].set_yticks((0, 0.5, 1))
        axes[1, j].set_xlabel("Training step", fontsize=10)
        for axis in axes[:, j]:
            axis.set_xscale("log")
            axis.set_xlim(steps[0], steps[-1])
            axis.set_xticks((1000, 10000, 100000, 1000000), ("1k", "10k", "100k", "1M"))
            axis.grid(which="major", color="0.87", linewidth=0.5, linestyle=":")
            axis.set_axisbelow(True)
            axis.tick_params(which="major", labelsize=9, length=2.5, width=0.6)
            axis.tick_params(which="minor", length=1.5, width=0.45)
            axis.spines[["top", "right"]].set_visible(False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
    axes[0, 0].set_ylabel("Inception FID", fontsize=10)
    axes[1, 0].set_ylabel(r"$f_{\mathrm{mem}}$", fontsize=11)
    for index, axis in enumerate(axes.flat):
        axis.set_title(f"({chr(97 + index)})", loc="left", fontsize=9, fontweight="bold", pad=7)
    # Group diagonal models and weighted controls in two columns, as in F07.
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
                  ncol=2, frameon=False, fontsize=10, handlelength=2.5,
                  columnspacing=2.0, handletextpad=0.6, labelspacing=0.4)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.80, hspace=0.20, wspace=0.12)
    for extension in ("pdf", "svg"):
        path = output.with_suffix(f".{extension}")
        figure.savefig(path, dpi=180)
        print(f"Wrote {path.resolve()}")
    plt.close(figure)
    for latex, suffix in ((False, ".caption.txt"), (True, ".caption.tex")):
        output.with_suffix(suffix).write_text(caption(matched, latex) + "\n", encoding="utf-8")


def caption(matched, latex=False):
    """State the compared objectives, aggregation, and displayed checkpoints."""
    comparison = (
        "Explicit velocity-loss weights and their matching prediction/loss objectives on CelebA32 grayscale U-Net. "
        "Solid red, blue, and green curves show v/x, v/epsilon, and v/v. "
        if matched else
        "Diagonal targets and explicitly weighted velocity objectives on CelebA32 grayscale U-Net. "
        "Solid red, blue, and green curves show x/x, epsilon/epsilon, and v/v. "
    )
    text = comparison + (
        "Dashed purple and dotted orange curves show native v/v with the x-equivalent weight w=(1-t)^2 "
        "and epsilon-equivalent weight w=t^2, respectively. "
        "Columns use N=512 and N=2048. The upper row shows Inception FID on a logarithmic scale; "
        "the lower row shows raw-pixel f_mem at rho=0.3. Curves and shaded bands show the mean and "
        "one sample SD across three seeds. Bands are clipped to the displayed metric bounds. "
        "All 71 positive saved checkpoints are shown on a logarithmic training-step axis without smoothing. "
        "Each checkpoint uses 10,000 generated samples. The repeated unit-weight v/v controls match "
        "the matrix v/v trajectories exactly and are counted once. "
    )
    text += (
        "The explicit weights and corresponding converted-target losses give closely agreeing, but not identical, trajectories."
        if matched else
        "Matching the induced velocity-loss weight does not make a direct x or epsilon predictor identical to a velocity predictor."
    )
    if latex:
        for plain, math_text in (
            ("w=(1-t)^2", r"$w=(1-t)^2$"), ("w=t^2", r"$w=t^2$"),
            ("N=512", r"$N=512$"), ("N=2048", r"$N=2048$"),
            ("rho=0.3", r"$\rho=0.3$"), ("f_mem", r"$f_{\mathrm{mem}}$"),
            ("x-equivalent", f"{TARGET_LABELS['x']}-equivalent"),
            ("epsilon-equivalent", f"{TARGET_LABELS['eps']}-equivalent"),
            ("v/epsilon", f"{TARGET_LABELS['v']}/{TARGET_LABELS['eps']}"),
            ("v/x", f"{TARGET_LABELS['v']}/{TARGET_LABELS['x']}"),
            ("epsilon/epsilon", TARGET_PAIR_LABELS["eps"]),
            ("x/x", TARGET_PAIR_LABELS["x"]), ("v/v", TARGET_PAIR_LABELS["v"]),
        ):
            text = text.replace(plain, math_text)
    return text


def main():
    selected, metrics = load_comparison()
    trajectories, best_rows, summary_rows = summarize(selected, metrics)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("trajectories", trajectories), ("best_fid_per_seed", best_rows), ("best_fid_mean_sd", summary_rows)):
        path = OUTPUT_ROOT / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path.resolve()} ({len(rows)} rows)")
    plot_comparison(selected, metrics)
    for row in summary_rows:
        print(f"N={row['n_train']} {row['condition']}: "
              f"{row['f_mem_raw_pixels_mean']:.4f} ± {row['f_mem_raw_pixels_std']:.4f} "
              f"(FID {row['inception_fid_mean']:.2f} ± {row['inception_fid_std']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
