"""Use the paper PDF's M3 layout for memorization at each seed's best FID."""

import math

from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb

from plot_f_mem_grid import TARGET_COLORS, TARGET_LABELS, TARGETS, plt
from summarize_f_mem_at_best_fid import RHO, summarize
from plot_diagonal_summary import FIGURE_WIDTH_INCHES


# The supplied paper PDF blends each diagonal target color with 92% white.
TINTS = {target: tuple(0.92 + 0.08 * channel for channel in to_rgb(color))
         for target, color in TARGET_COLORS.items()}


def render(output, n_trains=(512, 2048), uncertainty="sd",
           experiment="celeba32_unet_3x3", metric="f_mem_raw_pixels", rho=RHO, alpha=None):
    """Reuse the established checkpoint selection; only the table layout changes."""
    single_seed = experiment == "celeba32_dit_3x3"
    if uncertainty not in (("single_seed",) if single_seed else ("sd", "se")):
        raise ValueError("The uncertainty convention must match the available seeds.")
    _, rows = summarize(experiment, metric, rho, alpha, n_trains)
    metric_label = "SSCD" if metric == "f_mem_sscd" else "raw-pixel"
    threshold = rf"$\alpha={alpha:g}$" if alpha is not None else rf"$\rho={rho:g}$"
    model_label = "CelebA32 Small-DiT" if single_seed else "CelebA32 U-Net"
    cells = {(r["n_train"], r["prediction_target"], r["loss_target"]): r for r in rows}
    # Match the paper's wide cells while keeping the same full PDF width as M2.
    with plt.rc_context({"font.size": 8}):
        figure, axes = plt.subplots(1, len(n_trains),
                                    figsize=(FIGURE_WIDTH_INCHES, 2.5), squeeze=False)
        figure.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.70, wspace=0.25)
        for index, (axis, n_train) in enumerate(zip(axes[0], n_trains)):
            axis.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), aspect="auto")
            for i, prediction in enumerate(TARGETS):
                for j, loss in enumerate(TARGETS):
                    row = cells[(n_train, prediction, loss)]
                    diagonal = prediction == loss
                    axis.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        facecolor=TINTS[prediction] if diagonal else "#FAFAFA",
                        edgecolor=TARGET_COLORS[prediction] if diagonal else "#D8D8D8",
                        linewidth=1.15 if diagonal else 0.65,
                        zorder=2 if diagonal else 1,
                    ))
                    # SD describes seed variation; SE = SD / sqrt(number of seeds).
                    divisor = math.sqrt(row["n_seeds"]) if uncertainty == "se" else 1
                    mean = 100 * row[f"{metric}_mean"]
                    mem_text = f"{mean:.1f}"
                    fid_text = f"{row['inception_fid_mean']:.1f}"
                    if not single_seed:
                        mem_text += rf"\pm{100 * row[f'{metric}_std'] / divisor:.1f}"
                        fid_text += rf"\pm{row['inception_fid_std'] / divisor:.1f}"
                    axis.text(j, i - 0.13, f"${mem_text}$",
                              ha="center", va="center", fontsize=11.2, color="#111111")
                    axis.text(j, i + 0.23,
                              rf"FID$\downarrow$ ${fid_text}$",
                              ha="center", va="center", fontsize=7.4, color="#686868")
            axis.set_xticks(range(3), [TARGET_LABELS[t] for t in TARGETS])
            axis.set_yticks(range(3), [TARGET_LABELS[t] for t in TARGETS])
            axis.tick_params(axis="both", length=0, pad=4, labelsize=10)
            axis.xaxis.tick_top()
            axis.xaxis.set_label_position("top")
            axis.set_xlabel("Loss target", fontsize=9, labelpad=7)
            for tick, target in zip(axis.get_xticklabels(), TARGETS):
                tick.set_color(TARGET_COLORS[target])
            for tick, target in zip(axis.get_yticklabels(), TARGETS):
                tick.set_color(TARGET_COLORS[target])
            axis.set_title(rf"({chr(97 + index)})  $N={n_train:,}$",
                           loc="left", fontsize=9, pad=36)
            axis.spines[:].set_visible(False)
        axes[0, 0].set_ylabel("Prediction target", labelpad=8, fontsize=9)
        figure.text(0.53, 0.13,
                    rf"Large numbers: {metric_label} $f_{{\mathrm{{mem}}}}$ (%) · gray numbers: FID $\downarrow$",
                    ha="center", va="center", fontsize=8)
        seed_note = ("Small-DiT; seed 0 only; no across-seed uncertainty."
                     if single_seed else rf"Mean $\pm$ {uncertainty.upper()} over 3 seeds.")
        figure.text(0.53, 0.05,
                    f"{threshold}; best-FID checkpoint per seed. {seed_note}",
                    ha="center", va="center", fontsize=7.5, color="#6B7280")
        output.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".svg"):
            figure.savefig(output.with_suffix(suffix))
        plt.close(figure)
    uncertainty_name = "sample standard deviation" if uncertainty == "sd" else "standard error"
    threshold_note = (f"held-out-calibrated alpha={alpha} (run-specific rho)"
                      if alpha is not None else f"fixed rho={rho}")
    uncertainty_note = (
        "Only seed 0 is available; neither metric has an across-seed uncertainty estimate. "
        if single_seed else
        f"Both metrics report mean plus/minus {uncertainty_name} over three seeds. "
    )
    caption = (
        f"Memorization at each seed's best FID for {model_label}. "
        f"Panels show training sizes {', '.join(map(str, n_trains))}. "
        "Rows identify prediction targets and columns loss targets, ordered x, epsilon, v. "
        f"Large values show {metric_label} f_mem in percent at {threshold_note}. "
        "Gray values beneath show paired Inception FID. "
        f"{uncertainty_note}"
        "Each seed selects its own lowest saved FID, with exact ties resolved by the earlier step. "
        "Diagonal tints identify x/x (red), epsilon/epsilon (blue), and v/v (green); "
        "off-diagonal cells are neutral. These are per-run best-quality comparisons, "
        "not comparisons at an equal FID. Layout and colors follow the supplied paper PDF, "
        "Figure 3 on page 7; full PDF width matches M2. "
        f"Inputs: plot_data/{experiment} only.\n"
    )
    output.with_name(output.name + ".caption.txt").write_text(caption, encoding="utf-8")
