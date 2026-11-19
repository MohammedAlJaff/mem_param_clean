"""Render the best-FID memorization table as a 3x3 grid and LaTeX document."""

from plot_f_mem_grid import (
    DEFAULT_FIGURE_ROOT,
    N_TRAINS,
    PANEL_WIDTH_INCHES,
    TARGET_COLORS,
    TARGET_LABELS,
    TARGETS,
    plt,
    set_panel_width,
)
from summarize_f_mem_at_best_fid import RHO, summarize
from paper_style import LATEX_PREAMBLE


def plot_grid(cells, output_stem):
    """Stack the four training sizes within each prediction/loss panel."""
    figure, axes = plt.subplots(3, 3, figsize=(15, 8))
    for i, prediction in enumerate(TARGETS):
        for j, loss in enumerate(TARGETS):
            axis = axes[i, j]
            axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_color("#CBD2D8")
            axis.text(0.035, 0.9, r"$N_{\mathrm{train}}$", fontsize=9, va="center")
            axis.text(
                0.965, 0.9, "$f_{\\mathrm{mem}}$ ± SD\n(FID ± SD)",
                ha="right", va="center", fontsize=9, color="#475569",
            )
            axis.axhline(0.81, color="#CBD2D8", linewidth=0.8)
            for k, n_train in enumerate(N_TRAINS):
                row = cells[(n_train, prediction, loss)]
                y = 0.7 - 0.2 * k
                axis.text(
                    0.035, y, f"{n_train:,}", color="#17212B",
                    fontsize=9.5, fontweight="bold", va="center",
                )
                text = (
                    f"{row['f_mem_raw_pixels_mean']:.4f} ± "
                    f"{row['f_mem_raw_pixels_std']:.4f}\n"
                    f"({row['inception_fid_mean']:.2f} ± {row['inception_fid_std']:.2f})"
                )
                axis.text(
                    0.965, y, text, ha="right", va="center", fontsize=9.3,
                    color=TARGET_COLORS[prediction],
                )
                if k < len(N_TRAINS) - 1:
                    axis.axhline(y - 0.1, color="#EDF0F2", linewidth=0.6)
            if i == 0:
                axis.set_title(f"Loss target {TARGET_LABELS[loss]}", fontsize=11, pad=10)
            if j == 0:
                axis.set_ylabel(f"Predict {TARGET_LABELS[prediction]}", fontsize=11,
                                labelpad=10, color=TARGET_COLORS[prediction])

    figure.suptitle(
        "Memorization at each seed's best FID\n"
        rf"CelebA32 U-Net · raw pixels · $\rho={RHO:g}$",
        fontsize=13, y=0.985,
    )
    figure.text(
        0.5, 0.025,
        "Three-seed means ± sample SD. Parentheses: best Inception FID.\n"
        "Each seed selects its own checkpoint.",
        ha="center", fontsize=9,
    )
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.06, top=0.88, wspace=0.07, hspace=0.13
    )
    set_panel_width(figure, axes[0, 0])
    for extension in (".pdf", ".svg"):
        path = output_stem.with_suffix(extension)
        figure.savefig(path, dpi=180)
        print(f"Wrote {path.resolve()}")
    plt.close(figure)


def write_latex(cells, path):
    """Write a standalone LaTeX table with the same 36 displayed entries."""
    lines = [
        r"\documentclass[10pt]{article}",
        *LATEX_PREAMBLE,
        rf"\usepackage[paperwidth={3 * PANEL_WIDTH_INCHES + 2:g}in,paperheight=10.5in,margin=0.45in]{{geometry}}",
        r"\usepackage{xcolor}",
        r"\usepackage{graphicx}",
        r"\pagestyle{empty}",
    ]
    for target, color in TARGET_COLORS.items():
        lines.append(rf"\definecolor{{target{target}}}{{HTML}}{{{color[1:]}}}")
    lines.extend([
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries Memorization at each seed's best FID}\\[6pt]",
        rf"CelebA32 U-Net; raw pixels; $\rho={RHO:g}$.\\[12pt]",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{c|c|c|c}",
        " & " + " & ".join(f"Loss target {TARGET_LABELS[t]}" for t in TARGETS)
        + r" \\ \hline",
    ])
    for prediction in TARGETS:
        panels = []
        for loss in TARGETS:
            panel = [
                rf"\begin{{minipage}}[c][2.4in][c]{{{PANEL_WIDTH_INCHES:g}in}}\centering\small",
                r"\begin{tabular}{@{}r@{\quad}l@{}}",
                r"$N_{\mathrm{train}}$ & $f_{\mathrm{mem}}\pm\mathrm{SD}$ \\",
                r" & $(\mathrm{FID}\pm\mathrm{SD})$ \tabularnewline",
            ]
            for n_train in N_TRAINS:
                row = cells[(n_train, prediction, loss)]
                panel.append(
                    rf"\textbf{{{n_train:,}}} & \textcolor{{target{prediction}}}{{"
                    rf"${row['f_mem_raw_pixels_mean']:.4f}\pm"
                    rf"{row['f_mem_raw_pixels_std']:.4f}$}}"
                    + r" \\[-3pt]" + "\n"
                    + rf" & \textcolor{{target{prediction}}}{{$"
                    rf"({row['inception_fid_mean']:.2f}\pm{row['inception_fid_std']:.2f})$}}"
                    + r" \\[5pt]",
                )
            panel.extend([r"\end{tabular}", r"\end{minipage}"])
            panels.append("\n".join(panel))
        lines.append(
            rf"\rotatebox{{90}}{{\textcolor{{target{prediction}}}{{Predict {TARGET_LABELS[prediction]}}}}} & " + " &\n".join(panels)
            + r" \\ \hline"
        )
    lines.extend([
        r"\end{tabular}\\[12pt]",
        r"Three-seed means $\pm$ sample SD. Parentheses: best Inception FID.\\",
        r"Each seed selects its own lowest-FID checkpoint; exact ties use the earlier step.",
        r"\end{center}",
        r"\end{document}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path.resolve()}")


def main():
    _, summary_rows = summarize()
    cells = {
        (row["n_train"], row["prediction_target"], row["loss_target"]): row
        for row in summary_rows
    }
    DEFAULT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output_stem = DEFAULT_FIGURE_ROOT / "fmem_at_best_fid__rho_0p3__grid"
    plot_grid(cells, output_stem)
    write_latex(cells, output_stem.with_name(f"{output_stem.name}__latex.tex"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
