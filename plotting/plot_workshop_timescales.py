"""Reuse the workshop Figure 4 artwork with an explicit historical caption."""

import shutil
from pathlib import Path

from paper_style import TARGET_PAIR_LABELS


SOURCE = Path("plotting/references/workshop_figure4.pdf")


def caption(latex=False):
    text = (
        "Generation and memorization timescales in the original CelebA32 grayscale "
        "U-Net study, reproduced from our workshop Figure 4. Training sizes are "
        "N=256, 512, 1024, 2048, 4096, 8192, and 16384. "
        "Left: tau_gen is the first saved checkpoint with FID <= 1.10 times that "
        "run's minimum FID. Middle: tau_mem is the first saved checkpoint with "
        "F_mem >= 0.01, measured using raw-pixel nearest-neighbour ratios at rho=0.5. "
        "Right: Delta_tau=tau_mem-tau_gen; the dotted line marks zero. "
        "Colors identify x/x (red), epsilon/epsilon (blue), and v/v (green). "
        "Times are in training steps; error bars show +/- one standard error over "
        "three seeds. The original analysis assigns the final checkpoint T=10^6 "
        "to runs without an observed memorization onset. Consequently, plateaus "
        "at T and the associated gaps include capped, unobserved onsets; they "
        "do not establish measured onset times or continuous generation windows. "
        "This broader training-size sweep retains its original rho=0.5, first-1% "
        "onset, and standard-error conventions, unlike the current rho=0.3 "
        "results with sample standard deviations."
    )
    if latex:
        for plain, math_text in (
            ("N=256, 512, 1024, 2048, 4096, 8192, and 16384",
             r"$N\in\{256,512,1024,2048,4096,8192,16384\}$"),
            ("Delta_tau=tau_mem-tau_gen", r"$\Delta\tau=\tau_{\mathrm{mem}}-\tau_{\mathrm{gen}}$"),
            ("tau_gen", r"$\tau_{\mathrm{gen}}$"), ("tau_mem", r"$\tau_{\mathrm{mem}}$"),
            ("FID <= 1.10", r"$\mathrm{FID}\leq1.10$"),
            ("F_mem >= 0.01", r"$F_{\mathrm{mem}}\geq0.01$"),
            ("rho=0.5", r"$\rho=0.5$"), ("rho=0.3", r"$\rho=0.3$"),
            ("T=10^6", r"$T=10^6$"), ("at T", r"at $T$"),
            ("+/-", r"$\pm$"), ("1%", r"1\%"),
            ("x/x", TARGET_PAIR_LABELS["x"]),
            ("epsilon/epsilon", TARGET_PAIR_LABELS["eps"]),
            ("v/v", TARGET_PAIR_LABELS["v"]),
        ):
            text = text.replace(plain, math_text)
    return text


def render(output):
    """Copy vector artwork exactly; do not reconstruct historical numerical data."""
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, output.with_suffix(".pdf"))
    # This SVG belonged to the superseded current-data M5 and remains archived.
    output.with_suffix(".svg").unlink(missing_ok=True)
    for suffix, latex in ((".caption.txt", False), (".caption.tex", True)):
        output.with_suffix(suffix).write_text(caption(latex) + "\n", encoding="utf-8")
