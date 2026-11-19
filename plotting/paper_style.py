"""Shared paper typography and parameter labels for every figure template."""

import matplotlib


RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
}
# Bold italic letters, including Greek epsilon, share one math family.
TARGET_MATH = {target: rf"\boldsymbol{{{symbol}}}"
               for target, symbol in (("x", "x"), ("eps", r"\epsilon"), ("v", "v"))}
TARGET_LABELS = {target: f"${symbol}$" for target, symbol in TARGET_MATH.items()}
TARGET_PAIR_LABELS = {target: f"${symbol}/{symbol}$" for target, symbol in TARGET_MATH.items()}

# The standalone LaTeX alternative uses the same text and math families.
LATEX_PREAMBLE = (
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amsmath}",
    r"\usepackage[notext]{stix}",
    r"\usepackage{DejaVuSans}",
    r"\renewcommand{\familydefault}{\sfdefault}",
)


def apply_style():
    """Set typography without changing template-specific sizes or geometry."""
    matplotlib.rcParams.update(RC_PARAMS)
