"""M4 / manuscript Table 2: paired pixel and feature memorization at best quality."""

import csv
import math
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from celeba32_3x3.plot_f_mem_grid import TARGETS
from paper_style import TARGET_PAIR_LABELS
from plot_fid_by_training_size import EXPERIMENTS, load_records


DATASETS = ("celeba32_unet_3x3", "celeba64_rgb_diagonal",
            "mnist32_diagonal", "celeba32_dit_3x3")
N_TRAIN, RHO, ALPHA = 2048, 0.3, 0.01
RAW_METRIC = "f_mem_raw_pixels"
MANUSCRIPT_WIDTH_INCHES = 6.75
TABLE_COLORS = {"x": "red", "eps": "blue", "v": "green!60!black"}


def dataset_settings(experiment):
    """Reuse the accepted quality protocols and their full run coverage."""
    if experiment == "celeba32_unet_3x3":
        return {**EXPERIMENTS["celeba32_dit_3x3"], "title": "CelebA32 U-Net",
                "model_arch": "unet", "seeds": (0, 1, 2)}
    return EXPERIMENTS[experiment]


def feature_metric(experiment):
    return "f_mem_mnist_jointae32_v1" if experiment == "mnist32_diagonal" else "f_mem_sscd"


def select_paired_metrics(quality_rows, evaluation_rows, feature):
    """Choose each run's best quality first, then pair both memorization readouts."""
    best = {}
    for row in quality_rows:
        previous = best.get(row["run_id"])
        if previous is None or (row["value"], row["step"]) < (previous["value"], previous["step"]):
            best[row["run_id"]] = row
    paired = {run_id: {} for run_id in best}
    for row in evaluation_rows:
        run_id, metric = row["run_id"], row["metric"]
        if run_id not in best or metric not in (RAW_METRIC, feature):
            continue
        if int(row["step"]) != best[run_id]["step"]:
            continue
        if metric == RAW_METRIC:
            if row["space"] != "raw_pixels" or row["alpha"] or not row["rho"]:
                continue
            if not math.isclose(float(row["rho"]), RHO, abs_tol=1e-12):
                continue
        else:
            if row["space"] or not row["alpha"] or not row["rho"]:
                continue
            if not math.isclose(float(row["alpha"]), ALPHA, abs_tol=1e-12):
                continue
        value = float(row["value"])
        if row["sample_count"] != "10000" or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid memorization record: {run_id}, {metric}")
        if metric in paired[run_id]:
            raise ValueError(f"Duplicate memorization record: {run_id}, {metric}")
        paired[run_id][metric] = value
    selected = []
    for run_id, quality in best.items():
        if set(paired[run_id]) != {RAW_METRIC, feature}:
            raise ValueError(f"Missing paired memorization at best quality: {run_id}")
        selected.append({
            **{k: quality[k] for k in ("experiment", "run_id", "n_train", "seed",
                                      "prediction_space", "loss_space", "step")},
            quality["metric"]: quality["value"], **paired[run_id],
        })
    return selected


def load_table():
    """Read only portable exports; preserve per-seed selection before aggregation."""
    selected, cells = [], {}
    for experiment in DATASETS:
        settings = dataset_settings(experiment)
        quality = [row for row in load_records(experiment, settings)
                   if row["n_train"] == N_TRAIN and row["prediction_space"] == row["loss_space"]]
        feature = feature_metric(experiment)
        with (Path("plot_data") / experiment / "evaluation_metrics.csv").open(newline="") as handle:
            seeds = select_paired_metrics(quality, csv.DictReader(handle), feature)
        selected.extend(seeds)
        for target in TARGETS:
            group = [r for r in seeds if r["prediction_space"] == target]
            if sorted(r["seed"] for r in group) != list(settings["seeds"]):
                raise ValueError(f"Unexpected seed coverage: {experiment}, {target}")
            cell = {"n_seeds": len(group)}
            for metric in (RAW_METRIC, feature, settings["metric"]):
                values = [r[metric] for r in group]
                cell[f"{metric}_mean"] = statistics.fmean(values)
                cell[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            cells[(experiment, target)] = cell
    return selected, cells


def caption(latex=False):
    """Explain the representations instead of leaving the digit encoder unnamed."""
    text = (
        "Robustness across datasets and architectures at N=2048. "
        "Triplets follow x/x, epsilon/epsilon, v/v (red, blue, green). "
        "Memorization f_mem (%) and quality use each seed's lowest-FID checkpoint "
        "(lowest FD for MNIST). Values are means over three seeds; DiT uses seed 0. "
        "Raw pixels use rho=0.3. Features use SSCD image-copy-detection descriptors for "
        "CelebA and a frozen 128-dimensional encoder trained to reconstruct digit images "
        "and classify digit labels for MNIST; thresholds target a 1% held-out flagged "
        "rate (alpha=0.01). Quality is Inception FID for CelebA and Frechet distance in "
        "raw digit-encoder features for MNIST (lower is better). Underlining marks "
        "the lowest mean memorization per row and dataset."
    )
    if latex:
        for plain, math_text in (
            ("N=2048", r"$N=2048$"),
            ("x/x, epsilon/epsilon, v/v (red, blue, green)", " / ".join(
                rf"\textcolor{{{TABLE_COLORS[t]}}}{{{TARGET_PAIR_LABELS[t]}}}" for t in TARGETS)),
            ("f_mem (%)", r"$f_{\mathrm{mem}}$ (\%)"),
            ("rho=0.3", r"$\rho=0.3$"), ("alpha=0.01", r"$\alpha=0.01$"),
            ("Frechet", r"Fr\'echet"),
            ("1%", r"1\%"),
        ):
            text = text.replace(plain, math_text)
    return text


def triplet(cells, experiment, metric, memorization=True):
    """Display means only, as in the supplied original table."""
    minimum = min(cells[(experiment, t)][f"{metric}_mean"] for t in TARGETS)
    values = []
    digits = 2 if metric == "mnist_fd_jointae32_v1" else 1
    scale = 100 if memorization else 1
    for target in TARGETS:
        cell = cells[(experiment, target)]
        mean = cell[f"{metric}_mean"]
        value = f"{scale * mean:.{digits}f}"
        if memorization and mean == minimum:
            value = rf"\underline{{{value}}}"
        if memorization:
            color = TABLE_COLORS[target]
            value = rf"\textcolor{{{color}}}{{{value}}}"
        values.append(value)
    return " / ".join(values)


def render(output):
    """Write a manuscript table fragment and compile that exact fragment for preview."""
    _, cells = load_table()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".caption.txt").write_text(caption() + "\n", encoding="utf-8")
    output.with_suffix(".caption.tex").write_text(caption(latex=True) + "\n", encoding="utf-8")
    columns = [
        r"\makecell[l]{Raw pixels \\ Features \\ {\scriptsize\color{qgray}Best quality$\downarrow$}}"
    ]
    for experiment in DATASETS:
        quality_label = "FD" if experiment == "mnist32_diagonal" else "FID"
        columns.append("\n".join([
            r"\makecell{",
            triplet(cells, experiment, RAW_METRIC) + r" \\",
            triplet(cells, experiment, feature_metric(experiment)) + r" \\",
            r"{\scriptsize\color{qgray}" + quality_label + r"$\downarrow$\; "
            + triplet(cells, experiment, dataset_settings(experiment)["metric"], memorization=False)
            + "}", "}",
        ]))
    lines = [
        "% M4 / manuscript Table 2. Requires amsmath, xcolor, booktabs, makecell, colortbl.",
        "% Uses the manuscript's tablehead, tablerow, and qgray colors and fonts.",
        "% Include with \\input{M4.tex}; table numbering follows the manuscript.",
        r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption(latex=True)}}}",
        r"\label{tab:robustness}", r"\small",
        r"\renewcommand{\arraystretch}{1.55}",
        r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}cccc@{}}",
        r"\toprule", r"\rowcolor{tablehead}[0pt][0pt]",
        r"\textbf{Metric} & " + " & ".join(
            rf"\textbf{{{dataset_settings(e)['title'].replace('Small-DiT', 'DiT')}}}"
            for e in DATASETS) + r" \\",
        r"\midrule", r"\rowcolor{tablerow}[0pt][0pt]",
        "\n&\n".join(columns) + r" \\",
        r"\bottomrule", r"\end{tabular*}", r"\end{table*}",
    ]
    source = output.with_suffix(".tex")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mem_param_M4_") as temporary:
        directory = Path(temporary)
        wrapper = directory / "preview.tex"
        wrapper.write_text("\n".join([
            r"\documentclass[10pt]{article}",
            rf"\usepackage[paperwidth={MANUSCRIPT_WIDTH_INCHES + 0.24:g}in,paperheight=2.8in,margin=0.12in]{{geometry}}",
            r"\usepackage{amsmath,xcolor,booktabs,makecell,colortbl}",
            r"\definecolor{tablehead}{HTML}{F3F4F6}",
            r"\definecolor{tablerow}{HTML}{FAFAFA}",
            r"\definecolor{qgray}{gray}{0.5}",
            r"\pagestyle{empty}", r"\setcounter{table}{1}", r"\begin{document}",
            rf"\input{{{source.resolve()}}}", r"\end{document}",
        ]) + "\n", encoding="utf-8")
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 f"-output-directory={directory}", str(wrapper)],
                                capture_output=True, text=True)
        if result.returncode or "Overfull" in result.stdout:
            raise RuntimeError(result.stdout[-6000:])
        shutil.copyfile(directory / "preview.pdf", output.with_suffix(".pdf"))
