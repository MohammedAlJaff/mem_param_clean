"""List or rebuild portable paper figures by stable figure ID or type code."""

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_ROOT = Path("outputs/paper_figures")
CATALOG_PATH = Path(__file__).with_name("figure_catalog.json")
FIGURE_TYPES = {
    "QM": ("Quality matrix, 3x3", "plot_fid_by_training_size.py:plot_trajectories"),
    "QD": ("Quality diagonal, 1x3", "plot_fid_by_training_size.py:plot_trajectories"),
    "DM": ("Dual-metric matrix, 3x3", "celeba32_3x3/plot_f_mem_grid.py:render"),
    "WV": ("Velocity weighting, 2x2", "celeba32_3x3/compare_weighted_velocity.py:plot_comparison"),
    "BT": ("Best-quality matrices/tables", "celeba32_3x3/plot_f_mem_at_best_fid.py:plot_grid/write_latex; celeba32_3x3/plot_best_fid_matrices.py:render; plot_robustness_table.py:render"),
    "DT": ("Paired diagonal trajectories, 2x3", "plot_diagonal_trajectories.py:plot_grid"),
    "DS": ("Diagonal trajectories and summaries, 3+3+2", "plot_diagonal_summary.py:plot_grid"),
    "TS": ("Quality and memorization timescales, 1x3", "plot_diagonal_timescales.py:plot_grid"),
    "MS": ("Memorization sensitivity summaries/trajectories", "plot_memorization_sensitivity.py:render"),
}
def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate catalog key: {key}")
        result[key] = value
    return result


# The tracked catalog owns identities and review metadata; code owns rendering.
FIGURES = json.loads(CATALOG_PATH.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
for _spec in FIGURES.values():
    _spec["stem"] = Path(_spec["stem"])
ALIASES = {alias: code for code, spec in FIGURES.items() for alias in spec["aliases"]}
if (len({spec["stem"] for spec in FIGURES.values()}) != len(FIGURES)
        or len(ALIASES) != sum(len(spec["aliases"]) for spec in FIGURES.values())
        or set(ALIASES) & set(FIGURES)):
    raise ValueError("Figure IDs, output stems, and aliases must be unique.")
if any(not re.fullmatch(r"[MA][1-9][0-9]*(?:[-_][A-Z0-9]+)*", code) for code in FIGURES):
    raise ValueError("Use paper IDs such as M2 or M5, with descriptive suffixes for variants.")


def resolve_id(code):
    """Accept older command names while using M/A names for paper outputs."""
    return ALIASES.get(code.upper(), code.upper())


def renderer_owner(spec):
    if spec.get("renderer") == "workshop":
        return "plot_workshop_timescales.py:render"
    if spec["type"] == "BT":
        return {
            "python": "celeba32_3x3/plot_f_mem_at_best_fid.py:plot_grid",
            "latex": "celeba32_3x3/plot_f_mem_at_best_fid.py:write_latex",
            "split_n": "celeba32_3x3/plot_best_fid_matrices.py:render",
            "robustness": "plot_robustness_table.py:render",
        }[spec["renderer"]]
    return FIGURE_TYPES[spec["type"]][1]


def validate_spec(spec):
    """Reject metadata that would misdescribe the currently supported renderers."""
    if spec["placement"] not in ("main", "appendix", "unassigned"):
        raise ValueError("Placement must be main, appendix, or unassigned.")
    if spec["status"] not in ("draft", "approved", "rejected"):
        raise ValueError("Review status must be draft, approved, or rejected.")
    if spec["status"] == "approved" and not spec["approved_build"]:
        raise ValueError("Approval must name an exact build; only record explicit user approval.")
    if spec["type"] == "WV" and spec.get("renderer") not in (None, "matched_velocity"):
        raise ValueError("Weighting figures require diagonal or matched velocity comparisons.")
    if spec.get("renderer") == "workshop":
        expected = {
            "type": "TS", "datasets": [], "rho": 0.5, "uncertainty": "se",
            "selection": "workshop_timescales", "seeds": [0, 1, 2],
            "metrics": ["inception_fid", "f_mem_raw_pixels"],
            "n_trains": [256, 512, 1024, 2048, 4096, 8192, 16384],
            "onset_fraction": 0.01, "onset_rule": "first_crossing",
            "censoring": "assign_final_checkpoint", "quality_tolerance": 0.1,
            "reference_files": ["plotting/references/workshop_figure4.pdf",
                                "plotting/references/workshop_figure4.txt"],
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("Workshop reproduction metadata must describe the original analysis.")
        return
    if spec["type"] == "MS":
        from plot_memorization_sensitivity import METRICS

        representation = spec.get("renderer") == "representation"
        trajectories = spec.get("renderer") == "trajectories"
        expected = {
            "datasets": ["celeba32_unet_3x3"], "n_trains": [512, 1024, 2048, 4096],
            "seeds": [0, 1, 2], "uncertainty": "sd",
            "selection": "all_saved_checkpoints" if trajectories else "per_seed_best_fid",
            "metrics": (["f_mem_raw_pixels"] if trajectories else
                        ["inception_fid", *(METRICS if representation else ("f_mem_raw_pixels",))]),
            "alpha": 0.01 if representation else None,
            "rhos": None if representation else [0.3, 0.5, 0.7],
        }
        if (spec.get("renderer") not in ("representation", "threshold", "trajectories")
                or any(spec.get(key) != value for key, value in expected.items())):
            raise ValueError("Sensitivity metadata must match the paired metrics and thresholds.")
        return
    if spec.get("renderer") == "robustness":
        from plot_robustness_table import DATASETS

        expected = {
            "type": "BT", "datasets": list(DATASETS), "n_trains": [2048],
            "metrics": ["inception_fid", "mnist_fd_jointae32_v1", "f_mem_raw_pixels",
                        "f_mem_sscd", "f_mem_mnist_jointae32_v1"],
            "rho": 0.3, "alpha": 0.01, "selection": "per_seed_best_quality",
            "uncertainty": "none",
            "seeds": {e: [0] if e == "celeba32_dit_3x3" else [0, 1, 2] for e in DATASETS},
        }
        if any(spec.get(key) != value for key, value in expected.items()):
            raise ValueError("Robustness-table metadata must match its metrics, thresholds, and coverage.")
        return
    split = spec.get("renderer") == "split_n"
    fixed_datasets = {
        "BT": ["celeba32_unet_3x3"],
        "DT": ["celeba32_unet_3x3"], "DS": ["celeba32_unet_3x3"],
        "WV": ["celeba32_unet_3x3", "celeba32_vv_weighting"],
    }
    if split or spec["type"] == "DM":
        if spec["datasets"] not in (["celeba32_unet_3x3"], ["celeba32_dit_3x3"]):
            raise ValueError("Matrices require a supported CelebA32 experiment.")
    elif spec["type"] in fixed_datasets and spec["datasets"] != fixed_datasets[spec["type"]]:
        raise ValueError("The registered data sources do not match this renderer.")
    if spec["type"] in ("QM", "QD", "TS") and spec["datasets"] != [
            spec.get("experiment", "celeba32_unet_3x3")]:
        raise ValueError("The experiment and data-source declarations disagree.")
    selection = {"BT": "per_seed_best_fid", "TS": "timescales",
                 "DS": "trajectory_and_per_seed_best_fid"}.get(spec["type"], "all_saved_checkpoints")
    if spec["selection"] != selection:
        raise ValueError("The checkpoint selection does not match this renderer.")
    single_seed = spec["datasets"] == ["celeba32_dit_3x3"]
    expected_uncertainty = "single_seed" if single_seed else "sd"
    if spec["uncertainty"] not in (("sd", "se") if split and not single_seed else (expected_uncertainty,)):
        raise ValueError("This renderer does not support the declared uncertainty.")
    quality = "mnist_fd_jointae32_v1" if spec["datasets"][0].startswith("mnist") else "inception_fid"
    expected_metrics = [quality] + (["f_mem_raw_pixels"] if spec["type"] not in ("QM", "QD") else [])
    sscd = split and spec["metrics"] == ["inception_fid", "f_mem_sscd"]
    supported_metric = (
        spec.get("alpha") == 0.01 and spec.get("rho") is None if sscd else
        spec["metrics"] == expected_metrics and spec.get("rho", 0.3) == 0.3
        and spec.get("alpha") is None
    )
    if not supported_metric:
        raise ValueError("Add a renderer for the new metric; changing catalog labels is insufficient.")
    expected_sizes = [512, 2048] if spec["type"] in ("QD", "WV") else [512, 1024, 2048, 4096]
    if (not split and spec["n_trains"] != expected_sizes) or (split and (
            not spec["n_trains"] or len(set(spec["n_trains"])) != len(spec["n_trains"])
            or not set(spec["n_trains"]) <= set(expected_sizes))):
        raise ValueError("Training sizes do not match the renderer's supported coverage.")
    if spec["seeds"] != ([0] if single_seed else [0, 1, 2]):
        raise ValueError("Seed coverage does not match the renderer.")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_identity(spec):
    """Conservatively fingerprint all plot code, selected numerical inputs, and environment."""
    import matplotlib
    import numpy

    return {
        "spec": {k: str(v) if isinstance(v, Path) else v for k, v in spec.items()
                 if k not in ("placement", "status", "approved_build", "aliases")},
        "code": {str(p): sha256(p) for p in sorted(Path("plotting").rglob("*.py"))},
        "inputs": {str(p): sha256(p) for dataset in spec["datasets"]
                   for p in (Path("plot_data") / dataset / name
                             for name in ("runs.csv", "evaluation_metrics.csv"))},
        **({"reference_artifacts": {p: sha256(Path(p)) for p in spec["reference_files"]}}
           if spec.get("reference_files") else {}),
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                        "numpy": numpy.__version__, "platform": platform.platform()},
    }


def identity_id(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def record_build(code, spec, identity):
    """Preserve each distinct build and its exact plotting source before updating aliases."""
    build_id = identity_id(identity)
    directory = OUTPUT_ROOT / "versions" / code / build_id
    receipt = directory / "build.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["id"] != code or record["identity"] != identity:
            raise ValueError(f"Archived build record changed: {receipt}")
        if sha256(directory / "plotting_source.zip") != record["source_archive_sha256"]:
            raise ValueError(f"Archived plotting source changed: {directory}")
        # The first copy of a build stays fixed, including PDF creation metadata.
        for original, artifact in record["artifacts"].items():
            saved = directory / artifact["file"]
            if sha256(saved) != artifact["sha256"]:
                raise ValueError(f"Archived artifact changed: {saved}")
            shutil.copyfile(saved, original)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for suffix in (".pdf", ".svg", ".tex", ".caption.txt", ".caption.tex"):
            original = spec["stem"].with_suffix(suffix)
            if original.exists():
                saved = directory / (code + suffix)
                shutil.copyfile(original, saved)
                artifacts[str(original)] = {"file": saved.name, "sha256": sha256(saved)}
        with zipfile.ZipFile(directory / "plotting_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for source in identity["code"]:
                archive.write(source, source)
            for source in identity.get("reference_artifacts", {}):
                archive.write(source, source)
            archive.write(CATALOG_PATH, "plotting/figure_catalog.json")
        record = {"id": code, "build_id": build_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "renderer": renderer_owner(spec), "identity": identity, "artifacts": artifacts,
                  "source_archive_sha256": sha256(directory / "plotting_source.zip")}
        receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    pdf = spec["stem"].with_suffix(".pdf")
    if pdf.resolve() != (OUTPUT_ROOT / f"{code}.pdf").resolve():
        shutil.copyfile(pdf, OUTPUT_ROOT / f"{code}.pdf")
    shutil.copyfile(receipt, OUTPUT_ROOT / f"{code}.build.json")
    print(f"{code}@{build_id} [{spec['type']}, {spec['placement']}, {spec['status']}] "
          f"{OUTPUT_ROOT / (code + '.pdf')}", flush=True)


def figure_state(code, spec):
    """Keep old unrecorded PDFs distinct from verified builds and exact approvals."""
    receipt = OUTPUT_ROOT / f"{code}.build.json"
    if not receipt.exists():
        existing = any((OUTPUT_ROOT / f"{name}.pdf").exists() for name in (code, *spec["aliases"]))
        return ("needs-review" if spec["status"] == "approved" else spec["status"],
                "legacy" if existing else "unbuilt", None)
    record = json.loads(receipt.read_text())
    current_pdf = OUTPUT_ROOT / f"{code}.pdf"
    expected_pdf = next(a["sha256"] for a in record["artifacts"].values() if a["file"] == f"{code}.pdf")
    current = (record["id"] == code and record["build_id"] == identity_id(build_identity(spec))
               and current_pdf.exists() and sha256(current_pdf) == expected_pdf)
    approved = spec["approved_build"] == record["build_id"] and current
    status = "needs-review" if spec["status"] == "approved" and not approved else spec["status"]
    return status, "current" if current else "stale", record["build_id"]



def build(figure_id):
    """Render from plot_data, preserving all source and derived CSV files."""
    # Existing direct-script entrypoints import their sibling modules by name.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "celeba32_3x3"))
    figure_id = resolve_id(figure_id)
    spec = FIGURES[figure_id]
    validate_spec(spec)
    identity = build_identity(spec)
    stem = spec["stem"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    if spec.get("renderer") == "workshop":
        from plot_workshop_timescales import render

        render(stem)
    elif spec["type"] == "MS":
        from plot_memorization_sensitivity import render

        render(stem, spec["renderer"])
    elif spec["type"] in ("QM", "QD"):
        from plot_fid_by_training_size import EXPERIMENTS, load_records, plot_trajectories, summarize

        experiment = spec["experiment"]
        if experiment == "celeba32_unet_3x3":
            # Same matrix coverage as DiT, with U-Net and all three seeds.
            settings = {**EXPERIMENTS["celeba32_dit_3x3"], "title": "CelebA32 U-Net",
                        "model_arch": "unet", "seeds": (0, 1, 2)}
        else:
            settings = EXPERIMENTS[experiment]
        records = load_records(experiment, settings)
        plot_trajectories(settings, records, summarize(records), stem)
    elif spec["type"] == "DM":
        from plot_f_mem_grid import render

        render(stem, experiment=spec["datasets"][0])
    elif spec["type"] == "WV":
        from compare_weighted_velocity import load_comparison, plot_comparison

        matched = spec.get("renderer") == "matched_velocity"
        plot_comparison(*load_comparison(matched=matched), output=stem, matched=matched)
    elif spec["type"] == "DT":
        from plot_diagonal_trajectories import render

        render(stem)
    elif spec["type"] == "DS":
        from plot_diagonal_summary import render

        render(stem)
    elif spec["type"] == "TS":
        from plot_diagonal_timescales import render

        render(stem, experiment=spec.get("experiment", "celeba32_unet_3x3"))
    elif spec["type"] == "BT" and spec["renderer"] == "robustness":
        from plot_robustness_table import render

        render(stem)
    elif spec["type"] == "BT" and spec["renderer"] == "split_n":
        from plot_best_fid_matrices import render

        render(stem, spec["n_trains"], spec["uncertainty"],
               experiment=spec["datasets"][0], metric=spec["metrics"][1],
               rho=spec.get("rho"), alpha=spec.get("alpha"))
    elif spec["type"] == "BT":
        from plot_f_mem_at_best_fid import plot_grid, write_latex
        from summarize_f_mem_at_best_fid import summarize

        _, rows = summarize()
        cells = {(row["n_train"], row["prediction_target"], row["loss_target"]): row for row in rows}
        if spec["renderer"] == "python":
            plot_grid(cells, stem)
        else:
            source = stem.with_suffix(".tex")
            write_latex(cells, source)
            with tempfile.TemporaryDirectory(prefix="mem_param_latex_") as temporary:
                subprocess.run([
                    "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    f"-output-directory={temporary}", str(source.resolve()),
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                shutil.copyfile(Path(temporary) / stem.with_suffix(".pdf").name, stem.with_suffix(".pdf"))
    if identity != build_identity(spec):
        raise RuntimeError("Plot code or input data changed during rendering; rebuild before review.")
    record_build(figure_id, spec, identity)


def write_index():
    """Show placement, approval, provenance, and the renderer in one shared view."""
    lines = ["# Generated paper figures", "",
             "Source: [tracked catalog](../../plotting/figure_catalog.json). "
             "M1-M6 identify the main items; A1-A5 identify appendix groups.", "",
             "Only generated items appear here. See [paper review list](README.txt) "
             "for the planned manuscript items. Approval applies only to an explicitly approved build. "
             "Legacy PDFs have no build receipt; rebuild to establish provenance.", "",
             "| ID | Variant | Placement | Review | Build | PDF | Code |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for code, spec in FIGURES.items():
        status, state, build_id = figure_state(code, spec)
        if state == "unbuilt":
            continue
        paths = [OUTPUT_ROOT / f"{name}.pdf" for name in (code, *spec["aliases"])]
        pdf = next((f"[PDF]({p.name})" for p in paths if p.exists()), "Not built")
        owner = renderer_owner(spec).split(":")[0]
        build_link = f"[{build_id}](versions/{code}/{build_id}/{code}.pdf)" if build_id else ""
        lines.append(f"| {code} | {spec['variant']} | {spec['placement']} | {status} | "
                     f"{state} {build_link} | {pdf} | [renderer](../../plotting/{owner}) |")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", type=str.upper, metavar="M2",
                        help="one or more paper IDs, such as M2, M3, or M5")
    parser.add_argument("--type", nargs="+", type=str.upper, choices=FIGURE_TYPES)
    parser.add_argument("--placement", choices=("main", "appendix", "unassigned"))
    parser.add_argument("--status", choices=("draft", "approved", "rejected", "needs-review"))
    parser.add_argument("--metric", help="filter by a canonical metric key, such as f_mem_raw_pixels")
    parser.add_argument("--all", action="store_true", help="rebuild all catalog PDFs")
    parser.add_argument("--list", action="store_true", help="list matching IDs without rendering")
    parser.add_argument("--show", type=str.upper, metavar="M2", help="show specification and code owner")
    parser.add_argument("--index", action="store_true", help="refresh the linked index without rendering")
    args = parser.parse_args(argv)
    filters = args.type or args.placement or args.status or args.metric
    if args.ids and (filters or args.all or args.show or args.index):
        parser.error("use explicit IDs or filters separately")
    if args.all and filters:
        parser.error("use --all or filters separately")
    if any(resolve_id(code) not in FIGURES for code in [*args.ids, *([args.show] if args.show else [])]):
        parser.error("unknown ID; use --list to see available IDs")
    for spec in FIGURES.values():
        validate_spec(spec)
    if args.show:
        code = resolve_id(args.show)
        spec = FIGURES[code]
        status, state, build_id = figure_state(code, spec)
        print(json.dumps({"id": code, **spec, "renderer_owner": renderer_owner(spec),
                          "effective_status": status, "build_state": state, "build_id": build_id,
                          "command": f"python plotting/figures.py {code}"}, default=str, indent=2))
        return 0
    if args.index:
        write_index()
        return 0
    ids = list(dict.fromkeys(resolve_id(code) for code in args.ids)) if args.ids else [
        code for code, spec in FIGURES.items()
        if (not args.type or spec["type"] in args.type)
        and (not args.placement or spec["placement"] == args.placement)
        and (not args.metric or args.metric in spec["metrics"])
        and (not args.status or figure_state(code, spec)[0] == args.status)
    ]
    if args.list or not (args.ids or filters or args.all):
        for code in ids:
            spec = FIGURES[code]
            status, state, _ = figure_state(code, spec)
            print(f"{code:17}  {spec['placement']:10}  "
                  f"{status:12}  {state:7}  {spec['variant']}  [{spec['uncertainty']}]")
        print("\nShared width/palette: plotting/celeba32_3x3/plot_f_mem_grid.py")
        print("Shared typography/parameter labels: plotting/paper_style.py")
        for code, (title, owner) in FIGURE_TYPES.items():
            print(f"{code}: {title}; plotting/{owner}")
        return 0
    if not ids:
        parser.error("no figures match the requested filters")
    for figure_id in ids:
        build(figure_id)
    write_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
