#!/usr/bin/env python3
"""Validate and combine raw-first feature-space f_mem records."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PRIMARY_SPACES = ("full_pixels", "sscd", "lpips", "dinov2")
LEARNED_SPACES = ("sscd", "lpips", "dinov2")
SENSITIVITY_SPACES = ("sscd_288", "lpips_64", "dinov2_518")
PARAMETERIZATIONS = ("eps", "v", "x")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def collect_rows(output_root):
    rows = []
    for path in sorted(Path(output_root).glob("tasks/*/checkpoint_metrics.jsonl")):
        rows.extend(read_jsonl(path))
    keys = [
        (row["seed"], row["parameterization"], row["step"], row["space"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate seed/parameterization/step/space records.")
    return rows


def validate_rows(rows, manifest):
    task_count = len(manifest["tasks"])
    expected_primary = task_count * 3 * len(PRIMARY_SPACES)
    primary = [row for row in rows if row["space"] in PRIMARY_SPACES]
    learned = [row for row in primary if row["space"] in LEARNED_SPACES]
    sensitivity = [row for row in rows if row["space"] in SENSITIVITY_SPACES]
    if len(primary) != expected_primary:
        raise RuntimeError(f"Expected {expected_primary} primary rows, found {len(primary)}.")
    if len(learned) != task_count * 3 * len(LEARNED_SPACES):
        raise RuntimeError("Learned-space record count is incomplete.")
    if len(sensitivity) != 9:
        raise RuntimeError(f"Expected nine seed-0 sensitivity rows, found {len(sensitivity)}.")
    reference_hashes = {row["reference_artifact_sha256"] for row in rows}
    if len(reference_hashes) != 1:
        raise RuntimeError("Rows do not share one frozen validation artifact.")
    for row in rows:
        if row["sample_count"] != 10_000:
            raise RuntimeError("A feature evaluation does not contain 10,000 samples.")
        if row["calibration_count"] != 1024 or row["validation_count"] != 8976:
            raise RuntimeError("Held-out calibration/validation partition is incorrect.")
        diagnostics = row["raw_sample_stats"]
        if diagnostics["finite_fraction"] != 1.0:
            raise RuntimeError("A generated bank contains nonfinite values.")
        if not math.isfinite(float(row["fid_10k"])):
            raise RuntimeError("A FID-10k value is nonfinite.")
        if len(row["heldout_curve"]) != 5:
            raise RuntimeError("A held-out percentile curve is incomplete.")
    return {
        "task_count": task_count,
        "primary_records": len(primary),
        "learned_space_records": len(learned),
        "sensitivity_records": len(sensitivity),
        "reference_artifact_sha256": next(iter(reference_hashes)),
    }


def curve_rows(rows):
    result = []
    for row in rows:
        base = {
            key: row[key]
            for key in (
                "seed",
                "parameterization",
                "step",
                "space",
                "primary_space",
                "fid_10k",
                "generated_tensor_sha256",
            )
        }
        for point in row["heldout_curve"]:
            result.append({**base, **point})
    return result


def aggregate_curves(curves):
    grouped = defaultdict(list)
    for row in curves:
        if not row["primary_space"]:
            continue
        key = (
            row["parameterization"],
            row["step"],
            row["space"],
            row["heldout_quantile"],
        )
        grouped[key].append(row)
    aggregate = []
    for key, selected in sorted(grouped.items()):
        values = [float(row["generated_f_mem"]) for row in selected]
        validation = [float(row["validation_rate"]) for row in selected]
        record = {
            "parameterization": key[0],
            "step": key[1],
            "space": key[2],
            "heldout_quantile": key[3],
            "seed_count": len(values),
            "mean_generated_f_mem": statistics.mean(values),
            "std_generated_f_mem": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_validation_rate": statistics.mean(validation),
        }
        for row in selected:
            record[f"seed_{row['seed']}_generated_f_mem"] = row["generated_f_mem"]
            record[f"seed_{row['seed']}_rho"] = row["rho"]
        aggregate.append(record)
    return aggregate


def paired_differences(curves):
    indexed = {
        (
            row["seed"],
            row["step"],
            row["space"],
            row["heldout_quantile"],
            row["parameterization"],
        ): row
        for row in curves
        if row["primary_space"]
    }
    result = []
    seeds = sorted({row["seed"] for row in curves})
    steps = sorted({row["step"] for row in curves})
    quantiles = sorted({row["heldout_quantile"] for row in curves})
    for step, space, quantile, pair in itertools.product(
        steps,
        PRIMARY_SPACES,
        quantiles,
        (("eps", "v"), ("eps", "x"), ("v", "x")),
    ):
        differences = []
        record = {
            "step": step,
            "space": space,
            "heldout_quantile": quantile,
            "left": pair[0],
            "right": pair[1],
        }
        for seed in seeds:
            left = indexed[(seed, step, space, quantile, pair[0])]
            right = indexed[(seed, step, space, quantile, pair[1])]
            value = float(left["generated_f_mem"]) - float(right["generated_f_mem"])
            record[f"seed_{seed}_difference"] = value
            differences.append(value)
        record["mean_difference"] = statistics.mean(differences)
        record["std_difference"] = (
            statistics.stdev(differences) if len(differences) > 1 else 0.0
        )
        result.append(record)
    return result


def q_point(row, quantile=0.01):
    for point in row["heldout_curve"]:
        if math.isclose(float(point["heldout_quantile"]), quantile):
            return point
    raise KeyError(f"Missing held-out quantile {quantile}.")


def agreement_rows(rows):
    import torch
    from scipy.stats import spearmanr

    grouped = defaultdict(dict)
    for row in rows:
        if row["space"] in PRIMARY_SPACES:
            grouped[(row["seed"], row["parameterization"], row["step"])][
                row["space"]
            ] = row
    result = []
    for bank, by_space in sorted(grouped.items()):
        for left, right in itertools.combinations(PRIMARY_SPACES, 2):
            left_row, right_row = by_space[left], by_space[right]
            left_r = torch.load(
                left_row["generated_nn_stats"], map_location="cpu", weights_only=False
            )["r"].numpy()
            right_r = torch.load(
                right_row["generated_nn_stats"], map_location="cpu", weights_only=False
            )["r"].numpy()
            correlation = float(spearmanr(left_r, right_r).statistic)
            left_set = set((left_r < q_point(left_row)["rho"]).nonzero()[0].tolist())
            right_set = set((right_r < q_point(right_row)["rho"]).nonzero()[0].tolist())
            union = left_set | right_set
            result.append(
                {
                    "seed": bank[0],
                    "parameterization": bank[1],
                    "step": bank[2],
                    "left_space": left,
                    "right_space": right,
                    "spearman_r": correlation,
                    "memorized_jaccard_q0.01": (
                        len(left_set & right_set) / len(union) if union else 1.0
                    ),
                    "left_memorized_count": len(left_set),
                    "right_memorized_count": len(right_set),
                }
            )
    return result


def ordering_rows(curves):
    grouped = defaultdict(list)
    for row in curves:
        if row["primary_space"] and math.isclose(row["heldout_quantile"], 0.01):
            grouped[(row["seed"], row["step"], row["space"])].append(row)
    result = []
    for key, selected in sorted(grouped.items()):
        selected.sort(key=lambda row: (row["generated_f_mem"], row["parameterization"]))
        result.append(
            {
                "seed": key[0],
                "step": key[1],
                "space": key[2],
                "ascending_parameterization_order": " < ".join(
                    row["parameterization"] for row in selected
                ),
                "ordered_f_mem": [row["generated_f_mem"] for row in selected],
            }
        )
    return result


def plot_summary(curves, path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in curves
        if row["primary_space"] and math.isclose(row["heldout_quantile"], 0.01)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, space in zip(axes.flat, PRIMARY_SPACES):
        for parameterization in PARAMETERIZATIONS:
            rows = [
                row
                for row in selected
                if row["space"] == space
                and row["parameterization"] == parameterization
            ]
            for seed in sorted({row["seed"] for row in rows}):
                seed_rows = sorted(
                    (row for row in rows if row["seed"] == seed),
                    key=lambda row: row["step"],
                )
                axis.plot(
                    [row["step"] for row in seed_rows],
                    [row["generated_f_mem"] for row in seed_rows],
                    marker="o",
                    alpha=0.35,
                )
            means = []
            for step in sorted({row["step"] for row in rows}):
                values = [
                    row["generated_f_mem"] for row in rows if row["step"] == step
                ]
                means.append((step, statistics.mean(values)))
            axis.plot(
                [item[0] for item in means],
                [item[1] for item in means],
                marker="o",
                linewidth=2.4,
                label=parameterization,
            )
        axis.set_title(space)
        axis.set_xlabel("gradient-update step")
        axis.set_ylabel("f_mem at held-out q=0.01")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def combine(output_root, manifest_path):
    output_root = Path(output_root).expanduser().resolve()
    manifest = read_json(manifest_path)
    rows = collect_rows(output_root)
    summary = validate_rows(rows, manifest)
    curves = curve_rows(rows)
    aggregates = aggregate_curves(curves)
    differences = paired_differences(curves)
    agreements = agreement_rows(rows)
    orderings = ordering_rows(curves)
    combined = output_root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    write_csv(combined / "raw_metric_records.csv", rows)
    write_csv(combined / "percentile_curves.csv", curves)
    write_csv(combined / "seed_aggregates.csv", aggregates)
    write_csv(combined / "paired_parameterization_differences.csv", differences)
    write_csv(combined / "cross_space_agreement.csv", agreements)
    write_csv(combined / "parameterization_ordering.csv", orderings)
    pixel_rows = []
    for row in rows:
        if row["space"] != "full_pixels":
            continue
        for rho, value in row["pixel_f_mem"].items():
            pixel_rows.append(
                {
                    "seed": row["seed"],
                    "parameterization": row["parameterization"],
                    "step": row["step"],
                    "rho": rho,
                    "f_mem": value,
                }
            )
    write_csv(combined / "pixel_canonical_rhos.csv", pixel_rows)
    plot_summary(curves, combined / "feature_f_mem_q0.01.png")
    summary.update(
        {
            "curve_rows": len(curves),
            "aggregate_rows": len(aggregates),
            "paired_difference_rows": len(differences),
            "agreement_rows": len(agreements),
            "ordering_rows": len(orderings),
            "seeds": sorted({row["seed"] for row in rows}),
            "steps": sorted({row["step"] for row in rows}),
            "spaces": sorted({row["space"] for row in rows}),
            "fid_10k_range": [
                min(float(row["fid_10k"]) for row in rows),
                max(float(row["fid_10k"]) for row in rows),
            ],
            "interpretation_guard": (
                "Do not interpret low f_mem without acceptable FID and raw sample-range diagnostics."
            ),
        }
    )
    write_json(combined / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    print(combine(args.output_root, args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
