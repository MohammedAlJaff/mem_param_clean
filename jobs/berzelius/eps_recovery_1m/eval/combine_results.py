#!/usr/bin/env python3
"""Combine the three-point eps/eps configuration comparison."""

import argparse
import csv
import json
import math
from pathlib import Path


EPS_PROFILE_BY_SOURCE = {
    0: "old_full_heun",
    1: "direct_full_heun",
    2: "direct_final_euler",
    3: "exact_full_heun",
    4: "exact_final_euler",
    5: "direct_final_euler",
}


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def metric_key(row):
    return (
        int(row["source_index"]),
        str(row["sampling_profile"]),
        int(row["step"]),
        int(row["sample_count"]),
        int(row["fid_reference_count"]),
        str(row["sampling_signature"]),
    )


def collect_rows(output_root):
    """Load one durable copy of every completed configuration/checkpoint row."""
    rows_by_key = {}
    for path in sorted(Path(output_root).glob("tasks/*/checkpoint_metrics.jsonl")):
        for row in read_jsonl(path):
            rows_by_key[metric_key(row)] = row
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (int(row["source_index"]), int(row["step"])))
    return rows


def select_eps_config_rows(rows):
    """Keep each eps/eps run with the sampler configuration it was launched with."""
    return [
        row
        for row in rows
        if int(row["source_index"]) in EPS_PROFILE_BY_SOURCE
        and str(row["sampling_profile"])
        == EPS_PROFILE_BY_SOURCE[int(row["source_index"])]
    ]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_value(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _configure_axis(axis, ylabel):
    axis.set_xscale("linear")
    axis.set_xlabel("Gradient-update step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)


def plot_eps_configs(
    rows,
    path,
    title="Three-point eps/eps configuration comparison",
    metrics=None,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = (
            ("fid", "FID"),
            ("f_mem", "Pixel f_mem"),
            ("test_random_t_loss", "Held-out loss"),
        )
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(9, 3.6 * len(metrics)), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for source_index in sorted({int(row["source_index"]) for row in rows}):
        selected = [
            row for row in rows if int(row["source_index"]) == source_index
        ]
        selected.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in selected]
        label = selected[0]["source_label"]
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = [finite_value(row, metric) for row in selected]
            values = [math.nan if value is None else value for value in values]
            axis.plot(steps, values, marker="o", label=label)
            _configure_axis(axis, ylabel)
    axes[0].set_title(title)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def combine(output_root):
    output_root = Path(output_root).expanduser().resolve()
    rows = select_eps_config_rows(collect_rows(output_root))
    if not rows:
        raise FileNotFoundError(f"No eps/eps metric rows exist under {output_root}.")
    invalid_fid_10k = [
        row
        for row in rows
        if int(row.get("sample_count", 0)) == 10_000
        and int(row.get("fid_reference_count", 0)) < 10_000
    ]
    if invalid_fid_10k:
        raise RuntimeError(
            "Refusing to combine FID-10k rows with fewer than 10k references."
        )
    combined_dir = output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_csv(combined_dir / "eps_config_checkpoint_metrics.csv", rows)
    plot_eps_configs(rows, combined_dir / "eps_config_comparison.png")
    summary = {
        "metric_rows": len(rows),
        "source_runs": sorted({int(row["source_index"]) for row in rows}),
        "profiles": sorted({str(row["sampling_profile"]) for row in rows}),
        "steps": sorted({int(row["step"]) for row in rows}),
        "sample_counts": sorted({int(row["sample_count"]) for row in rows}),
        "fid_reference_counts": sorted(
            {int(row["fid_reference_count"]) for row in rows}
        ),
        "ok_rows": sum(row.get("status") == "ok" for row in rows),
        "nonfinite_rows": sum(row.get("status") == "nonfinite" for row in rows),
    }
    with (combined_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return combined_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(combine(args.output_root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
