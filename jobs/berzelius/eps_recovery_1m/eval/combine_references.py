#!/usr/bin/env python3
"""Combine v/v and x/x results with the corrected epsilon comparison."""

import argparse
import json
from pathlib import Path

from combine_results import (
    collect_rows,
    plot_eps_configs,
    select_eps_config_rows,
    write_csv,
)


def reference_rows(rows):
    return [
        row
        for row in rows
        if int(row["source_index"]) in {6, 7}
        and str(row["sampling_profile"]) == "saved"
    ]


def combine(output_root):
    output_root = Path(output_root).expanduser().resolve()
    collected = collect_rows(output_root)
    references = reference_rows(collected)
    expected_steps = {100_000, 390_000, 695_000}
    if len(references) != 6:
        raise RuntimeError(f"Expected six reference rows, found {len(references)}.")
    if {int(row["source_index"]) for row in references} != {6, 7}:
        raise RuntimeError("Both v/v and x/x reference rows are required.")
    if {int(row["step"]) for row in references} != expected_steps:
        raise RuntimeError("Reference checkpoint steps do not match the decision run.")
    if any(
        int(row["sample_count"]) != 10_000
        or int(row["fid_reference_count"]) != 10_000
        for row in references
    ):
        raise RuntimeError("Reference comparison requires the 10k/10k FID protocol.")

    epsilon = select_eps_config_rows(collected)
    if len(epsilon) != 18:
        raise RuntimeError(f"Expected 18 corrected epsilon rows, found {len(epsilon)}.")
    all_rows = sorted(
        epsilon + references,
        key=lambda row: (int(row["source_index"]), int(row["step"])),
    )
    combined_dir = output_root / "combined"
    write_csv(combined_dir / "reference_checkpoint_metrics.csv", references)
    plot_eps_configs(
        references,
        combined_dir / "reference_comparison.png",
        title="Three-point v/v and x/x reference comparison",
        metrics=(("fid", "FID"), ("f_mem", "Pixel f_mem")),
    )
    write_csv(
        combined_dir / "all_parameterization_checkpoint_metrics.csv", all_rows
    )
    plot_eps_configs(
        all_rows,
        combined_dir / "all_parameterization_comparison.png",
        title="Three-point parameterization comparison",
        metrics=(("fid", "FID"), ("f_mem", "Pixel f_mem")),
    )
    summary = {
        "reference_metric_rows": len(references),
        "all_metric_rows": len(all_rows),
        "reference_sources": sorted(
            {int(row["source_index"]) for row in references}
        ),
        "steps": sorted(expected_steps),
        "sample_counts": sorted({int(row["sample_count"]) for row in all_rows}),
        "fid_reference_counts": sorted(
            {int(row["fid_reference_count"]) for row in all_rows}
        ),
        "ok_rows": sum(row.get("status") == "ok" for row in all_rows),
    }
    summary_path = combined_dir / "reference_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
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
