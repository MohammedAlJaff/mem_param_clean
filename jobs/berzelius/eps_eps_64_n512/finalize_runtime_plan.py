#!/usr/bin/env python3
"""Combine optimizer probes and full monitor smokes into the launch-time estimate."""

import csv
import json
import sys
from pathlib import Path


CHECKPOINT_COUNT_TO_400K = 14
STAGE_ONE_CHECKPOINT_COUNT = 11
STAGE_TWO_CHECKPOINT_COUNT = 3


def read_tsv(path):
    """Read one tab-separated record set."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    """Estimate 400k walltime and select direct or two-segment execution."""
    batch_root = Path(sys.argv[1]).resolve()
    selected = read_tsv(batch_root / "validation/selected_batches.tsv")
    estimates = []
    for row in selected:
        profile_id = int(row["profile_id"])
        run_dir = (
            batch_root
            / "validation/smokes"
            / f"{profile_id}_{row['capacity']}_{row['attention']}"
        )
        smoke_wall_seconds = float((run_dir / "smoke_wall_seconds.txt").read_text())
        seconds_per_step = float(row["seconds_per_optimizer_step"])
        optimizer_seconds = seconds_per_step * 400_000
        smoke_optimizer_seconds = seconds_per_step * 250
        monitor_seconds_per_checkpoint = max(
            0.0,
            smoke_wall_seconds - smoke_optimizer_seconds,
        ) / 2.0
        estimate_seconds = 1.30 * (
            optimizer_seconds
            + CHECKPOINT_COUNT_TO_400K * monitor_seconds_per_checkpoint
        )
        stage_one_seconds = 1.30 * (
            seconds_per_step * 200_000
            + STAGE_ONE_CHECKPOINT_COUNT * monitor_seconds_per_checkpoint
        )
        stage_two_seconds = 1.30 * (
            seconds_per_step * 200_000
            + STAGE_TWO_CHECKPOINT_COUNT * monitor_seconds_per_checkpoint
        )
        estimates.append(
            {
                "profile_id": profile_id,
                "capacity": row["capacity"],
                "attention": row["attention"],
                "constraint": row["constraint"],
                "smoke_wall_seconds": smoke_wall_seconds,
                "optimizer_seconds_per_step": seconds_per_step,
                "estimated_monitor_seconds_per_checkpoint": monitor_seconds_per_checkpoint,
                "estimated_400k_hours_with_30pct_margin": estimate_seconds / 3600.0,
                "estimated_stage1_hours_with_30pct_margin": stage_one_seconds / 3600.0,
                "estimated_stage2_hours_with_30pct_margin": stage_two_seconds / 3600.0,
            }
        )
    maximum_hours = max(row["estimated_400k_hours_with_30pct_margin"] for row in estimates)
    maximum_stage_one_hours = max(
        row["estimated_stage1_hours_with_30pct_margin"] for row in estimates
    )
    maximum_stage_two_hours = max(
        row["estimated_stage2_hours_with_30pct_margin"] for row in estimates
    )
    launch_mode = "split" if maximum_hours > 40.0 else "direct"
    if launch_mode == "split" and max(
        maximum_stage_one_hours,
        maximum_stage_two_hours,
    ) > 40.0:
        raise RuntimeError(
            "A planned 200k segment exceeds the 40-hour safe limit after its "
            "actual monitor count and 30 percent margin. Review shorter segments."
        )
    record = {
        "checkpoint_count_through_400k": CHECKPOINT_COUNT_TO_400K,
        "stage_one_checkpoint_count": STAGE_ONE_CHECKPOINT_COUNT,
        "stage_two_checkpoint_count": STAGE_TWO_CHECKPOINT_COUNT,
        "monitor_margin": 0.30,
        "maximum_estimated_400k_hours": maximum_hours,
        "maximum_estimated_stage1_hours": maximum_stage_one_hours,
        "maximum_estimated_stage2_hours": maximum_stage_two_hours,
        "launch_mode": launch_mode,
        "split_rule": "0-200k then exact resume 200k-400k" if launch_mode == "split" else None,
        "profiles": estimates,
    }
    validation_dir = batch_root / "validation"
    (validation_dir / "runtime_plan.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    launch_mode_path = validation_dir / "launch_mode.txt"
    launch_mode_temporary = validation_dir / "launch_mode.txt.tmp"
    launch_mode_temporary.write_text(launch_mode + "\n")
    launch_mode_temporary.replace(launch_mode_path)
    (batch_root / "SMOKE_COMPLETE").write_text(
        "six W&B smokes, online exact resume, and runtime plan passed\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
