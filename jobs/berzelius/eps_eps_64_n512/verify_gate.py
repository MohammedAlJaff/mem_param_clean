#!/usr/bin/env python3
"""Require all 24 main runs to satisfy the 400k acceptance contract."""

import csv
import json
import sys
from pathlib import Path

from mem_param_src.runners.recovery import load_latest_recovery


PROFILES = ("direct_safe", "via_x_safe", "direct_exact", "via_x_exact")
MONITOR_STEPS = (
    0,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    200_000,
    300_000,
    390_000,
    400_000,
)


def read_rows(path):
    """Read the frozen 24-row study matrix."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    """Check state, checkpoint, recovery, W&B, and four sampler profiles."""
    batch_root = Path(sys.argv[1]).resolve()
    rows = read_rows(batch_root / "variants.tsv")
    if len(rows) != 24:
        raise AssertionError(f"Expected 24 variants, found {len(rows)}")
    for row in rows:
        run_dir = batch_root / "main/runs" / f"{row['id']}_{row['label']}_s0"
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("state") != "paused" or int(record.get("latest_step", -1)) != 400_000:
            raise AssertionError(f"Run did not pause at 400k: {run_dir}")
        for step in MONITOR_STEPS:
            checkpoint = run_dir / "checkpoints" / f"step_{step:06d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        recovery_step, recovery_path, recovery = load_latest_recovery(run_dir)
        if recovery_step != 400_000 or int(recovery.get("step", -1)) != 400_000:
            raise AssertionError(f"Latest recovery is not step 400k: {recovery_path}")
        wandb = record.get("wandb", {})
        if not wandb.get("enabled") or not wandb.get("success") or not wandb.get("run_id"):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        for step in MONITOR_STEPS:
            for profile in PROFILES:
                diagnostic_path = (
                    run_dir
                    / "monitoring/sample_diagnostics"
                    / profile
                    / f"step_{step:06d}.json"
                )
                diagnostic = json.loads(diagnostic_path.read_text())
                status = diagnostic.get("status")
                if profile == "direct_safe" and status != "ok":
                    raise AssertionError(f"Required sampler failed: {diagnostic_path}")
                if status not in ("ok", "nonfinite"):
                    raise AssertionError(f"Unknown sampler status: {diagnostic_path}")
                required = [f"generated_samples/{profile}/step_{step:06d}.pt"]
                if status == "ok":
                    required.extend(
                        [
                            f"comparison_grids/{profile}/step_{step:06d}.png",
                            f"f_mem/{profile}/step_{step:06d}.json",
                        ]
                    )
                for relative in required:
                    path = run_dir / "monitoring" / relative
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing monitor artifact: {path}")
    (batch_root / "GATE_400K_COMPLETE").write_text(
        "24 runs passed all 14 checkpoints and four sampling-profile records\n"
    )
    print("Validated all 24 runs at the 400k gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
