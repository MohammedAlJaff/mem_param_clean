#!/usr/bin/env python3
"""Require six successful W&B profile smokes and the exact-resume marker."""

import json
import sys
from pathlib import Path


PROFILES = ("direct_safe", "via_x_safe", "direct_exact", "via_x_exact")


def main():
    """Validate final run state plus every step-250 monitoring artifact."""
    batch_root = Path(sys.argv[1]).resolve()
    smoke_root = batch_root / "validation/smokes"
    profile_rows = [
        line.split("\t")
        for line in (batch_root / "profiles.tsv").read_text().splitlines()[1:]
        if line.strip()
    ]
    if len(profile_rows) != 6:
        raise AssertionError(f"Expected six profile rows, found {len(profile_rows)}")
    for row in profile_rows:
        profile_id, capacity, _, _, _, attention = row[:6]
        run_dir = smoke_root / f"{profile_id}_{capacity}_{attention}"
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("state") != "trained" or int(record.get("latest_step", -1)) != 250:
            raise AssertionError(f"Smoke did not finish at step 250: {run_dir}")
        wandb = record.get("wandb", {})
        if not wandb.get("enabled") or not wandb.get("success") or not wandb.get("run_id"):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        for sampler_profile in PROFILES:
            expected = (
                run_dir
                / "monitoring"
                / "sample_diagnostics"
                / sampler_profile
                / "step_000250.json"
            )
            diagnostic = json.loads(expected.read_text())
            status = diagnostic.get("status")
            if sampler_profile == "direct_safe" and status != "ok":
                raise AssertionError(f"Required sampler failed: {expected}")
            if status not in ("ok", "nonfinite"):
                raise AssertionError(f"Unknown sampler status in {expected}")
            required = [
                f"generated_samples/{sampler_profile}/step_000250.pt",
            ]
            if status == "ok":
                required.extend(
                    [
                        f"comparison_grids/{sampler_profile}/step_000250.png",
                        f"f_mem/{sampler_profile}/step_000250.json",
                    ]
                )
            for relative in required:
                if not (run_dir / "monitoring" / relative).is_file():
                    raise FileNotFoundError(f"Missing smoke artifact: {relative}")

    marker = batch_root / "validation/EXACT_RESUME_COMPLETE"
    if not marker.is_file():
        raise FileNotFoundError("The wide-attention exact-resume smoke did not pass.")
    print("Six W&B smokes and the wide-attention exact-resume check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
