#!/usr/bin/env python3
"""Require exact state and monitoring equality for the wide-attention resume smoke."""

import hashlib
import sys
from pathlib import Path

import torch

from mem_param_src.runners.artifacts import load_torch, read_json
from mem_param_src.runners.recovery import load_latest_recovery


def require_equal(first, second, label):
    """Compare nested optimizer, RNG, sampler, and tensor objects exactly."""
    if torch.is_tensor(first):
        if not torch.equal(first, second):
            raise AssertionError(f"Tensor mismatch: {label}")
        return
    if isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"Key mismatch: {label}")
        for key in first:
            require_equal(first[key], second[key], f"{label}.{key}")
        return
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError(f"Length mismatch: {label}")
        for index, (left, right) in enumerate(zip(first, second)):
            require_equal(left, right, f"{label}[{index}]")
        return
    if hasattr(first, "shape") and hasattr(first, "dtype"):
        if first.shape != second.shape or first.dtype != second.dtype:
            raise AssertionError(f"Array metadata mismatch: {label}")
        if not (first == second).all():
            raise AssertionError(f"Array mismatch: {label}")
        return
    if first != second:
        raise AssertionError(f"Value mismatch: {label}: {first!r} != {second!r}")


def file_sha256(path):
    """Hash one monitoring artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monitoring_hashes(run_dir):
    """Return relative hashes for every deterministic monitoring artifact."""
    root = Path(run_dir) / "monitoring"
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main():
    """Compare final state, attempts, and four-profile monitoring artifacts."""
    root = Path(sys.argv[1]).resolve()
    marker = Path(sys.argv[2]).resolve()
    baseline_dir = root / "baseline"
    resumed_dir = root / "resumed"
    baseline_model = load_torch(baseline_dir / "checkpoints/step_000004.pt")
    resumed_model = load_torch(resumed_dir / "checkpoints/step_000004.pt")
    require_equal(baseline_model, resumed_model, "checkpoint")

    _, _, baseline = load_latest_recovery(baseline_dir)
    _, _, resumed = load_latest_recovery(resumed_dir)
    for key in ("step", "model", "optimizer", "rng", "batch_sampler", "history"):
        require_equal(baseline[key], resumed[key], key)

    baseline_record = read_json(baseline_dir / "run.json")
    paused_record = read_json(root / "resumed_paused_run.json")
    resumed_record = read_json(resumed_dir / "run.json")
    if baseline_record["state"] != "trained" or baseline_record["attempt_count"] != 1:
        raise AssertionError("Baseline run did not complete in one attempt.")
    if resumed_record["state"] != "trained" or resumed_record["attempt_count"] != 2:
        raise AssertionError("Resumed run did not complete in two attempts.")
    for label, record in (
        ("baseline", baseline_record),
        ("paused", paused_record),
        ("resumed", resumed_record),
    ):
        wandb = record.get("wandb", {})
        if not wandb.get("enabled") or not wandb.get("success") or not wandb.get("run_id"):
            raise AssertionError(f"{label} W&B run did not finish successfully.")
    if paused_record["wandb"]["run_id"] != resumed_record["wandb"]["run_id"]:
        raise AssertionError("The resumed process did not reuse its W&B run ID.")

    baseline_monitoring = monitoring_hashes(baseline_dir)
    resumed_monitoring = monitoring_hashes(resumed_dir)
    if baseline_monitoring != resumed_monitoring:
        raise AssertionError("Monitoring artifacts differ after exact resume.")
    profiles = ("direct_safe", "via_x_safe", "direct_exact", "via_x_exact")
    for profile in profiles:
        expected = f"sample_diagnostics/{profile}/step_000004.json"
        if expected not in baseline_monitoring:
            raise AssertionError(f"Missing final monitoring profile: {profile}")

    marker.write_text("wide_deep attention-on exact resume passed\n")
    print("Wide-attention CUDA state and monitoring match exact resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
