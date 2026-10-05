#!/usr/bin/env python3
"""Require exact final state equality for the CUDA resume smoke."""

import sys
from pathlib import Path

import torch

from mem_param_src.runners.artifacts import load_torch, read_json
from mem_param_src.runners.recovery import load_latest_recovery


def require_equal(first, second, label):
    """Compare nested optimizer and RNG objects with exact tensor equality."""
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


def main():
    """Compare the final model, optimizer, RNG, sampler, history, and attempts."""
    root = Path(sys.argv[1]).resolve()
    baseline_dir = root / "baseline"
    resumed_dir = root / "resumed"
    baseline_model = load_torch(baseline_dir / "checkpoints/step_000040.pt")
    resumed_model = load_torch(resumed_dir / "checkpoints/step_000040.pt")
    require_equal(baseline_model, resumed_model, "checkpoint")

    _, _, baseline = load_latest_recovery(baseline_dir)
    _, _, resumed = load_latest_recovery(resumed_dir)
    for key in ("step", "model", "optimizer", "rng", "batch_sampler", "history"):
        require_equal(baseline[key], resumed[key], key)

    record = read_json(resumed_dir / "run.json")
    if record["state"] != "trained" or record["attempt_count"] != 2:
        raise AssertionError("Resumed run did not record two completed attempts.")
    print("CUDA resume state matches uninterrupted training exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
