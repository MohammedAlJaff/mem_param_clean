#!/usr/bin/env python3
"""Require all 36 fresh MNIST trajectories to finish at one million."""

import csv
import hashlib
import json
import sys
from pathlib import Path


CHECKPOINT_STEPS = (
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
    500_000,
    600_000,
    695_000,
    800_000,
    900_000,
    1_000_000,
)


def read_json(path):
    """Read one JSON artifact."""
    return json.loads(Path(path).read_text())


def file_sha256(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    """Read the frozen production manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_recovery(run_dir):
    """Require the latest indexed recovery state to be step one million."""
    recovery_dir = run_dir / "recovery"
    index = read_json(recovery_dir / "index.json")
    latest = max(index["slots"].values(), key=lambda item: int(item["step"]))
    if int(latest["step"]) != 1_000_000:
        raise AssertionError(f"Latest recovery is not step 1M: {run_dir}")
    path = recovery_dir / latest["path"]
    if not path.is_file() or file_sha256(path) != latest["sha256"]:
        raise AssertionError(f"Invalid latest recovery artifact: {path}")


def main():
    """Check completion, artifacts, W&B, and paired source identities."""
    batch_root = Path(sys.argv[1]).resolve()
    rows = read_rows(batch_root / "variants.tsv")
    if len(rows) != 36:
        raise AssertionError(f"Expected 36 variants, found {len(rows)}")

    identities = {}
    for row in rows:
        run_dir = batch_root / "runs" / f"{row['task_id']}_{row['label']}"
        record = read_json(run_dir / "run.json")
        if record.get("state") != "trained" or int(record.get("latest_step", -1)) != 1_000_000:
            raise AssertionError(f"Run did not finish at 1M: {run_dir}")
        if record.get("error") is not None:
            raise AssertionError(f"Run records an error: {run_dir}")
        wandb = record.get("wandb", {})
        if not wandb.get("enabled") or not wandb.get("success") or not wandb.get("run_id"):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        config = read_json(run_dir / "config.json")
        if int(config["training"]["batch_size"]) != 256:
            raise AssertionError(f"Physical batch is not 256: {run_dir}")
        if int(config["training"]["effective_batch_size"]) != 256:
            raise AssertionError(f"Effective batch is not 256: {run_dir}")
        if int(config["training"]["accumulation_steps"]) != 1:
            raise AssertionError(f"Accumulation is not one: {run_dir}")
        if config["sampling"]["eps_to_v"] != "via_x":
            raise AssertionError(f"Sampling route is not via_x: {run_dir}")

        for step in CHECKPOINT_STEPS:
            required = (
                run_dir / "checkpoints" / f"step_{step:06d}.pt",
                run_dir / "monitoring/generated_samples" / f"step_{step:06d}.pt",
                run_dir / "monitoring/comparison_grids" / f"step_{step:06d}.png",
                run_dir / "monitoring" / f"f_mem_step_{step:06d}.json",
            )
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing production artifact: {path}")
        require_recovery(run_dir)

        manifest = read_json(run_dir / "data_manifest.json")
        sample_ids = manifest["partitions"]["train"]["sample_ids"]
        key = (int(row["image_size"]), int(row["n_train"]), int(row["seed"]))
        identities.setdefault(key, []).append(sample_ids)

    canonical = {}
    for key, values in identities.items():
        if len(values) != 3 or any(value != values[0] for value in values[1:]):
            raise AssertionError(f"Target cells do not share source IDs: {key}")
        canonical[key] = values[0]
    for seed in (0, 1, 2):
        for n_train in (512, 2048):
            if canonical[(32, n_train, seed)] != canonical[(64, n_train, seed)]:
                raise AssertionError(
                    f"MNIST32 and MNIST64 source IDs differ for N={n_train}, seed={seed}."
                )
        if canonical[(32, 512, seed)] != canonical[(32, 2048, seed)][:512]:
            raise AssertionError(f"Training IDs are not nested for seed {seed}.")

    (batch_root / "TRAINING_COMPLETE").write_text(
        "36 fresh MNIST trajectories passed the one-million-step contract\n"
    )
    print("Validated all 36 fresh MNIST trajectories at one million updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
