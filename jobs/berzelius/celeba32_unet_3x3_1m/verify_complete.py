#!/usr/bin/env python3
"""Require all 108 fresh CelebA32 trajectories to finish at one million."""

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from mem_param_src.config import load_python_config
from mem_param_src.runners.training import resolve_checkpoint_steps


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


def require_recovery(run_dir):
    """Require a valid latest recovery state at step one million."""
    index = read_json(run_dir / "recovery/index.json")
    latest = max(index["slots"].values(), key=lambda item: int(item["step"]))
    if int(latest["step"]) != 1_000_000:
        raise AssertionError(f"Latest recovery is not step 1M: {run_dir}")
    path = run_dir / "recovery" / latest["path"]
    if not path.is_file() or file_sha256(path) != latest["sha256"]:
        raise AssertionError(f"Invalid latest recovery artifact: {path}")


def main():
    """Check completion, artifacts, tracking, freshness, and paired source IDs."""
    batch_root = Path(sys.argv[1]).resolve()
    with (batch_root / "variants.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 108:
        raise AssertionError(f"Expected 108 variants, found {len(rows)}")
    checkpoints = resolve_checkpoint_steps(load_python_config(batch_root / "run_config.py"))
    if len(checkpoints) != 72:
        raise AssertionError("Frozen config no longer resolves to 72 checkpoints.")

    identities = defaultdict(list)
    reference_ids = defaultdict(list)
    wandb_ids = []
    bundle_inodes = []
    init_inodes = []
    for row in rows:
        task_id = int(row["task_id"])
        run_dir = batch_root / "runs" / f"{task_id}_{row['label']}"
        record = read_json(run_dir / "run.json")
        if record.get("state") != "trained" or int(record.get("latest_step", -1)) != 1_000_000:
            raise AssertionError(f"Run did not finish at 1M: {run_dir}")
        if record.get("error") is not None or int(record.get("parameter_count", -1)) != 3_917_409:
            raise AssertionError(f"Run records an error or wrong model size: {run_dir}")
        wandb = record.get("wandb", {})
        if not wandb.get("enabled") or not wandb.get("success") or not wandb.get("run_id"):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        wandb_ids.append(wandb["run_id"])

        config = read_json(run_dir / "config.json")
        if (
            int(config["training"]["batch_size"]),
            int(config["training"]["effective_batch_size"]),
            int(config["training"]["accumulation_steps"]),
        ) != (256, 256, 1):
            raise AssertionError(f"Unexpected batch contract: {run_dir}")
        if bool(config["training"]["use_safe_div"]):
            raise AssertionError(f"Training is not exact: {run_dir}")
        if not bool(config["sampling"]["use_safe_div"]):
            raise AssertionError(f"Sampling is not safe: {run_dir}")
        if config["sampling"]["eps_to_v"] != "via_x":
            raise AssertionError(f"Sampling route is not via_x: {run_dir}")
        if bool(config["runtime"]["tf32_enabled"]):
            raise AssertionError(f"TF32 is enabled: {run_dir}")

        for step in checkpoints:
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
        key = (int(row["n_train"]), int(row["seed"]))
        identities[key].append(manifest["partitions"]["train"]["sample_ids"])
        reference_ids[key].append(manifest["partitions"]["reference"]["sample_ids"])
        if file_sha256(run_dir / "data_bundle.pt") != manifest["bundle"]["sha256"]:
            raise AssertionError(f"Data bundle hash mismatch: {run_dir}")
        bundle_stat = os.stat(run_dir / "data_bundle.pt")
        init_stat = os.stat(run_dir / "z_init.pt")
        bundle_inodes.append((bundle_stat.st_dev, bundle_stat.st_ino))
        init_inodes.append((init_stat.st_dev, init_stat.st_ino))

    if len(wandb_ids) != len(set(wandb_ids)):
        raise AssertionError("W&B run IDs are not unique.")
    if len(bundle_inodes) != len(set(bundle_inodes)) or len(init_inodes) != len(set(init_inodes)):
        raise AssertionError("Fresh bundle or initialization files were reused by hard link.")
    canonical = {}
    canonical_reference = {}
    for key, values in identities.items():
        if len(values) != 9 or any(value != values[0] for value in values[1:]):
            raise AssertionError(f"Target cells do not share training IDs: {key}")
        canonical[key] = values[0]
    for key, values in reference_ids.items():
        if len(values) != 9 or any(value != values[0] for value in values[1:]):
            raise AssertionError(f"Target cells do not share reference IDs: {key}")
        canonical_reference[key] = values[0]
    for seed in (0, 1, 2):
        largest = canonical[(4096, seed)]
        for n_train in (512, 1024, 2048):
            if canonical[(n_train, seed)] != largest[:n_train]:
                raise AssertionError(f"Training IDs are not nested for seed {seed}, N={n_train}.")
        references = [canonical_reference[(n_train, seed)] for n_train in (512,1024,2048,4096)]
        if any(value != references[0] for value in references[1:]):
            raise AssertionError(f"Reference IDs differ across N for seed {seed}.")

    (batch_root / "TRAINING_COMPLETE").write_text(
        "108 fresh CelebA32 U-Net-S trajectories passed the one-million-step contract\n"
    )
    print("Validated all 108 fresh CelebA32 trajectories at one million updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
