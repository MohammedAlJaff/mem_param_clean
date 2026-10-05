#!/usr/bin/env python3
"""Require all nine N=2048 weighted-v/v trajectories to finish at one million."""

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

from mem_param_src.config import load_python_config
from mem_param_src.runners.training import resolve_checkpoint_steps


WEIGHTINGS = {"x_equivalent", "unit", "eps_equivalent"}


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


def require_equal_state(left, right, description):
    """Require two tensor state dictionaries to be bitwise equal."""
    if set(left) != set(right):
        raise AssertionError(f"Different keys in paired {description}.")
    for key in left:
        if not torch.equal(left[key], right[key]):
            raise AssertionError(f"Different paired {description} tensor: {key}")


def main():
    """Check completion, artifacts, tracking, freshness, and paired identities."""
    batch_root = Path(sys.argv[1]).resolve()
    with (batch_root / "variants.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 9:
        raise AssertionError(f"Expected nine variants, found {len(rows)}")
    if Counter(row["weighting"] for row in rows) != Counter(
        {"x_equivalent": 3, "unit": 3, "eps_equivalent": 3}
    ):
        raise AssertionError("Expected three trajectories for each raw weighting.")

    checkpoints = resolve_checkpoint_steps(load_python_config(batch_root / "run_config.py"))
    if len(checkpoints) != 72:
        raise AssertionError("Frozen config no longer resolves to 72 checkpoints.")

    paired = {}
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
        if (
            not wandb.get("enabled")
            or not wandb.get("success")
            or not wandb.get("run_id")
            or wandb.get("project") != "celeba32-unet-vv-weighting-1m"
        ):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        wandb_ids.append(wandb["run_id"])

        config = read_json(run_dir / "config.json")
        if (
            config["framework"]["prediction_space"],
            config["framework"]["loss_space"],
            config["framework"]["loss_weighting"],
        ) != ("v", "v", row["weighting"]):
            raise AssertionError(f"Unexpected native-v/v weighting: {run_dir}")
        if row["weighting"] not in WEIGHTINGS:
            raise AssertionError(f"Unknown weighting: {row['weighting']}")
        if (
            int(config["training"]["batch_size"]),
            int(config["training"]["effective_batch_size"]),
            int(config["training"]["accumulation_steps"]),
        ) != (256, 256, 1):
            raise AssertionError(f"Unexpected batch contract: {run_dir}")
        if bool(config["training"]["use_safe_div"]):
            raise AssertionError(f"Training is not exact: {run_dir}")
        sampling = config["sampling"]
        if (
            sampling["method"],
            int(sampling["n_steps"]),
            bool(sampling["use_safe_div"]),
            float(sampling["safe_div_denominator_floor"]),
            sampling["eps_to_v"],
            bool(sampling["heun_final_euler"]),
            sampling["output"],
            bool(sampling["final_pixel_clamp"]),
        ) != ("heun", 50, True, 0.05, "via_x", False, "x_t_max", False):
            raise AssertionError(f"Unexpected monitoring sampler: {run_dir}")
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
        train = manifest["partitions"]["train"]
        reference = manifest["partitions"]["reference"]
        if train["tensor"]["shape"] != [2048, 1, 32, 32]:
            raise AssertionError(f"Unexpected training tensor shape: {run_dir}")
        if reference["tensor"]["shape"] != [1024, 1, 32, 32]:
            raise AssertionError(f"Unexpected reference tensor shape: {run_dir}")
        if file_sha256(run_dir / "data_bundle.pt") != manifest["bundle"]["sha256"]:
            raise AssertionError(f"Data bundle hash mismatch: {run_dir}")

        step_zero = torch.load(
            run_dir / "checkpoints/step_000000.pt", map_location="cpu", weights_only=False
        )["model"]
        z_init = torch.load(run_dir / "z_init.pt", map_location="cpu", weights_only=False)
        seed = int(row["seed"])
        identity = (
            train["sample_ids"],
            reference["sample_ids"],
            train["tensor"]["sha256"],
            reference["tensor"]["sha256"],
        )
        if seed not in paired:
            paired[seed] = (identity, step_zero, z_init)
        else:
            base_identity, base_state, base_z = paired[seed]
            if identity != base_identity:
                raise AssertionError(f"Source identities differ within seed {seed}.")
            require_equal_state(step_zero, base_state, f"initial model for seed {seed}")
            if not torch.equal(z_init, base_z):
                raise AssertionError(f"Monitoring noise differs within seed {seed}.")

        bundle_stat = os.stat(run_dir / "data_bundle.pt")
        init_stat = os.stat(run_dir / "z_init.pt")
        bundle_inodes.append((bundle_stat.st_dev, bundle_stat.st_ino))
        init_inodes.append((init_stat.st_dev, init_stat.st_ino))

    if len(wandb_ids) != len(set(wandb_ids)):
        raise AssertionError("W&B run IDs are not unique.")
    if len(bundle_inodes) != len(set(bundle_inodes)) or len(init_inodes) != len(set(init_inodes)):
        raise AssertionError("Fresh bundle or initialization files were reused by hard link.")
    if set(paired) != {0, 1, 2}:
        raise AssertionError("Expected paired artifacts for seeds 0, 1, and 2.")

    (batch_root / "TRAINING_COMPLETE").write_text(
        "Nine fresh CelebA32 N=2048 weighted-v/v controls passed at one million.\n"
    )
    print("Validated all nine N=2048 weighted-v/v trajectories at one million updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
