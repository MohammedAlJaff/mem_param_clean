#!/usr/bin/env python3
"""Require all 18 fresh CelebA64 RGB trajectories to finish at one million."""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

from mem_param_src.config import load_python_config
from mem_param_src.runners.artifacts import file_sha256, load_torch
from mem_param_src.runners.training import resolve_checkpoint_steps


def read_json(path):
    """Read one JSON artifact."""
    return json.loads(Path(path).read_text())


def require_recovery(run_dir):
    """Require a valid latest recovery state at step one million."""
    index = read_json(run_dir / "recovery/index.json")
    latest = max(index["slots"].values(), key=lambda item: int(item["step"]))
    if int(latest["step"]) != 1_000_000:
        raise AssertionError(f"Latest recovery is not step 1M: {run_dir}")
    path = run_dir / "recovery" / latest["path"]
    if not path.is_file() or file_sha256(path) != latest["sha256"]:
        raise AssertionError(f"Invalid latest recovery artifact: {path}")


def require_attempt_history(record, run_dir):
    """Require contiguous exact-resume attempts from zero through one million."""
    attempts = record.get("attempts", [])
    if len(attempts) < 5:
        raise AssertionError(f"Expected at least five staged attempts: {run_dir}")
    expected_start = 0
    for index, attempt in enumerate(attempts):
        if int(attempt.get("index", -1)) != index:
            raise AssertionError(f"Non-contiguous attempt indices: {run_dir}")
        start = int(attempt.get("start_step", -1))
        end = int(attempt.get("end_step", -1))
        if start != expected_start or end <= start:
            raise AssertionError(f"Invalid exact-resume boundary in attempt {index}: {run_dir}")
        expected_start = end
    if expected_start != 1_000_000:
        raise AssertionError(f"Attempt history does not end at one million: {run_dir}")


def main():
    """Check completion, artifacts, tracking, freshness, and paired identities."""
    batch_root = Path(sys.argv[1]).resolve()
    with (batch_root / "variants.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 18:
        raise AssertionError(f"Expected 18 variants, found {len(rows)}")

    frozen_config = load_python_config(batch_root / "run_config.py")
    checkpoints = resolve_checkpoint_steps(frozen_config)
    if len(checkpoints) != 72 or checkpoints[0] != 0 or checkpoints[-1] != 1_000_000:
        raise AssertionError("Frozen config no longer resolves to 72 Bonnaire checkpoints.")

    train_ids = defaultdict(list)
    reference_ids = defaultdict(list)
    noise_tensors = defaultdict(list)
    wandb_ids = []
    bundle_inodes = []
    noise_inodes = []
    for row in rows:
        task_id = int(row["task_id"])
        n_train = int(row["n_train"])
        seed = int(row["seed"])
        run_dir = batch_root / "runs" / f"{task_id}_{row['label']}"
        record = read_json(run_dir / "run.json")
        if record.get("state") != "trained" or int(record.get("latest_step", -1)) != 1_000_000:
            raise AssertionError(f"Run did not finish at 1M: {run_dir}")
        if record.get("error") is not None or int(record.get("parameter_count", -1)) != 24_386_627:
            raise AssertionError(f"Run records an error or wrong model size: {run_dir}")
        require_attempt_history(record, run_dir)

        wandb = record.get("wandb", {})
        if (
            not wandb.get("enabled")
            or not wandb.get("success")
            or not wandb.get("run_id")
            or wandb.get("project") != "celeba64-rgb-unet-diagonal-1m"
        ):
            raise AssertionError(f"W&B did not finish successfully: {run_dir}")
        wandb_ids.append(wandb["run_id"])

        config = read_json(run_dir / "config.json")
        if (
            int(config["data"]["n_train"]),
            int(config["data"]["n_test"]),
            int(config["data"]["image_size"]),
            int(config["data"]["image_channels"]),
        ) != (n_train, 1000, 64, 3):
            raise AssertionError(f"Unexpected data contract: {run_dir}")
        if (
            int(config["network"]["base_channels"]),
            tuple(config["network"]["channel_mult"]),
            tuple(config["network"]["attn_resolutions"]),
            bool(config["network"]["middle_attention"]),
        ) != (64, (1, 2, 4, 4), (16, 8), True):
            raise AssertionError(f"Unexpected model profile: {run_dir}")
        if (
            int(config["training"]["batch_size"]),
            int(config["training"]["effective_batch_size"]),
            int(config["training"]["accumulation_steps"]),
        ) != (256, 256, 1):
            raise AssertionError(f"Unexpected batch contract: {run_dir}")
        if (
            config["framework"]["prediction_space"],
            config["framework"]["loss_space"],
            config["framework"]["loss_weighting"],
        ) != (row["prediction"], row["loss"], "unit"):
            raise AssertionError(f"Unexpected target parameterization: {run_dir}")
        if bool(config["training"]["use_safe_div"]):
            raise AssertionError(f"Training is not exact: {run_dir}")
        if tuple(config["training"]["resolved_ckpt_steps"]) != tuple(checkpoints):
            raise AssertionError(f"Resolved checkpoints changed: {run_dir}")
        sampling = config["sampling"]
        observed_sampling = (
            sampling["method"],
            int(sampling["n_steps"]),
            float(sampling["start_time"]),
            float(sampling["end_time"]),
            bool(sampling["use_safe_div"]),
            float(sampling["safe_div_denominator_floor"]),
            sampling["eps_to_v"],
            bool(sampling["heun_final_euler"]),
            sampling["output"],
            bool(sampling["final_pixel_clamp"]),
        )
        expected_sampling = (
            "heun", 50, 0.001, 0.999, True, 0.05,
            "via_x", False, "x_t_max", False,
        )
        if observed_sampling != expected_sampling:
            raise AssertionError(f"Unexpected sampler contract: {run_dir}")
        if not bool(config["runtime"]["tf32_enabled"]):
            raise AssertionError(f"TF32 is disabled: {run_dir}")

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
        preprocessing = manifest["preprocessing"]
        if (
            preprocessing["image_channels"],
            preprocessing["image_size"],
            preprocessing["resize_size"],
            preprocessing["training_augmentation"],
            preprocessing["value_range"],
        ) != (3, 64, 64, False, [-1.0, 1.0]):
            raise AssertionError(f"Unexpected saved preprocessing: {run_dir}")
        train_shape = manifest["partitions"]["train"]["tensor"]["shape"]
        if train_shape != [n_train, 3, 64, 64]:
            raise AssertionError(f"Unexpected saved training tensor shape: {run_dir}")
        reference_shape = manifest["partitions"]["reference"]["tensor"]["shape"]
        if reference_shape != [1000, 3, 64, 64]:
            raise AssertionError(f"Unexpected saved reference tensor shape: {run_dir}")
        key = (n_train, seed)
        train_ids[key].append(manifest["partitions"]["train"]["sample_ids"])
        reference_ids[key].append(manifest["partitions"]["reference"]["sample_ids"])
        if file_sha256(run_dir / "data_bundle.pt") != manifest["bundle"]["sha256"]:
            raise AssertionError(f"Data bundle hash mismatch: {run_dir}")

        bundle_stat = os.stat(run_dir / "data_bundle.pt")
        noise_stat = os.stat(run_dir / "z_init.pt")
        bundle_inodes.append((bundle_stat.st_dev, bundle_stat.st_ino))
        noise_inodes.append((noise_stat.st_dev, noise_stat.st_ino))
        noise_tensors[seed].append(load_torch(run_dir / "z_init.pt"))

    if len(wandb_ids) != len(set(wandb_ids)):
        raise AssertionError("W&B run IDs are not unique.")
    if len(bundle_inodes) != len(set(bundle_inodes)):
        raise AssertionError("Fresh data-bundle files were reused by hard link.")
    if len(noise_inodes) != len(set(noise_inodes)):
        raise AssertionError("Fresh fixed-noise files were reused by hard link.")

    canonical_train = {}
    canonical_reference = {}
    for key, values in train_ids.items():
        if len(values) != 3 or any(value != values[0] for value in values[1:]):
            raise AssertionError(f"Diagonal cells do not share training IDs: {key}")
        canonical_train[key] = values[0]
    for key, values in reference_ids.items():
        if len(values) != 3 or any(value != values[0] for value in values[1:]):
            raise AssertionError(f"Diagonal cells do not share reference IDs: {key}")
        canonical_reference[key] = values[0]
    for seed in (0, 1, 2):
        if canonical_train[(512, seed)] != canonical_train[(2048, seed)][:512]:
            raise AssertionError(f"Training IDs are not nested for seed {seed}.")
        if canonical_reference[(512, seed)] != canonical_reference[(2048, seed)]:
            raise AssertionError(f"Reference IDs differ across N for seed {seed}.")
        tensors = noise_tensors[seed]
        if len(tensors) != 6 or any(not torch.equal(value, tensors[0]) for value in tensors[1:]):
            raise AssertionError(f"Fixed monitoring noise is not paired for seed {seed}.")

    (batch_root / "TRAINING_COMPLETE").write_text(
        "18 fresh CelebA64 RGB trajectories passed the one-million-step contract\n"
    )
    print("Validated all 18 fresh CelebA64 RGB trajectories at one million updates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
