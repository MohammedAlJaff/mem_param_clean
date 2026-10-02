#!/usr/bin/env python3
"""Validate the fresh 36-run MNIST diagonal production suite."""

import csv
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from mem_param_src.config import clone_config, load_python_config
from mem_param_src.datasets.common import seeded_indices
from mem_param_src.model import GDFM
from mem_param_src.runners.training import resolve_checkpoint_steps


EXPECTED_CHECKPOINTS = (
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


def read_rows(path):
    """Read the tab-separated production manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_rows(rows):
    """Require the exact two-resolution diagonal matrix."""
    if [int(row["task_id"]) for row in rows] != list(range(36)):
        raise AssertionError("Variant IDs must be exactly 0 through 35.")

    expected_profiles = {
        32: ("unet-s", 32, [1, 2, 4], True, [16, 8], "thin"),
        64: ("unet-l", 64, [1, 2, 4, 4], False, [], "fat"),
    }
    factors = Counter()
    for row in rows:
        image_size = int(row["image_size"])
        observed_profile = (
            row["model_label"],
            int(row["base_channels"]),
            json.loads(row["channel_mult"]),
            json.loads(row["middle_attention"]),
            json.loads(row["attn_resolutions"]),
            row["constraint"],
        )
        if observed_profile != expected_profiles[image_size]:
            raise AssertionError(f"Unexpected architecture row: {row}")
        if row["prediction"] != row["loss"]:
            raise AssertionError(f"Only diagonal targets are allowed: {row}")
        factors[(image_size, int(row["n_train"]), int(row["seed"]), row["prediction"])] += 1

    expected = {
        (image_size, n_train, seed, target)
        for image_size in (32, 64)
        for n_train in (512, 2048)
        for seed in (0, 1, 2)
        for target in ("x", "eps", "v")
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("Variants are not the complete 2x2x3x3 diagonal matrix.")
    if any(int(row["image_size"]) != 32 for row in rows[:18]):
        raise AssertionError("Tasks 0-17 must be MNIST32.")
    if any(int(row["image_size"]) != 64 for row in rows[18:]):
        raise AssertionError("Tasks 18-35 must be MNIST64.")


def validate_config(config):
    """Require the frozen scientific and runtime contract."""
    if str(config.data.dataset) != "mnist":
        raise AssertionError("Expected the canonical MNIST loader.")
    if int(config.data.n_test) != 1000:
        raise AssertionError("Expected 1,000 saved test references.")
    if bool(config.data.balance_train_classes) or bool(config.data.balance_reference_classes):
        raise AssertionError("Class balancing must remain disabled.")
    if str(config.network.arch) != "unet":
        raise AssertionError("Both profiles must use the shared U-Net.")
    if int(config.network.num_res_blocks) != 2 or float(config.network.dropout) != 0.1:
        raise AssertionError("Unexpected U-Net residual-block or dropout setting.")
    if int(config.network.time_emb_dim) != 128:
        raise AssertionError("Expected time embedding dimension 128.")

    training = config.training
    if (
        str(training.optimizer),
        float(training.lr),
        float(training.weight_decay),
    ) != ("adam", 1.0e-4, 0.0):
        raise AssertionError("Unexpected optimizer settings.")
    if (int(training.batch_cap), int(training.effective_batch_size)) != (256, 256):
        raise AssertionError("Physical/effective batch must resolve to 256/256.")
    if int(training.n_training_steps) != 1_000_000:
        raise AssertionError("Every trajectory must target one million updates.")
    if (float(training.start_time), float(training.end_time)) != (0.001, 0.999):
        raise AssertionError("Unexpected training-time interval.")
    if bool(training.use_safe_div):
        raise AssertionError("Training conversions must use exact division.")
    if str(config.framework.loss_weighting) != "unit":
        raise AssertionError("Expected unit loss weighting.")
    if resolve_checkpoint_steps(config) != EXPECTED_CHECKPOINTS:
        raise AssertionError("Unexpected 20-point checkpoint schedule.")

    sampling = config.sampling
    observed_sampling = (
        str(sampling.method),
        int(sampling.n_steps),
        float(sampling.start_time),
        float(sampling.end_time),
        bool(sampling.use_safe_div),
        float(sampling.safe_div_denominator_floor),
        str(sampling.eps_to_v),
        bool(sampling.heun_final_euler),
        str(sampling.output),
        bool(sampling.final_pixel_clamp),
    )
    expected_sampling = (
        "heun",
        50,
        0.001,
        0.999,
        True,
        0.05,
        "via_x",
        False,
        "x_t_max",
        False,
    )
    if observed_sampling != expected_sampling:
        raise AssertionError(f"Unexpected sampling settings: {observed_sampling}")

    runtime = config.runtime
    if (
        str(runtime.precision),
        bool(runtime.tf32_enabled),
        bool(runtime.deterministic),
        str(runtime.autocast_dtype),
    ) != ("fp32", True, True, "none"):
        raise AssertionError("Unexpected CUDA arithmetic contract.")
    if str(config.tracking.wandb.project) != "mnist-unet-diagonal-1m":
        raise AssertionError("Expected the dedicated MNIST W&B project.")
    if not bool(config.tracking.wandb.strict):
        raise AssertionError("W&B identity preservation must be strict.")

    adam = inspect.signature(torch.optim.Adam).parameters
    if adam["betas"].default != (0.9, 0.999) or adam["eps"].default != 1.0e-8:
        raise AssertionError("Installed Adam defaults do not match the frozen recipe.")


def validate_models(config):
    """Construct both exact U-Net profiles and require known parameter counts."""
    small = clone_config(config)
    small.data.image_size = 32
    small.data.resize_size = 32
    small.network.base_channels = 32
    small.network.channel_mult = (1, 2, 4)
    small.network.attn_resolutions = (16, 8)
    small.network.middle_attention = True

    large = clone_config(config)
    large.data.image_size = 64
    large.data.resize_size = 64
    large.network.base_channels = 64
    large.network.channel_mult = (1, 2, 4, 4)
    large.network.attn_resolutions = ()
    large.network.middle_attention = False

    counts = (GDFM(small).n_params(), GDFM(large).n_params())
    if counts != (3_917_409, 23_773_889):
        raise AssertionError(f"Unexpected U-Net parameter counts: {counts}")


def validate_nested_ids():
    """Require N=512 to be the prefix of N=2048 for every seed."""
    for seed in (0, 1, 2):
        small = seeded_indices(60_000, 512, seed)
        large = seeded_indices(60_000, 2048, seed)
        if not torch.equal(small, large[:512]):
            raise AssertionError(f"MNIST training IDs are not nested for seed {seed}.")


def main():
    """Validate rows, config, architectures, and paired-data selection."""
    rows = read_rows(sys.argv[1])
    config = load_python_config(sys.argv[2])
    validate_rows(rows)
    validate_config(config)
    validate_models(config)
    validate_nested_ids()
    print("Validated 36 fresh MNIST diagonal trajectories.")
    print("Validated physical/effective batch 256/256 and accumulation 1.")
    print("Validated U-Net-S 3,917,409 and U-Net-L 23,773,889 parameters.")
    print("Validated 20 checkpoints and paired, nested MNIST IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
