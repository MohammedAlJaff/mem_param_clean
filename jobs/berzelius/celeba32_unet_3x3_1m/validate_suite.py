#!/usr/bin/env python3
"""Validate the fresh 108-run CelebA32 U-Net-S production suite."""

import csv
import inspect
import sys
from collections import Counter
from pathlib import Path

import torch

from mem_param_src.config import load_python_config
from mem_param_src.datasets.common import seeded_indices
from mem_param_src.model import GDFM
from mem_param_src.runners.training import resolve_checkpoint_steps


def read_rows(path):
    """Read the tab-separated production manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_rows(rows):
    """Require the complete 4x3x3x3 matrix exactly once."""
    if [int(row["task_id"]) for row in rows] != list(range(108)):
        raise AssertionError("Variant IDs must be exactly 0 through 107.")
    factors = Counter(
        (
            int(row["n_train"]),
            int(row["seed"]),
            row["prediction"],
            row["loss"],
        )
        for row in rows
    )
    expected = {
        (n_train, seed, prediction, loss)
        for n_train in (512, 1024, 2048, 4096)
        for seed in (0, 1, 2)
        for prediction in ("x", "eps", "v")
        for loss in ("x", "eps", "v")
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("Variants are not the complete 4x3x3x3 matrix.")
    for row in rows:
        expected_label = (
            f"n{row['n_train']}_{row['prediction']}_{row['loss']}_s{row['seed']}"
        )
        if row["label"] != expected_label:
            raise AssertionError(f"Unexpected variant label: {row}")


def validate_config(config):
    """Require the frozen scientific and runtime contract."""
    if (
        str(config.data.dataset),
        str(config.data.train_split),
        str(config.data.reference_split),
        int(config.data.n_test),
        int(config.data.image_size),
        int(config.data.resize_size),
        int(config.data.image_channels),
    ) != ("celeba32_grey", "train", "valid", 1024, 32, 32, 1):
        raise AssertionError("Unexpected CelebA32 data contract.")
    architecture = (
        str(config.network.arch),
        int(config.network.base_channels),
        tuple(config.network.channel_mult),
        int(config.network.num_res_blocks),
        float(config.network.dropout),
        int(config.network.time_emb_dim),
        tuple(config.network.attn_resolutions),
        bool(config.network.middle_attention),
        bool(config.network.learn_sigma),
    )
    if architecture != ("unet", 32, (1, 2, 4), 2, 0.1, 128, (16, 8), True, False):
        raise AssertionError(f"Unexpected U-Net-S architecture: {architecture}")

    training = config.training
    if (
        str(training.optimizer),
        float(training.lr),
        float(training.weight_decay),
        int(training.batch_cap),
        int(training.effective_batch_size),
        int(training.n_training_steps),
    ) != ("adam", 1.0e-4, 0.0, 256, 256, 1_000_000):
        raise AssertionError("Unexpected optimizer, batch, or update contract.")
    if (float(training.start_time), float(training.end_time)) != (0.001, 0.999):
        raise AssertionError("Unexpected training time interval.")
    if bool(training.use_safe_div):
        raise AssertionError("Training conversions must use exact division.")
    if str(config.framework.loss_weighting) != "unit":
        raise AssertionError("The full grid must use unit loss weighting.")
    checkpoints = resolve_checkpoint_steps(config)
    if len(checkpoints) != 72 or checkpoints[0] != 0 or checkpoints[-1] != 1_000_000:
        raise AssertionError("Expected all 72 Bonnaire checkpoints through one million.")

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
        "heun", 50, 0.001, 0.999, True, 0.05, "via_x", False, "x_t_max", False
    )
    if observed_sampling != expected_sampling:
        raise AssertionError(f"Unexpected sampling settings: {observed_sampling}")

    runtime = config.runtime
    if (
        str(runtime.precision),
        bool(runtime.tf32_enabled),
        bool(runtime.deterministic),
        str(runtime.autocast_dtype),
    ) != ("fp32", False, True, "none"):
        raise AssertionError("Expected deterministic FP32 with TF32 disabled.")
    if str(config.tracking.wandb.project) != "celeba32-unet-3x3-1m":
        raise AssertionError("Unexpected W&B project.")
    if not bool(config.tracking.wandb.strict):
        raise AssertionError("W&B must be strict across exact resumes.")

    adam = inspect.signature(torch.optim.Adam).parameters
    if adam["betas"].default != (0.9, 0.999) or adam["eps"].default != 1.0e-8:
        raise AssertionError("Installed Adam defaults do not match the frozen recipe.")


def validate_model_and_ids(config):
    """Require the exact U-Net and prefix-nested source selections."""
    model = GDFM(config)
    if model.n_params() != 3_917_409:
        raise AssertionError(f"Unexpected U-Net-S parameter count: {model.n_params()}")
    for seed in (0, 1, 2):
        largest = seeded_indices(162_770, 4096, seed)
        for n_train in (512, 1024, 2048):
            selected = seeded_indices(162_770, n_train, seed)
            if not torch.equal(selected, largest[:n_train]):
                raise AssertionError(f"CelebA training IDs are not nested for seed {seed}.")


def main():
    """Check matrix, config, model, optimizer defaults, and source IDs."""
    rows = read_rows(sys.argv[1])
    config = load_python_config(sys.argv[2])
    validate_rows(rows)
    validate_config(config)
    validate_model_and_ids(config)
    print("Validated 108 fresh CelebA32 full-grid trajectories.")
    print("Validated physical/effective batch 256/256 and accumulation 1.")
    print("Validated U-Net-S with 3,917,409 parameters and 72 checkpoints.")
    print("Validated exact training and safe via-x full-Heun sampling.")
    print("Validated paired target cells and nested N={512,1024,2048,4096} IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
