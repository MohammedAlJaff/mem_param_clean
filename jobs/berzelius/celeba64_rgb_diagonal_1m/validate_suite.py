#!/usr/bin/env python3
"""Validate the fresh 18-run CelebA64 RGB diagonal production suite."""

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
    """Require the complete 2x3x3 diagonal matrix exactly once."""
    if [int(row["task_id"]) for row in rows] != list(range(18)):
        raise AssertionError("Variant IDs must be exactly 0 through 17.")
    factors = Counter()
    for row in rows:
        if row["prediction"] != row["loss"]:
            raise AssertionError(f"Only diagonal targets are allowed: {row}")
        factors[(int(row["n_train"]), int(row["seed"]), row["prediction"])] += 1
        expected_label = (
            f"n{row['n_train']}_{row['prediction']}_{row['loss']}_s{row['seed']}"
        )
        if row["label"] != expected_label:
            raise AssertionError(f"Unexpected variant label: {row}")

    expected = {
        (n_train, seed, target)
        for n_train in (512, 2048)
        for seed in (0, 1, 2)
        for target in ("x", "eps", "v")
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("Variants are not the complete 2x3x3 diagonal matrix.")


def validate_config(config):
    """Require the frozen scientific, monitoring, and runtime contract."""
    data = (
        str(config.study.dataset_key),
        str(config.data.dataset),
        str(config.data.train_split),
        str(config.data.reference_split),
        int(config.data.n_test),
        int(config.data.image_size),
        int(config.data.resize_size),
        int(config.data.image_channels),
        bool(config.data.balance_train_classes),
        bool(config.data.balance_reference_classes),
    )
    expected_data = (
        "celeba64-rgb", "celeba32_grey", "train", "valid", 1000,
        64, 64, 3, False, False,
    )
    if data != expected_data:
        raise AssertionError(f"Unexpected CelebA64 RGB data contract: {data}")

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
    expected_architecture = (
        "unet", 64, (1, 2, 4, 4), 2, 0.1, 128, (16, 8), True, False,
    )
    if architecture != expected_architecture:
        raise AssertionError(f"Unexpected wide/deep attention U-Net: {architecture}")

    training = config.training
    optimizer = (
        str(training.optimizer),
        float(training.lr),
        float(training.weight_decay),
        int(training.batch_cap),
        int(training.effective_batch_size),
        int(training.n_training_steps),
        int(training.recovery_every),
    )
    if optimizer != ("adam", 1.0e-4, 0.0, 256, 256, 1_000_000, 5_000):
        raise AssertionError(f"Unexpected optimizer, batch, or update contract: {optimizer}")
    if (float(training.start_time), float(training.end_time)) != (0.001, 0.999):
        raise AssertionError("Unexpected training time interval.")
    if bool(training.use_safe_div):
        raise AssertionError("Training conversions must use exact division.")
    if str(config.framework.loss_weighting) != "unit":
        raise AssertionError("The diagonal matrix must use unit loss weighting.")
    if str(training.ckpt_schedule) != "bonnaire-1m" or tuple(training.ckpt_steps):
        raise AssertionError("Expected the named Bonnaire schedule without overrides.")
    checkpoints = resolve_checkpoint_steps(config)
    if (
        len(checkpoints) != 72
        or checkpoints[:4] != (0, 250, 500, 750)
        or checkpoints[-4:] != (855_000, 900_000, 945_000, 1_000_000)
    ):
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
        "heun", 50, 0.001, 0.999, True, 0.05, "via_x", False, "x_t_max", False,
    )
    if observed_sampling != expected_sampling:
        raise AssertionError(f"Unexpected sampling settings: {observed_sampling}")
    if tuple(config.generation_monitor.sampling_profiles):
        raise AssertionError("Production monitoring must use only the primary sampler.")
    monitoring = (
        int(config.evaluation.sample_bank_size),
        int(config.evaluation.n_samples),
        int(config.evaluation.batch_size),
        int(config.generation_monitor.n_samples),
        int(config.generation_monitor.n_train_samples),
        int(config.random_projected_f_mem.n_samples),
        int(config.random_projected_f_mem.resize_size),
        int(config.random_projected_f_mem.projection_dim),
        int(config.random_projected_f_mem.projection_seed),
        int(config.monitoring.batch_size),
    )
    if monitoring != (128, 128, 32, 64, 32, 128, 16, 256, 3001, 32):
        raise AssertionError(f"Unexpected monitoring contract: {monitoring}")

    runtime = config.runtime
    if (
        str(runtime.precision),
        bool(runtime.tf32_enabled),
        bool(runtime.deterministic),
        str(runtime.autocast_dtype),
    ) != ("fp32", True, True, "none"):
        raise AssertionError("Expected deterministic FP32 with TF32 enabled.")
    if str(config.tracking.wandb.project) != "celeba64-rgb-unet-diagonal-1m":
        raise AssertionError("Unexpected W&B project.")
    if not bool(config.tracking.wandb.strict):
        raise AssertionError("W&B must be strict across exact resumes.")

    adam = inspect.signature(torch.optim.Adam).parameters
    if adam["betas"].default != (0.9, 0.999) or adam["eps"].default != 1.0e-8:
        raise AssertionError("Installed Adam defaults do not match the frozen recipe.")


def validate_model_and_ids(config):
    """Require the successful U-Net and prefix-nested source selections."""
    model = GDFM(config)
    if model.n_params() != 24_386_627:
        raise AssertionError(f"Unexpected U-Net parameter count: {model.n_params()}")
    for seed in (0, 1, 2):
        small = seeded_indices(162_770, 512, seed)
        large = seeded_indices(162_770, 2048, seed)
        if not torch.equal(small, large[:512]):
            raise AssertionError(f"CelebA training IDs are not nested for seed {seed}.")


def main():
    """Check matrix, config, model, optimizer defaults, and source IDs."""
    rows = read_rows(sys.argv[1])
    config = load_python_config(sys.argv[2])
    validate_rows(rows)
    validate_config(config)
    validate_model_and_ids(config)
    print("Validated 18 fresh CelebA64 RGB diagonal trajectories.")
    print("Validated physical/effective batch 256/256 and accumulation 1.")
    print("Validated attention U-Net with 24,386,627 parameters.")
    print("Validated exact training, 72 Bonnaire checkpoints, and one safe via-x sampler.")
    print("Validated paired target cells and nested N={512,2048} source IDs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
