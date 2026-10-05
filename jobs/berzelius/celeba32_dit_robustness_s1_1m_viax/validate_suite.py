#!/usr/bin/env python3
"""Validate the frozen seed-1 Small-DiT robustness matrix."""

import csv
import sys
from collections import Counter
from pathlib import Path

from mem_param_src.config import load_python_config
from mem_param_src.model import GDFM
from mem_param_src.runners.training import resolve_checkpoint_steps


def read_variants(path):
    """Read the tab-separated matrix manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    """Require the exact factors and production controls."""
    variants_path = Path(sys.argv[1]).resolve()
    config = load_python_config(sys.argv[2])
    rows = read_variants(variants_path)
    if [int(row["task_id"]) for row in rows] != list(range(18)):
        raise AssertionError("Variant IDs must be exactly 0 through 17.")

    expected_n = {512, 2048}
    spaces = {"x", "eps", "v"}
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
        (n_train, 1, prediction, loss)
        for n_train in expected_n
        for prediction in spaces
        for loss in spaces
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("Variants are not the complete 2x1x3x3 matrix.")
    if int(config.training.seed) != 1:
        raise AssertionError("The shared configuration must default to seed 1.")

    if str(config.network.arch) != "dit":
        raise AssertionError("The production backbone must be DiT.")
    architecture = (
        int(config.network.depth),
        int(config.network.hidden_size),
        int(config.network.num_heads),
        float(config.network.mlp_ratio),
        int(config.network.patch_size),
        bool(config.network.learn_sigma),
    )
    if architecture != (12, 384, 6, 4.0, 2, False):
        raise AssertionError(f"Unexpected Small-DiT architecture: {architecture}.")
    if (
        str(config.data.dataset),
        int(config.data.image_size),
        int(config.data.image_channels),
    ) != ("celeba32_grey", 32, 1):
        raise AssertionError("Expected grayscale CelebA at 32x32.")
    if int(config.training.n_training_steps) != 1_000_000:
        raise AssertionError("The scientific target must be one million updates.")
    if (
        int(config.training.batch_cap),
        int(config.training.effective_batch_size),
    ) != (64, 256):
        raise AssertionError("Expected physical/effective batch 64/256.")
    optimizer = (
        str(config.training.optimizer),
        float(config.training.lr),
        float(config.training.weight_decay),
        float(config.training.start_time),
        float(config.training.end_time),
        int(config.training.recovery_every),
        int(config.training.log_every),
    )
    if optimizer != ("adam", 1.0e-4, 0.0, 0.001, 0.999, 5_000, 250):
        raise AssertionError(f"Unexpected optimizer/training controls: {optimizer}.")
    runtime = (
        str(config.runtime.precision),
        bool(config.runtime.tf32_enabled),
        bool(config.runtime.deterministic),
        str(config.runtime.autocast_dtype),
    )
    if runtime != ("fp32", True, True, "none"):
        raise AssertionError(f"Unexpected deterministic runtime: {runtime}.")
    sampler = (
        str(config.sampling.method),
        int(config.sampling.n_steps),
        float(config.sampling.start_time),
        float(config.sampling.end_time),
        bool(config.sampling.use_safe_div),
        float(config.sampling.safe_div_denominator_floor),
        str(config.sampling.eps_to_v),
        bool(config.sampling.heun_final_euler),
        str(config.sampling.output),
        bool(config.sampling.final_pixel_clamp),
    )
    expected_sampler = (
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
    if sampler != expected_sampler:
        raise AssertionError(f"Unexpected monitoring sampler: {sampler}.")
    if int(config.data.n_test) != 1_024:
        raise AssertionError("Expected 1,024 held-out references.")
    tags = set(config.tracking.wandb.tags)
    if "seed-1" not in tags or "seed-0" in tags:
        raise AssertionError(f"Unexpected W&B seed tags: {sorted(tags)}.")
    if str(config.tracking.wandb.subexperiment) != "dit-robustness-s1-1m-viax":
        raise AssertionError("Unexpected W&B subexperiment.")
    if str(config.framework.loss_weighting) != "unit":
        raise AssertionError("Expected unit loss weighting.")
    if bool(config.training.use_safe_div):
        raise AssertionError("Training conversions must remain exact.")
    if not bool(config.sampling.use_safe_div):
        raise AssertionError("Sampling safe division must be enabled.")
    if float(config.sampling.safe_div_denominator_floor) != 0.05:
        raise AssertionError("Sampling denominator floor must be 0.05.")
    if str(config.sampling.eps_to_v) != "via_x":
        raise AssertionError("Sampling epsilon conversion must use via_x.")
    if bool(config.sampling.heun_final_euler):
        raise AssertionError("Sampling must use full Heun.")
    if bool(config.sampling.final_pixel_clamp):
        raise AssertionError("Raw completed samples must stay unclipped.")
    if not bool(config.tracking.wandb.strict):
        raise AssertionError("W&B must be strict to preserve one resume identity.")

    checkpoints = resolve_checkpoint_steps(config)
    if len(checkpoints) != 72 or checkpoints[0] != 0 or checkpoints[-1] != 1_000_000:
        raise AssertionError("Expected all 72 Bonnaire checkpoints through one million.")
    model = GDFM(config)
    if model.n_params() != 32_563_972:
        raise AssertionError(f"Unexpected parameter count: {model.n_params()}.")
    print("Validated 18 rows, N={512,2048}, nine cells, and seed 1.")
    print("Validated Small DiT (32,563,972 parameters) and 72 checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
