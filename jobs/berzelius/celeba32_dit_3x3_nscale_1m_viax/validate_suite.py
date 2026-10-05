#!/usr/bin/env python3
"""Validate the frozen Small-DiT 3x3 training-size production matrix."""

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
    if [int(row["task_id"]) for row in rows] != list(range(36)):
        raise AssertionError("Variant IDs must be exactly 0 through 35.")

    expected_n = {512, 1024, 2048, 4096}
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
        (n_train, 0, prediction, loss)
        for n_train in expected_n
        for prediction in spaces
        for loss in spaces
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("Variants are not the complete 4x1x3x3 matrix.")

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
    print("Validated 36 rows, four N values, nine cells, and one seed.")
    print("Validated Small DiT (32,563,972 parameters) and 72 checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
