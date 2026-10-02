#!/usr/bin/env python3
"""Statically validate the 24-row N=512 study and six-profile routing."""

import csv
import sys
from collections import Counter
from pathlib import Path

from mem_param_src.config import load_python_config
from mem_param_src.runners.image_monitoring import resolve_monitor_sampling_profiles


def read_tsv(path):
    """Read a tab-separated study manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    """Check factors, GPU classes, attention removal, and monitor profiles."""
    suite = Path(sys.argv[1]).resolve()
    config = load_python_config(sys.argv[2])
    rows = read_tsv(suite / "variants.tsv")
    profiles = read_tsv(suite / "profiles.tsv")
    if [int(row["id"]) for row in rows] != list(range(24)):
        raise AssertionError("Variant IDs must be exactly 0 through 23.")
    if [int(row["profile_id"]) for row in profiles] != list(range(6)):
        raise AssertionError("Profile IDs must be exactly 0 through 5.")
    expected_formats = {"celeba64_gray", "celeba64_rgb", "mnist64_gray", "flowers64_rgb"}
    expected_capacities = {"small", "deep", "wide_deep"}
    expected_attention = {"off", "on"}
    factors = Counter((row["format"], row["capacity"], row["attention"]) for row in rows)
    expected = {
        (data_format, capacity, attention)
        for data_format in expected_formats
        for capacity in expected_capacities
        for attention in expected_attention
    }
    if set(factors) != expected or set(factors.values()) != {1}:
        raise AssertionError("The matrix is not the complete 4x3x2 factorization.")
    if sum(row["constraint"] == "thin" for row in rows) != 16:
        raise AssertionError("Expected 16 thin rows.")
    if sum(row["constraint"] == "fat" for row in rows) != 8:
        raise AssertionError("Expected 8 fat rows.")
    profiles_by_id = {row["profile_id"]: row for row in profiles}
    for row in rows:
        profile = profiles_by_id[row["profile_id"]]
        for field in (
            "capacity", "base_channels", "channel_mult", "time_emb_dim",
            "attention", "middle_attention", "attn_resolutions", "constraint",
        ):
            if row[field] != profile[field]:
                raise AssertionError(
                    f"Variant {row['id']} disagrees with profile "
                    f"{row['profile_id']} for {field}."
                )
        if row["capacity"] == "wide_deep" and row["constraint"] != "fat":
            raise AssertionError("Every wide_deep row must use a fat GPU.")
        if row["capacity"] != "wide_deep" and row["constraint"] != "thin":
            raise AssertionError("Small and deep rows must use thin GPUs.")
        if row["attention"] == "off":
            if row["middle_attention"] != "false" or row["attn_resolutions"] != "[]":
                raise AssertionError("Attention-off must remove middle and resolution attention.")
        else:
            if row["middle_attention"] != "true" or row["attn_resolutions"] != "[16,8]":
                raise AssertionError("Attention-on must enable middle plus 16/8 attention.")
    if int(config.data.n_train) != 512 or int(config.training.n_training_steps) != 1_000_000:
        raise AssertionError("Config must retain N=512 and a one-million-step target.")
    names = [profile["name"] for profile in resolve_monitor_sampling_profiles(config)]
    if names != ["direct_safe", "via_x_safe", "direct_exact", "via_x_exact"]:
        raise AssertionError("Monitor sampling profiles do not match the study contract.")
    required_steps = {
        250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000,
        100_000, 200_000, 300_000, 390_000, 400_000, 500_000,
        600_000, 695_000, 800_000, 900_000, 1_000_000,
    }
    if not required_steps <= set(config.training.ckpt_steps):
        raise AssertionError("Checkpoint schedule does not match the monitoring plan.")
    print("Validated 24 rows, six profiles, 16 thin routes, and 8 fat routes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
