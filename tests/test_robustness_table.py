"""Protect paired checkpoint and threshold choices in the M4 robustness table."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
import figures
from plot_robustness_table import RAW_METRIC, select_paired_metrics


def paired_fixture():
    quality, evaluation = [], []
    for seed in (0, 1):
        for step in (0, 100):
            quality.append({
                "experiment": "fixture", "run_id": f"s{seed}", "seed": seed,
                "n_train": 2048, "prediction_space": "x", "loss_space": "x", "step": step,
                "metric": "inception_fid", "value": 10 if seed == 0 and step == 0 else 5,
            })
            for metric, space, alpha, rho, value in (
                (RAW_METRIC, "raw_pixels", "", "0.3", 0.1 if step == 0 else 0.2),
                (RAW_METRIC, "raw_pixels", "0.01", "0.9", 0.99),
                ("f_mem_sscd", "", "0.01", "0.92", 0.6 if step == 0 else 0.8),
                ("f_mem_sscd", "", "0.02", "0.94", 0.99),
            ):
                evaluation.append({
                    "run_id": f"s{seed}", "step": str(step), "metric": metric, "space": space,
                    "alpha": alpha, "rho": rho, "value": str(value), "sample_count": "10000",
                })
    return quality, evaluation


def test_both_readouts_follow_per_seed_best_quality_and_thresholds():
    quality, evaluation = paired_fixture()
    selected = select_paired_metrics(quality, evaluation, "f_mem_sscd")
    # Seed 0 improves at step 100; seed 1 ties and must use the earlier checkpoint.
    assert [(r["step"], r[RAW_METRIC], r["f_mem_sscd"]) for r in selected] == [
        (100, 0.2, 0.8), (0, 0.1, 0.6),
    ]


def test_missing_or_duplicate_feature_value_cannot_change_selection():
    quality, evaluation = paired_fixture()
    missing = [r for r in evaluation if not (
        r["run_id"] == "s0" and r["step"] == "100"
        and r["metric"] == "f_mem_sscd" and r["alpha"] == "0.01")]
    with pytest.raises(ValueError, match="Missing paired"):
        select_paired_metrics(quality, missing, "f_mem_sscd")
    duplicate = next(r for r in evaluation if r["run_id"] == "s1"
                     and r["step"] == "0" and r["metric"] == "f_mem_sscd" and r["alpha"] == "0.01")
    with pytest.raises(ValueError, match="Duplicate memorization"):
        select_paired_metrics(quality, [*evaluation, duplicate], "f_mem_sscd")


@pytest.mark.parametrize("changes", [
    {"rho": 0.5}, {"alpha": 0.02}, {"n_trains": [512]}, {"uncertainty": "sd"},
    {"selection": "per_seed_best_fid"}, {"seeds": [0, 1, 2]},
])
def test_m4_catalog_rejects_inaccurate_scientific_labels(changes):
    spec = copy.deepcopy(figures.FIGURES["M4"])
    spec.update(changes)
    with pytest.raises(ValueError, match="Robustness-table"):
        figures.validate_spec(spec)
