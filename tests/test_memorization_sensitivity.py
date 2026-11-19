"""Check that changing the readout cannot change the selected checkpoint."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from plot_memorization_sensitivity import ALPHA, METRICS, RAW, RHOS, select_readouts
import plot_memorization_sensitivity as sensitivity


def paired_records():
    quality = [dict(run_id="run", prediction_space="x", loss_space="x", step=s, value=v)
               for s, v in ((10, 12), (20, 8), (30, 8))]
    evaluation = []
    for step in (10, 20, 30):
        for metric, alpha, rho, value in (
            *((RAW, "", str(r), r) for r in RHOS),
            *((m, str(ALPHA), "0.8", 0.6) for m in METRICS),
        ):
            evaluation.append(dict(run_id="run", step=str(step), metric=metric,
                                   space="raw_pixels" if metric == RAW else "",
                                   alpha=alpha, rho=rho, value=str(value), sample_count="10000"))
    return quality, evaluation


def test_same_best_fid_step_with_earliest_tie_and_distinct_pixel_rules():
    quality, evaluation = paired_records()
    # A better memorization value at another step must not select that step.
    evaluation[0]["value"] = "0.0"
    row, = select_readouts(quality, evaluation)
    assert row["step"] == 20
    assert row["readouts"][(RAW, "rho", 0.3)] == 0.3
    assert row["readouts"][(RAW, "alpha", ALPHA)] == 0.6


def test_missing_selected_metric_does_not_fall_back_to_another_step():
    quality, evaluation = paired_records()
    evaluation = [r for r in evaluation if not (r["metric"] == "f_mem_sscd" and r["step"] == "20")]
    with pytest.raises(ValueError, match="Incomplete"):
        select_readouts(quality, evaluation)


def test_duplicate_selected_readout_is_rejected():
    quality, evaluation = paired_records()
    evaluation.append(copy.copy(next(r for r in evaluation if r["step"] == "20")))
    with pytest.raises(ValueError, match="Duplicate"):
        select_readouts(quality, evaluation)


def test_trajectory_means_include_zero_seeds_before_log_display(monkeypatch):
    runs = {(n, t, t, seed): str((n, t, seed))
            for n in sensitivity.N_TRAINS for t in sensitivity.TARGETS for seed in (0, 1, 2)}
    monkeypatch.setattr(sensitivity, "_selected_runs", lambda *args: runs)
    monkeypatch.setattr(sensitivity, "_load_series", lambda root, runs, rho, metric: {
        key: {0: 0.0, 250: rho if key[-1] == 2 else 0.0} for key in runs
    })
    _, curves = sensitivity.load_trajectories()
    curve = curves[(0.3, "eps", 512)]
    assert curve["mean"][0] == 0
    assert curve["mean"][1] == pytest.approx(0.1)
    assert curve["std"][1] == pytest.approx(0.3 / 3**0.5)
