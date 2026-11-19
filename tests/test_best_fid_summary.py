"""Check calibration selection and pairing memorization with per-seed best FID."""

import csv
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting" / "celeba32_3x3"))
import plot_f_mem_grid as grid
import summarize_f_mem_at_best_fid as summary


def test_sscd_selects_alpha_without_using_fixed_pixel_rho(tmp_path, monkeypatch):
    monkeypatch.setattr(grid, "EXPECTED_CHECKPOINT_COUNT", 2)
    fields = ("run_id", "step", "metric", "space", "alpha", "rho", "sample_count", "value")
    with (tmp_path / "evaluation_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for step in (0, 1_000_000):
            writer.writerow(("run", step, "f_mem_sscd", "", 0.01, 0.92, 10000, 0.6))
            writer.writerow(("run", step, "f_mem_sscd", "", 0.02, 0.94, 10000, 0.8))
    key = (512, "x", "x", 0)
    result = grid._load_series(tmp_path, {key: "run"}, None, "f_mem_sscd", alpha=0.01)
    assert result[key] == {0: 0.6, 1_000_000: 0.6}
    with pytest.raises(ValueError, match="not a fixed rho"):
        grid._load_series(tmp_path, {key: "run"}, 0.3, "f_mem_sscd", alpha=0.01)


@pytest.mark.parametrize("experiment,seeds", [
    ("celeba32_unet_3x3", (0, 1, 2)), ("celeba32_dit_3x3", (0,)),
])
def test_each_seed_selects_own_best_fid_with_early_ties(monkeypatch, experiment, seeds):
    monkeypatch.setattr(summary, "TARGETS", ("x",))
    runs = {(512, "x", "x", seed): f"run{seed}" for seed in seeds}
    monkeypatch.setattr(summary, "_selected_runs", lambda *args: runs)
    fid = {key: ({0: 10, 100: 5} if key[-1] == 0 else {0: 5, 100: 5}) for key in runs}
    mem = {key: {0: 0.1 * key[-1], 100: 0.9} for key in runs}
    monkeypatch.setattr(summary, "_load_series",
                        lambda root, runs, rho, metric, *args: fid if metric == "inception_fid" else mem)
    selected, rows = summary.summarize(experiment=experiment, n_trains=(512,))
    assert [row["step"] for row in selected] == ([100, 0, 0] if len(seeds) > 1 else [100])
    values = [0.9, 0.1, 0.2] if len(seeds) > 1 else [0.9]
    assert rows[0]["f_mem_raw_pixels_mean"] == pytest.approx(statistics.mean(values))
    if len(seeds) > 1:
        assert rows[0]["f_mem_raw_pixels_std"] == pytest.approx(statistics.stdev(values))
    else:
        assert rows[0]["f_mem_raw_pixels_std"] is None
        assert rows[0]["inception_fid_std"] is None
