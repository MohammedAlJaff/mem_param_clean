"""Check seed aggregation and absent uncertainty for joint trajectory grids."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from celeba32_3x3 import plot_f_mem_grid as grid


@pytest.mark.parametrize("seeds,expected_mean,expected_sd", [
    ((0, 1, 2), 0.1, 0.3 / (3 ** 0.5)),
    ((0,), 0.0, None),
])
def test_seed_means_include_zeros_and_single_seed_has_no_sd(monkeypatch, seeds, expected_mean, expected_sd):
    series = {(512, prediction, loss, seed): {0: 0.0, 250: (0.3 if seed == 2 else 0.0)}
              for prediction in grid.TARGETS for loss in grid.TARGETS for seed in seeds}
    monkeypatch.setattr(grid, "_selected_runs", lambda *args: {})
    monkeypatch.setattr(grid, "_load_series", lambda *args: series)
    rows = grid._aggregate(Path("unused"), (512,), 0.3, "f_mem_raw_pixels", seeds=seeds)
    positive = [row for row in rows if row["step"] == 250]
    assert len(positive) == 9
    for row in positive:
        assert row["mean"] == pytest.approx(expected_mean)
        assert row["std"] == pytest.approx(expected_sd) if expected_sd is not None else row["std"] is None


def test_different_seed_checkpoint_grids_are_rejected(monkeypatch):
    series = {(512, prediction, loss, seed): {0: 0.0, 250: 0.1}
              for prediction in grid.TARGETS for loss in grid.TARGETS for seed in (0, 1, 2)}
    series[(512, "x", "x", 1)] = {0: 0.0, 500: 0.1}
    monkeypatch.setattr(grid, "_selected_runs", lambda *args: {})
    monkeypatch.setattr(grid, "_load_series", lambda *args: series)
    with pytest.raises(ValueError, match="Seed checkpoint grids differ"):
        grid._aggregate(Path("unused"), (512,), 0.3, "f_mem_raw_pixels")
