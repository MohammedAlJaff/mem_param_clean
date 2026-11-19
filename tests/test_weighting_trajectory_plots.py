"""Tests for seed-aggregated loss-weighting trajectories."""

import math
import unittest

from mem_param_src.runners.weighting_trajectory_plots import (
    _aggregate_observations,
    _run_identity,
    _series_key,
)


class TestWeightingTrajectoryPlots(unittest.TestCase):
    def test_run_identity_parses_the_frozen_matrix_names(self):
        self.assertEqual(
            _run_identity("n2048_epseq_s2"),
            {"n_train": 2048, "weighting": "epseq", "seed": 2},
        )
        self.assertIsNone(_run_identity("_job"))
        self.assertIsNone(_run_identity("n512_epseq_s3"))

    def test_aggregate_uses_mean_and_sample_standard_deviation(self):
        series = _series_key("inception_fid")
        observations = {
            (512, "xeq", series, 0): {0: 1.0, 1: 2.0, 2: 3.0},
        }

        rows = _aggregate_observations(observations)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mean"], 2.0)
        self.assertEqual(rows[0]["std"], 1.0)
        self.assertEqual(rows[0]["minimum"], 1.0)
        self.assertEqual(rows[0]["maximum"], 3.0)
        self.assertTrue(math.isfinite(rows[0]["std"]))

    def test_aggregate_rejects_a_missing_seed(self):
        series = _series_key("inception_fid")
        observations = {
            (512, "unit", series, 0): {0: 1.0, 2: 3.0},
        }

        with self.assertRaisesRegex(ValueError, "Incomplete seeds"):
            _aggregate_observations(observations)


if __name__ == "__main__":
    unittest.main()
