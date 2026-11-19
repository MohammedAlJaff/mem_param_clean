"""Tests for the saved-checkpoint sampling benchmark."""

import unittest

import torch

from mem_param_src.runners.sampling_benchmark import (
    build_evaluation_noise,
    summarize_sampling_runtime,
)


class TestEvaluationNoise(unittest.TestCase):
    def test_extension_preserves_saved_prefix_and_continues_seeded_stream(self):
        generator = torch.Generator().manual_seed(2003)
        saved = torch.randn((4, 1, 2, 2), generator=generator)
        expected_tail = torch.randn((6, 1, 2, 2), generator=generator)

        extended = build_evaluation_noise(saved, 10, 2003)

        self.assertTrue(torch.equal(extended[:4], saved))
        self.assertTrue(torch.equal(extended[4:], expected_tail))

    def test_extension_rejects_a_seed_that_does_not_reproduce_saved_noise(self):
        saved = torch.randn((4, 1, 2, 2), generator=torch.Generator().manual_seed(7))

        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            build_evaluation_noise(saved, 10, 8)


class TestRuntimeProjection(unittest.TestCase):
    def test_full_measurement_reports_measured_checkpoint_and_run_projection(self):
        result = summarize_sampling_runtime(10000, 600.0, 10000, 72)

        self.assertEqual(result["one_checkpoint_timing_kind"], "measured")
        self.assertEqual(result["one_checkpoint_sampling_minutes"], 10.0)
        self.assertEqual(result["all_checkpoints_sampling_hours"], 12.0)

    def test_partial_measurement_projects_both_scales(self):
        result = summarize_sampling_runtime(1000, 60.0, 10000, 72)

        self.assertEqual(result["one_checkpoint_timing_kind"], "projected")
        self.assertEqual(result["one_checkpoint_sampling_minutes"], 10.0)
        self.assertEqual(result["all_checkpoints_sampling_hours"], 12.0)


if __name__ == "__main__":
    unittest.main()
