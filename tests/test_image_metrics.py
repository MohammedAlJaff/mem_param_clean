"""Focused tests for the clean image-metric package."""

import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from mem_param_src.config import get_base_config
from mem_param_src.metrics import nearest_neighbors
from mem_param_src.metrics.image import (
    RHO_SWEEP,
    LPIPSFeatureEncoder,
    SSCDFeatureEncoder,
    collect_image32_metric_record,
    compute_f_mem_by_rho,
    compute_f_mem_reference_curve,
    compute_image_fid,
    compute_image_memorization_metrics,
    compute_nn_stats,
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    encode_image_features,
    make_random_projection_matrix,
)
from mem_param_src.metrics.image.preprocessing import normalize_images, to_uint8_rgb


class TestImagePreprocessing(unittest.TestCase):
    def test_normalizes_channel_last_uint8_images(self):
        images = torch.tensor(
            [
                [[[0], [255]], [[128], [64]]],
                [[[255], [0]], [[32], [16]]],
            ],
            dtype=torch.uint8,
        )

        normalized = normalize_images(images, image_range="0_255")

        self.assertEqual(tuple(normalized.shape), (2, 1, 2, 2))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)

    def test_uint8_rgb_handles_minus_one_to_one_grayscale(self):
        images = torch.tensor([[[[-1.0, 0.0], [1.0, 0.5]]]])

        converted = to_uint8_rgb(images, image_range="-1_1")

        self.assertEqual(tuple(converted.shape), (1, 3, 2, 2))
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertTrue(torch.equal(converted[:, 0], converted[:, 1]))
        self.assertTrue(torch.equal(converted[:, 1], converted[:, 2]))
        self.assertEqual(int(converted[0, 0, 0, 0]), 0)
        self.assertEqual(int(converted[0, 0, 1, 0]), 255)


class TestImageMemorization(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.25, 0.25], [0.25, 0.25]]],
                [[[0.75, 0.75], [0.75, 0.75]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ]
        )

    def test_reuses_generic_nearest_neighbor_math(self):
        self.assertIs(compute_nn_stats, nearest_neighbors.compute_nn_stats)

    def test_pixel_metrics_detect_exact_copies(self):
        metrics = compute_image_memorization_metrics(
            self.reference[[0, 2]],
            self.reference,
            rho=0.5,
            query_batch_size=1,
            reference_batch_size=2,
            image_range="0_1",
        )

        self.assertEqual(metrics["space"], "pixels")
        self.assertEqual(metrics["encoder"], "none")
        self.assertEqual(metrics["feature_dim"], 4)
        self.assertEqual(metrics["f_mem"], 1.0)

    def test_callable_encoder_metrics_restore_training_mode(self):
        class DummyEncoder(torch.nn.Module):
            def forward(self, batch):
                return {"features": batch.flatten(1)[:, :2]}

        encoder = DummyEncoder()
        encoder.train()

        metrics = compute_image_memorization_metrics(
            self.reference[[0, 2]],
            self.reference,
            encoder=encoder,
            rho=0.5,
            query_batch_size=1,
            reference_batch_size=2,
            image_range="0_1",
        )

        self.assertTrue(encoder.training)
        self.assertEqual(metrics["space"], "features")
        self.assertEqual(metrics["encoder"], "DummyEncoder")
        self.assertEqual(metrics["feature_dim"], 2)
        self.assertEqual(metrics["f_mem"], 1.0)

    def test_rho_sweep(self):
        sweep = compute_f_mem_by_rho(
            torch.tensor([0.2, 0.6, 0.9]),
            [0.5, 0.8],
        )

        self.assertAlmostEqual(sweep[0.5], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(sweep[0.8], 2.0 / 3.0, places=6)


class TestProjectedImageMemorization(unittest.TestCase):
    def test_seeded_and_explicit_projection_match(self):
        torch.manual_seed(3)
        reference = torch.rand(6, 1, 4, 4)
        samples = reference[:2].clone()
        projection = make_random_projection_matrix(4, 3, 11)

        seeded = compute_projected_image_memorization_metrics(
            samples,
            reference,
            resize_size=2,
            projection_dim=3,
            projection_seed=11,
            image_range="0_1",
        )
        explicit = compute_projected_image_memorization_metrics(
            samples,
            reference,
            resize_size=2,
            projection_dim=3,
            projection_matrix=projection,
            image_range="0_1",
        )

        for key in ("f_mem", "mean_r", "median_r", "mean_d1", "mean_d2"):
            self.assertAlmostEqual(seeded[key], explicit[key], places=6)

    def test_cached_reference_features_match_direct_computation(self):
        torch.manual_seed(5)
        reference = torch.rand(7, 1, 4, 4)
        samples = reference[:3].clone()
        cached_reference = compute_projected_image_features(
            reference,
            resize_size=2,
            projection_dim=5,
            projection_seed=19,
            image_range="0_1",
            batch_size=2,
        )

        direct = compute_projected_image_memorization_metrics(
            samples,
            reference,
            resize_size=2,
            projection_dim=5,
            projection_seed=19,
            rho=0.5,
            image_range="0_1",
        )
        cached = compute_projected_image_memorization_metrics(
            samples,
            projected_reference_vectors=cached_reference,
            resize_size=2,
            projection_dim=5,
            projection_seed=19,
            rho=0.5,
            image_range="0_1",
        )

        self.assertFalse(direct["projected_reference_cached"])
        self.assertTrue(cached["projected_reference_cached"])
        for key in ("f_mem", "mean_r", "median_r", "mean_d1", "mean_d2"):
            self.assertAlmostEqual(direct[key], cached[key], places=6)


class TestImageMetricRecords(unittest.TestCase):
    def test_legacy_threshold_name_is_rejected(self):
        config = get_base_config()
        del config.evaluation.f_mem_threshold
        config.evaluation.ratio_threshold = 0.5

        with self.assertRaisesRegex(ValueError, "evaluation.f_mem_threshold"):
            nearest_neighbors.resolve_rho(config=config)

    def test_checkpoint_record_has_threshold_sweep_and_delta(self):
        config = get_base_config()
        config.evaluation.f_mem_threshold = 0.3
        train = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.3, 0.3], [0.3, 0.3]]],
                [[[0.8, 0.8], [0.8, 0.8]]],
            ]
        )

        record = collect_image32_metric_record(
            5,
            train[:2].clone(),
            train[[1, 2]].clone(),
            train,
            chunk_size=2,
            config=config,
        )

        self.assertEqual(record["step"], 5)
        self.assertEqual(record["rho"], 0.3)
        self.assertAlmostEqual(
            record["delta_f_mem"],
            record["f_mem"] - record["baseline_f_mem"],
            places=6,
        )
        for rho in RHO_SWEEP:
            self.assertIn(f"f_mem_rho{rho}", record)
            self.assertIn(f"baseline_f_mem_rho{rho}", record)
            self.assertIn(f"delta_f_mem_rho{rho}", record)


class TestImageFid(unittest.TestCase):
    def test_tensor_reference_uses_uint8_dataset_adapter(self):
        calculate_metrics = mock.Mock(return_value={"frechet_inception_distance": 12.5})
        torch_fidelity = SimpleNamespace(calculate_metrics=calculate_metrics)
        samples = torch.rand(2, 1, 4, 4)
        reference = torch.rand(3, 1, 4, 4)

        with mock.patch.dict(sys.modules, {"torch_fidelity": torch_fidelity}):
            metrics = compute_image_fid(
                samples,
                reference,
                batch_size=2,
                device="cpu",
                image_range="0_1",
            )

        kwargs = calculate_metrics.call_args.kwargs
        self.assertEqual(metrics["fid"], 12.5)
        self.assertEqual(metrics["n_samples"], 2)
        self.assertEqual(metrics["n_reference"], 3)
        self.assertFalse(kwargs["cuda"])
        self.assertFalse(kwargs.get("isc", False))
        self.assertEqual(kwargs["input1"][0].dtype, torch.uint8)
        self.assertEqual(tuple(kwargs["input1"][0].shape), (3, 4, 4))

    def test_reference_stats_enable_torch_fidelity_stats_path(self):
        calculate_metrics = mock.Mock(return_value={"frechet_inception_distance": 4.0})
        torch_fidelity = SimpleNamespace(calculate_metrics=calculate_metrics)

        with mock.patch.dict(sys.modules, {"torch_fidelity": torch_fidelity}):
            metrics = compute_image_fid(
                torch.rand(2, 3, 4, 4),
                reference_stats_path="reference_stats.npz",
                device="cpu",
                image_range="0_1",
            )

        kwargs = calculate_metrics.call_args.kwargs
        self.assertTrue(kwargs["isc"])
        self.assertEqual(kwargs["fid_statistics_file"], "reference_stats.npz")
        self.assertIsNone(metrics["n_reference"])
        self.assertEqual(metrics["reference_source"], "stats:reference_stats.npz")


class TestRawFeatureMemorization(unittest.TestCase):
    def test_chunked_neighbors_keep_both_reference_indices(self):
        reference = torch.tensor([[0.0], [0.3], [0.8], [1.0]])
        samples = torch.tensor([[0.05], [0.95]])

        direct = compute_nn_stats(samples, reference, query_batch_size=2)
        chunked = compute_nn_stats(
            samples,
            reference,
            query_batch_size=1,
            reference_batch_size=2,
        )

        for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
            self.assertTrue(torch.equal(direct[key], chunked[key]))
        self.assertEqual(direct["nn_idx"].tolist(), [0, 3])
        self.assertEqual(direct["second_nn_idx"].tolist(), [1, 2])

    def test_exact_training_copies_have_zero_ratio_and_correct_identity(self):
        reference = torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.9, 1.0]])
        samples = reference[torch.tensor([2, 0])]

        stats = compute_nn_stats(samples, reference, query_batch_size=1)

        self.assertTrue(torch.equal(stats["d1"], torch.zeros(2)))
        self.assertTrue(torch.equal(stats["r"], torch.zeros(2)))
        self.assertEqual(stats["nn_idx"].tolist(), [2, 0])
        self.assertNotEqual(stats["nn_idx"].tolist(), stats["second_nn_idx"].tolist())

    def test_heldout_curve_keeps_thresholds_as_derived_values(self):
        calibration = torch.arange(1, 101, dtype=torch.float32) / 100.0
        validation = calibration.clone()
        generated = torch.tensor([0.001, 0.2, 0.8])

        records = compute_f_mem_reference_curve(
            generated,
            calibration,
            validation,
            quantiles=[0.01, 0.05],
        )

        self.assertEqual([row["heldout_quantile"] for row in records], [0.01, 0.05])
        self.assertLess(records[0]["rho"], records[1]["rho"])
        self.assertAlmostEqual(records[0]["generated_f_mem"], 1.0 / 3.0)
        self.assertIn("validation_rate", records[0])

    def test_lpips_embedding_matches_the_package_distance(self):
        try:
            import lpips
        except ModuleNotFoundError:
            self.skipTest("lpips is not installed in this environment")

        torch.manual_seed(31)
        model = lpips.LPIPS(
            net="alex",
            version="0.1",
            pnet_rand=True,
            verbose=False,
        )
        encoder = LPIPSFeatureEncoder(model=model, input_size=32)
        images = torch.rand(2, 1, 32, 32)
        features = encode_image_features(
            images,
            encoder=encoder,
            batch_size=2,
            image_range="0_1",
        )
        rgb = images.repeat(1, 3, 1, 1)
        expected = model(rgb[:1] * 2.0 - 1.0, rgb[1:] * 2.0 - 1.0)
        squared_distance = (features[0] - features[1]).square().sum()

        self.assertEqual(tuple(features.shape), (2, 5760))
        self.assertAlmostEqual(
            float(squared_distance),
            float(expected.flatten()[0]),
            places=5,
        )

    def test_sscd_adapter_is_deterministic_finite_and_512_dimensional(self):
        class FakeSSCD(torch.nn.Module):
            def forward(self, images):
                means = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
                return means.repeat(1, 512) + 1.0

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/sscd.pt"
            example = torch.rand(1, 3, 16, 16)
            torch.jit.trace(FakeSSCD(), example).save(path)
            encoder = SSCDFeatureEncoder(path, input_size=16)
            images = torch.rand(3, 1, 32, 32)
            first = encode_image_features(
                images, encoder=encoder, batch_size=2, image_range="0_1"
            )
            second = encode_image_features(
                images, encoder=encoder, batch_size=1, image_range="0_1"
            )

        self.assertEqual(tuple(first.shape), (3, 512))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(first.norm(dim=1), torch.ones(3), atol=1.0e-6)
        )


if __name__ == "__main__":
    unittest.main()
