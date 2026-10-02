"""Focused tests for the frozen MNIST JointAE-32 metric."""

import unittest

import torch

from mem_param_src.metrics.image import (
    MNIST_F_MEM_ALPHAS,
    canonicalize_mnist32,
    compute_feature_statistics,
    compute_frechet_distance,
    compute_mnist_feature_memorization,
    compute_mnist_jointae32_fd,
    load_mnist_jointae32,
    load_mnist_jointae32_manifest,
    load_mnist_jointae32_reference_stats,
    normalize_mnist_memorization_features,
)


class TestMNISTCanonicalization(unittest.TestCase):
    def test_mnist64_and_replicated_rgb_share_one_canonical_space(self):
        base = torch.linspace(-1.0, 1.0, 64 * 64).reshape(1, 1, 64, 64)
        rgb = base.repeat(1, 3, 1, 1)

        grayscale = canonicalize_mnist32(base, image_range="-1_1")
        replicated = canonicalize_mnist32(rgb, image_range="-1_1")

        self.assertEqual(tuple(grayscale.shape), (1, 1, 32, 32))
        self.assertTrue(torch.equal(grayscale, replicated))
        self.assertGreaterEqual(float(grayscale.min()), 0.0)
        self.assertLessEqual(float(grayscale.max()), 1.0)

    def test_non_grayscale_rgb_is_rejected(self):
        images = torch.zeros(1, 3, 32, 32)
        images[:, 1] = 1.0

        with self.assertRaisesRegex(ValueError, "replicated grayscale"):
            canonicalize_mnist32(images, image_range="0_1")


class TestMNISTJointAEAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_mnist_jointae32(device="cpu")

    def test_loader_returns_frozen_raw_128d_features(self):
        images = torch.rand(2, 1, 32, 32)

        with torch.no_grad():
            features = self.encoder(images, image_range="0_1")

        self.assertEqual(tuple(features.shape), (2, 128))
        self.assertTrue(torch.isfinite(features).all())
        self.assertFalse(self.encoder.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.encoder.parameters())
        )

    def test_manifest_and_reference_statistics_identify_same_encoder(self):
        manifest = load_mnist_jointae32_manifest()
        reference = load_mnist_jointae32_reference_stats()

        self.assertEqual(manifest["feature_dimension"], 128)
        self.assertEqual(reference["global"]["count"], 10000)
        self.assertEqual(reference["global"]["feature_dim"], 128)
        self.assertEqual(
            reference["encoder_sha256"], manifest["weight_asset"]["sha256"]
        )
        self.assertEqual(len(reference["per_class"]), 10)


class TestMNISTFrechetDistance(unittest.TestCase):
    def test_identical_feature_statistics_have_zero_distance(self):
        torch.manual_seed(11)
        features = torch.randn(32, 8)
        statistics = compute_feature_statistics(features)

        distance = compute_frechet_distance(statistics, statistics)

        self.assertLess(distance, 1.0e-10)

    def test_fd_record_uses_explicit_mnist_metric_name(self):
        torch.manual_seed(13)
        features = torch.randn(20, 4)
        reference = compute_feature_statistics(torch.randn(24, 4))

        record = compute_mnist_jointae32_fd(
            features=features,
            reference_stats=reference,
        )

        self.assertEqual(record["metric"], "mnist_fd_jointae32_v1")
        self.assertNotIn("fid", record)
        self.assertEqual(record["n_samples"], 20)
        self.assertEqual(record["n_reference"], 24)


class TestMNISTFeatureMemorization(unittest.TestCase):
    def test_normalization_uses_the_evaluated_training_bank(self):
        training = torch.tensor(
            [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]]
        )
        query = training[[1, 3]].clone()

        query_vectors, training_vectors, mean, scale = (
            normalize_mnist_memorization_features(query, training)
        )

        self.assertTrue(torch.allclose(mean, training.mean(0, keepdim=True)))
        self.assertTrue(
            torch.allclose(
                scale,
                training.std(0, unbiased=False, keepdim=True),
            )
        )
        self.assertTrue(torch.allclose(query_vectors, training_vectors[[1, 3]]))

    def test_exact_copies_retain_both_neighbor_identities(self):
        torch.manual_seed(17)
        training = torch.randn(12, 6)
        query = training[[7, 2, 10]].clone()

        record = compute_mnist_feature_memorization(
            query,
            training,
            rho=0.05,
            query_batch_size=1,
            reference_batch_size=4,
        )

        self.assertEqual(record["nn_idx"].tolist(), [7, 2, 10])
        self.assertTrue(torch.equal(record["d1"], torch.zeros(3)))
        self.assertTrue(torch.equal(record["r"], torch.zeros(3)))
        self.assertEqual(record["f_mem"], 1.0)
        self.assertEqual(tuple(MNIST_F_MEM_ALPHAS), (0.005, 0.01, 0.02, 0.05))
        self.assertTrue(
            torch.all(record["nn_idx"] != record["second_nn_idx"])
        )


if __name__ == "__main__":
    unittest.main()
