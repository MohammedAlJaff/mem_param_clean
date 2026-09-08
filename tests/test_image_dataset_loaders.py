"""Tests for MNIST and CIFAR-10 image-shape controls."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from mem_param_src.config import get_base_config
from mem_param_src.datasets.cifar10 import load_cifar10_bundle
from mem_param_src.datasets.mnist import _match_image_channels, load_mnist_bundle


class _FakeCIFAR10:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.classes = [str(index) for index in range(10)]
        self.class_to_idx = {str(index): index for index in range(10)}

    def __len__(self):
        return self.data.shape[0]


class TestMNISTChannels(unittest.TestCase):
    def test_grayscale_values_can_repeat_across_rgb_channels(self):
        tensor = torch.linspace(-1.0, 1.0, 18).reshape(2, 1, 3, 3)
        config = get_base_config()

        config.data.image_channels = 1
        self.assertTrue(torch.equal(_match_image_channels(tensor, config), tensor))

        config.data.image_channels = 3
        rgb = _match_image_channels(tensor, config)
        self.assertEqual(tuple(rgb.shape), (2, 3, 3, 3))
        self.assertTrue(torch.equal(rgb[:, 0], rgb[:, 1]))
        self.assertTrue(torch.equal(rgb[:, 1], rgb[:, 2]))

        config.data.image_channels = 2
        with self.assertRaisesRegex(ValueError, "1 or 3"):
            _match_image_channels(tensor, config)

    def test_train_and_reference_subsets_can_be_class_balanced(self):
        images = torch.arange(30 * 28 * 28).remainder(256).to(torch.uint8)
        images = images.reshape(30, 28, 28)
        targets = torch.arange(30) % 10
        config = get_base_config()
        config.data.dataset = "mnist"
        config.data.n_train = 20
        config.data.n_test = 10
        config.data.image_size = 28
        config.data.image_channels = 1
        config.data.balance_train_classes = True
        config.data.balance_reference_classes = True

        with patch(
            "mem_param_src.datasets.mnist._load_mnist_split",
            side_effect=((images, targets), (images, targets)),
        ):
            bundle = load_mnist_bundle(config)

        self.assertEqual(set(bundle["train_class_counts"].values()), {2})
        self.assertEqual(set(bundle["reference_class_counts"].values()), {1})
        self.assertIs(bundle["balanced_train_classes"], True)
        self.assertIs(bundle["balanced_reference_classes"], True)


class TestCIFAR10Bundle(unittest.TestCase):
    def _dataset(self, count, offset):
        values = np.arange(count * 32 * 32 * 3, dtype=np.uint32)
        data = ((values + offset) % 256).astype(np.uint8).reshape(count, 32, 32, 3)
        targets = [index % 10 for index in range(count)]
        return _FakeCIFAR10(data, targets)

    def test_fixed_bundle_supports_size_channels_and_seed(self):
        train_set = self._dataset(20, 0)
        reference_set = self._dataset(12, 17)

        config = get_base_config()
        config.data.dataset = "cifar10"
        config.data.n_train = 6
        config.data.n_test = 4
        config.data.image_size = 16
        config.data.image_channels = 1
        config.training.seed = 5

        with patch(
            "mem_param_src.datasets.cifar10.datasets.CIFAR10",
            side_effect=(train_set, reference_set, train_set, reference_set),
        ):
            first = load_cifar10_bundle(config)
            repeat = load_cifar10_bundle(config)

        self.assertEqual(tuple(first["train"].shape), (6, 1, 16, 16))
        self.assertEqual(tuple(first["reference"].shape), (4, 1, 16, 16))
        self.assertTrue(torch.isfinite(first["train"]).all())
        self.assertGreaterEqual(float(first["train"].min()), -1.0)
        self.assertLessEqual(float(first["train"].max()), 1.0)
        self.assertTrue(torch.equal(first["train_indices"], repeat["train_indices"]))
        self.assertTrue(torch.equal(first["train"], repeat["train"]))

        config.data.image_channels = 3
        config.data.image_size = 64
        config.training.seed = 6
        with patch(
            "mem_param_src.datasets.cifar10.datasets.CIFAR10",
            side_effect=(train_set, reference_set),
        ):
            changed = load_cifar10_bundle(config)
        self.assertEqual(tuple(changed["train"].shape), (6, 3, 64, 64))
        self.assertFalse(torch.equal(first["train_indices"], changed["train_indices"]))

    def test_train_and_reference_subsets_can_be_class_balanced(self):
        train_set = self._dataset(30, 0)
        reference_set = self._dataset(20, 17)
        config = get_base_config()
        config.data.dataset = "cifar10"
        config.data.n_train = 20
        config.data.n_test = 10
        config.data.image_size = 32
        config.data.image_channels = 3
        config.data.balance_train_classes = True
        config.data.balance_reference_classes = True

        with patch(
            "mem_param_src.datasets.cifar10.datasets.CIFAR10",
            side_effect=(train_set, reference_set),
        ):
            bundle = load_cifar10_bundle(config)

        self.assertEqual(set(bundle["train_class_counts"].values()), {2})
        self.assertEqual(set(bundle["reference_class_counts"].values()), {1})
        self.assertIs(bundle["balanced_train_classes"], True)
        self.assertIs(bundle["balanced_reference_classes"], True)

    def test_exact_balance_rejects_an_incompatible_sample_count(self):
        train_set = self._dataset(20, 0)
        reference_set = self._dataset(20, 17)
        config = get_base_config()
        config.data.dataset = "cifar10"
        config.data.n_train = 6
        config.data.n_test = 10
        config.data.image_size = 32
        config.data.image_channels = 3
        config.data.balance_train_classes = True

        with (
            patch(
                "mem_param_src.datasets.cifar10.datasets.CIFAR10",
                side_effect=(train_set, reference_set),
            ),
            self.assertRaisesRegex(ValueError, "divisible by 10"),
        ):
            load_cifar10_bundle(config)


if __name__ == "__main__":
    unittest.main()
