"""Tests for configurable CelebA image transforms."""

import unittest

import torch
from PIL import Image

from mem_param_src.config import get_base_config
from mem_param_src.datasets.celeba32 import _build_transform


class TestCelebATransform(unittest.TestCase):
    def test_requested_resolution_and_channels(self):
        image = Image.new("RGB", (178, 218), color=(32, 128, 224))

        for image_size in (16, 32, 64, 512):
            for channels in (1, 3):
                with self.subTest(image_size=image_size, channels=channels):
                    config = get_base_config()
                    config.data.image_size = image_size
                    config.data.resize_size = image_size
                    config.data.image_channels = channels

                    tensor = _build_transform(config)(image)

                    self.assertEqual(
                        tuple(tensor.shape),
                        (channels, image_size, image_size),
                    )
                    self.assertTrue(torch.isfinite(tensor).all())
                    self.assertGreaterEqual(float(tensor.min()), -1.0)
                    self.assertLessEqual(float(tensor.max()), 1.0)

    def test_invalid_image_configuration_is_rejected(self):
        config = get_base_config()
        config.data.image_size = 0
        config.data.image_channels = 1
        with self.assertRaisesRegex(ValueError, "image_size"):
            _build_transform(config)

        config.data.image_size = 32
        config.data.image_channels = 2
        with self.assertRaisesRegex(ValueError, "1 or 3"):
            _build_transform(config)

        config.data.image_channels = 3
        config.data.resize_size = 16
        with self.assertRaisesRegex(ValueError, "at least"):
            _build_transform(config)


if __name__ == "__main__":
    unittest.main()
