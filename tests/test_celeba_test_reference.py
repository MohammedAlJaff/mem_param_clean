"""Tests for the clean CelebA test-reference builder."""

import unittest

import torch
from PIL import Image
from torchvision import transforms

from mem_param_src.runners.artifacts import tensor_sha256
from mem_param_src.runners.celeba_test_reference import (
    _build_transform_and_verify_source,
    map_split_indices_to_filenames,
    select_clean_reference,
)


class TestSplitCatalogMapping(unittest.TestCase):
    def test_split_local_indices_map_through_global_filenames(self):
        filenames = ["000101.jpg", "000205.jpg", "001009.jpg"]

        mapped = map_split_indices_to_filenames([2, 0], filenames)

        self.assertEqual(mapped, ["001009.jpg", "000101.jpg"])


class TestCleanCandidateSelection(unittest.TestCase):
    def test_rejects_raw_and_transformed_collisions(self):
        tensors = {
            0: torch.tensor([[[0.0]]]),
            1: torch.tensor([[[1.0]]]),
            2: torch.tensor([[[2.0]]]),
            3: torch.tensor([[[3.0]]]),
            4: torch.tensor([[[1.0]]]),
            5: torch.tensor([[[5.0]]]),
        }
        jpeg_hashes = {
            0: "training-jpeg",
            1: "accepted-jpeg",
            2: "accepted-jpeg",
            3: "tensor-collision-jpeg",
            4: "transform-collision-jpeg",
            5: "clean-jpeg",
        }
        training_tensor = tensors[3]

        selected = select_clean_reference(
            range(6),
            2,
            filename_for_index=lambda index: f"test-{index}.jpg",
            jpeg_hash_for_index=lambda index: jpeg_hashes[index],
            tensor_and_label_for_index=lambda index: (tensors[index], index),
            training_jpeg_hashes={"training-jpeg"},
            training_tensor_hashes={tensor_sha256(training_tensor)},
        )

        self.assertEqual(selected["indices"], [1, 5])
        self.assertEqual(
            selected["rejection_counts"],
            {
                "duplicate_filename": 0,
                "training_jpeg_overlap": 1,
                "internal_jpeg_duplicate": 1,
                "training_tensor_overlap": 1,
                "internal_tensor_duplicate": 1,
            },
        )

    def test_refills_deterministically_to_exactly_10000(self):
        candidate_count = 10003
        order = torch.randperm(
            candidate_count, generator=torch.Generator().manual_seed(1)
        ).tolist()
        training_tensor = torch.tensor([[[2.0]]])
        training_tensor_hashes = {tensor_sha256(training_tensor)}

        def jpeg_hash(index):
            if index == 0:
                return "training-jpeg"
            if index in (1, 3):
                return "internal-duplicate"
            return f"jpeg-{index}"

        def tensor_and_label(index):
            if index == 2:
                return training_tensor, index
            return torch.tensor([[[float(index)]]]), index

        kwargs = {
            "filename_for_index": lambda index: f"{index:06d}.jpg",
            "jpeg_hash_for_index": jpeg_hash,
            "tensor_and_label_for_index": tensor_and_label,
            "training_jpeg_hashes": {"training-jpeg"},
            "training_tensor_hashes": training_tensor_hashes,
        }
        first = select_clean_reference(order, 10000, **kwargs)
        second = select_clean_reference(order, 10000, **kwargs)

        self.assertEqual(len(first["indices"]), 10000)
        self.assertEqual(first["indices"], second["indices"])
        self.assertEqual(first["candidates_examined"], candidate_count)
        self.assertEqual(sum(first["rejection_counts"].values()), 3)


class TestExactCelebATransform(unittest.TestCase):
    def test_builder_uses_the_training_transform(self):
        raw = bytes((index * 37) % 256 for index in range(180 * 160 * 3))
        image = Image.frombytes("RGB", (180, 160), raw)
        actual_transform = _build_transform_and_verify_source()
        expected_transform = transforms.Compose(
            [
                transforms.CenterCrop(140),
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        actual = actual_transform(image)
        expected = expected_transform(image)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), (1, 32, 32))

    def test_builder_supports_the_saved_rgb64_transform(self):
        raw = bytes((index * 37) % 256 for index in range(180 * 160 * 3))
        image = Image.frombytes("RGB", (180, 160), raw)
        actual_transform = _build_transform_and_verify_source(
            image_size=64,
            image_channels=3,
            resize_size=64,
        )
        expected_transform = transforms.Compose(
            [
                transforms.CenterCrop(140),
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.Lambda(lambda value: value.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

        actual = actual_transform(image)
        expected = expected_transform(image)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
