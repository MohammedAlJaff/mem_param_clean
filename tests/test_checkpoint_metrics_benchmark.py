"""Tests for the interactive checkpoint metric benchmark."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from mem_param_src.runners.artifacts import save_torch, tensor_sha256, write_json
from mem_param_src.runners.checkpoint_metrics_benchmark import (
    _curve_record,
    _expected_generated_bank_identity,
    _load_or_sample,
    _validate_reference_compatibility,
    _validate_reference_payload,
    evaluate_checkpoint_metrics,
    project_full_run_runtime,
)


class TestMetricRuntimeProjection(unittest.TestCase):
    def test_shared_reference_work_is_charged_once(self):
        result = project_full_run_runtime(72, 180.0, 600.0, 120.0)

        self.assertEqual(result["per_checkpoint_sampling_and_metrics_seconds"], 300.0)
        self.assertEqual(result["per_checkpoint_sampling_and_metrics_minutes"], 5.0)
        self.assertEqual(result["all_checkpoints_projected_seconds"], 22200.0)


class TestHeldoutCalibration(unittest.TestCase):
    def test_curve_includes_alpha_point_zero_one_headline(self):
        generated = {"r": torch.tensor([0.1, 0.2, 0.9])}
        calibration = {"r": torch.linspace(0.1, 0.9, 100)}
        validation = {"r": torch.linspace(0.15, 0.95, 100)}

        curve, headline = _curve_record(generated, calibration, validation)

        self.assertEqual(len(curve), 4)
        self.assertEqual(headline["heldout_quantile"], 0.01)
        self.assertIn("validation_rate", headline)


class TestFrozenReferenceCompatibility(unittest.TestCase):
    def setUp(self):
        self.training = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        self.reference = torch.arange(20, 40, dtype=torch.float32).reshape(5, 1, 2, 2)
        self.bundle = {"train": self.training}
        self.payload = {
            "reference_split": "test",
            "reference_tensor_sha256s": [
                tensor_sha256(image) for image in self.reference
            ],
        }
        self.data_manifest = {
            "preprocessing": {
                "image_channels": 1,
                "image_size": 2,
                "resize_size": 2,
                "training_augmentation": False,
                "value_range": [-1.0, 1.0],
            }
        }
        self.manifest = {
            "reference_split": "test",
            "preprocessing": self.data_manifest["preprocessing"].copy(),
        }

    def test_clean_test_reference_is_accepted(self):
        _validate_reference_compatibility(
            self.bundle,
            self.data_manifest,
            self.reference,
            self.payload,
            self.manifest,
        )

    def test_training_tensor_collision_is_rejected(self):
        self.payload["reference_tensor_sha256s"][0] = tensor_sha256(self.training[0])
        with self.assertRaisesRegex(ValueError, "contaminated"):
            _validate_reference_compatibility(
                self.bundle,
                self.data_manifest,
                self.reference,
                self.payload,
                self.manifest,
            )

    def test_reference_must_use_official_test_split(self):
        self.manifest["reference_split"] = "valid"
        with self.assertRaisesRegex(ValueError, "official test split"):
            _validate_reference_compatibility(
                self.bundle,
                self.data_manifest,
                self.reference,
                self.payload,
                self.manifest,
            )


class TestFrozenReferencePayload(unittest.TestCase):
    def test_rgb64_shape_is_read_from_the_manifest(self):
        reference = torch.arange(5 * 3 * 64 * 64, dtype=torch.float32).reshape(
            5, 3, 64, 64
        )
        image_hashes = [tensor_sha256(image) for image in reference]
        payload = {
            "reference_split": "test",
            "reference_source_id": "torchvision_celeba:test",
            "reference_filenames": [f"{index:06d}.jpg" for index in range(5)],
            "reference_jpeg_sha256s": [f"jpeg-{index}" for index in range(5)],
            "reference_tensor_sha256s": image_hashes,
            "reference_indices": torch.arange(5),
            "reference_labels": torch.arange(5),
            "reference_tensor_sha256": tensor_sha256(reference),
            "training_union_manifest_sha256": "training-union",
            "partition_file_sha256": "partition",
        }
        manifest = {
            "reference_split": "test",
            "preprocessing": {"image_channels": 3, "image_size": 64},
            "training_union_manifest_sha256": "training-union",
            "partition_file_sha256": "partition",
        }

        hashes = _validate_reference_payload(
            reference, payload, manifest, expected_count=5
        )

        self.assertEqual(hashes, set(image_hashes))


class TestGeneratedBankReuse(unittest.TestCase):
    def test_explicit_matching_bank_is_reused_without_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            output_dir = root / "output"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            output_dir.mkdir()
            checkpoint = checkpoint_dir / "step_000007.pt"
            checkpoint.write_bytes(b"checkpoint")
            sampling = {
                "method": "heun",
                "n_steps": 50,
                "heun_final_euler": False,
            }
            write_json(run_dir / "config.json", {"sampling": sampling})
            write_json(run_dir / "run.json", {"z_init_seed": 2003})
            identity = _expected_generated_bank_identity(run_dir, 7, 3)
            samples = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2)
            bank_path = root / "generated_samples.pt"
            save_torch(
                bank_path,
                {
                    "metadata": identity,
                    "samples": samples,
                    "sample_tensor_sha256": tensor_sha256(samples),
                    "sampling_summary": {
                        "checkpoint": {
                            "step": 7,
                            "sha256": identity["checkpoint_sha256"],
                            "saved_checkpoint_count": 72,
                        },
                        "timing": {
                            "measured_samples": 3,
                            "total_wall_seconds": 1.0,
                            "one_checkpoint_sampling_seconds": 0.8,
                        },
                    },
                },
            )

            with mock.patch(
                "mem_param_src.runners.checkpoint_metrics_benchmark."
                "benchmark_checkpoint_sampling"
            ) as sampler:
                loaded, summary, reused, loaded_path = _load_or_sample(
                    run_dir,
                    7,
                    3,
                    2,
                    "cpu",
                    output_dir,
                    generated_bank=bank_path,
                )

            sampler.assert_not_called()
            self.assertTrue(reused)
            self.assertTrue(torch.equal(loaded, samples))
            self.assertEqual(summary["checkpoint"]["step"], 7)
            self.assertEqual(loaded_path, bank_path.resolve())


class TestContaminatedArtifactPreflight(unittest.TestCase):
    def test_contamination_fails_before_assets_or_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            training = torch.zeros(1, 1, 2, 2)
            save_torch(run_dir / "data_bundle.pt", {"train": training})
            preprocessing = {
                "image_channels": 1,
                "image_size": 2,
                "resize_size": 2,
                "training_augmentation": False,
                "value_range": [-1.0, 1.0],
            }
            write_json(
                run_dir / "data_manifest.json",
                {"preprocessing": preprocessing},
            )
            reference = torch.ones(1, 1, 2, 2)
            payload = {
                "reference_split": "test",
                "reference_tensor_sha256s": [tensor_sha256(training[0])],
            }
            manifest = {
                "reference_split": "test",
                "preprocessing": preprocessing,
            }

            with (
                mock.patch(
                    "mem_param_src.runners.checkpoint_metrics_benchmark._load_reference",
                    return_value=(reference, payload, manifest, root / "reference.pt"),
                ),
                mock.patch(
                    "mem_param_src.runners.checkpoint_metrics_benchmark._load_assets"
                ) as load_assets,
                mock.patch(
                    "mem_param_src.runners.checkpoint_metrics_benchmark._load_or_sample"
                ) as load_or_sample,
                self.assertRaisesRegex(ValueError, "contaminated"),
            ):
                evaluate_checkpoint_metrics(
                    run_dir,
                    checkpoint_step=7,
                    reference_artifact=root / "reference.pt",
                    assets_manifest=root / "assets.json",
                    output_dir=root / "metrics",
                    sample_count=10000,
                    device="cpu",
                )

            load_assets.assert_not_called()
            load_or_sample.assert_not_called()


if __name__ == "__main__":
    unittest.main()
