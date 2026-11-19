"""Tests for dataset-aware saved-run image evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mem_param_src.config import get_base_config
from mem_param_src.datasets.common import source_catalog_sha256
from mem_param_src.runners.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    file_sha256,
    load_torch,
    read_json,
    save_data_manifest,
    save_torch,
    tensor_sha256,
    write_json,
)
from mem_param_src.runners.evaluation_noise import build_canonical_noise
from mem_param_src.runners.full_suite_validate import validate_full_suite
from mem_param_src.runners.image_protocol import (
    evaluate_image_protocol,
    resolve_checkpoint_selection,
    resolve_image_metric_protocol,
)


def _mnist_config():
    config = get_base_config()
    config.data.dataset = "mnist"
    config.data.image_size = 32
    config.data.resize_size = 32
    config.data.image_channels = 1
    config.evaluation.batch_size = 2
    config.evaluation.chunk_size = 2
    config.sampling.n_steps = 1
    config.sampling.output = "x_t_max"
    config.runtime.non_blocking_h2d = False
    return config


def _mnist_bundle(config=None):
    config = _mnist_config() if config is None else config
    generator = torch.Generator().manual_seed(73)
    train_source_id = "fixture:mnist_train"
    reference_source_id = "fixture:mnist_test"
    image_shape = (
        int(config.data.image_channels),
        int(config.data.image_size),
        int(config.data.image_size),
    )
    return {
        "train": torch.rand(6, *image_shape, generator=generator).mul(2).sub(1),
        "reference": torch.rand(4, *image_shape, generator=generator).mul(2).sub(1),
        "train_labels": torch.arange(6).remainder(10),
        "reference_labels": torch.arange(4).remainder(10),
        "train_indices": torch.arange(6),
        "reference_indices": torch.arange(4),
        "train_sample_ids": list(range(6)),
        "reference_sample_ids": list(range(4)),
        "train_sample_id_kind": "index",
        "reference_sample_id_kind": "index",
        "train_source_size": 6,
        "reference_source_size": 4,
        "train_source_catalog_sha256": source_catalog_sha256(train_source_id, range(6)),
        "reference_source_catalog_sha256": source_catalog_sha256(
            reference_source_id, range(4)
        ),
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_split": "train",
        "reference_split": "test",
        "train_role": "train",
        "reference_role": "heldout_reference",
        "display_shape": image_shape,
    }


class _FakeModel:
    def to(self, device):
        return self

    def load_state_dict(self, state, strict=True):
        return None

    def eval(self):
        return self

    def compute_loss(self, x, eps=None, t=None):
        return (x + 0.01 * eps + 0.01 * t[:, None, None, None]).square().mean()


class _FakeFeatureEncoder(torch.nn.Module):
    def __init__(self, dimension, calls):
        super().__init__()
        self.dimension = int(dimension)
        self.calls = calls

    def forward(self, images):
        self.calls.append(self.dimension)
        flat = images.flatten(1)
        summary = torch.stack(
            (flat.mean(1), flat.std(1), flat.amin(1), flat.amax(1)), dim=1
        )
        repeats = (self.dimension + summary.shape[1] - 1) // summary.shape[1]
        return summary.repeat(1, repeats)[:, : self.dimension]


class TestImageMetricRegistry(unittest.TestCase):
    def test_mnist32_and_mnist64_resolve_to_jointae(self):
        config = _mnist_config()
        for image_size in (32, 64):
            config.data.image_size = image_size
            protocol = resolve_image_metric_protocol(
                config,
                "fid,f_mem,class_fd",
                "dataset_default",
            )
            self.assertEqual(protocol["dataset_family"], "mnist")
            self.assertEqual(protocol["image_channels"], 1)
            self.assertEqual(protocol["quality_metric"], "mnist_fd_jointae32_v1")
            self.assertEqual(protocol["feature_spaces"], ["mnist_jointae32"])

    def test_mnist_rejects_natural_image_encoders(self):
        with self.assertRaisesRegex(ValueError, "natural-image spaces"):
            resolve_image_metric_protocol(_mnist_config(), "fid", "dinov2")

    def test_natural_images_resolve_fid_to_inception(self):
        config = _mnist_config()
        config.data.dataset = "celeba32_grey"

        protocol = resolve_image_metric_protocol(
            config,
            "fid",
            "dataset_default",
        )

        self.assertEqual(protocol["quality_metric"], "inception_fid")
        self.assertEqual(protocol["feature_spaces"], ["inception_v3"])

    def test_natural_memorization_requires_and_resolves_learned_spaces(self):
        config = _mnist_config()
        config.data.dataset = "celeba32_grey"

        with self.assertRaisesRegex(ValueError, "requires at least one"):
            resolve_image_metric_protocol(config, "f_mem", "dataset_default")

        protocol = resolve_image_metric_protocol(
            config,
            "fid,f_mem",
            "dataset_default,sscd,dinov2,lpips",
        )
        self.assertEqual(
            protocol["feature_spaces"],
            ["inception_v3", "sscd", "dinov2", "lpips"],
        )
        self.assertEqual(protocol["memorization_spaces"], ["sscd", "dinov2", "lpips"])

    def test_full_suite_resolves_all_dataset_specific_branches(self):
        mnist = resolve_image_metric_protocol(
            _mnist_config(), "full", "dataset_default"
        )
        self.assertTrue(mnist["full_suite"])
        self.assertIn("f_mem_raw_pixels", mnist["metrics"])
        self.assertIn("heldout_loss", mnist["metrics"])
        self.assertIn("mnist_cafd_jointae32_v1", mnist["metrics"])

        config = _mnist_config()
        config.data.dataset = "celeba32_grey"
        natural = resolve_image_metric_protocol(config, "full", "dataset_default")
        self.assertTrue(natural["full_suite"])
        self.assertEqual(
            natural["feature_spaces"],
            ["inception_v3", "sscd", "dinov2", "lpips"],
        )
        self.assertIn("dinov2_fd", natural["metrics"])
        self.assertIn("f_mem_raw_pixels", natural["metrics"])
        self.assertIn("heldout_loss", natural["metrics"])


class TestMNISTProtocolIntegration(unittest.TestCase):
    def _write_run(self, root, config=None):
        run_dir = root / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "evaluations").mkdir()
        config = _mnist_config() if config is None else config
        bundle = _mnist_bundle(config)
        write_json(run_dir / "config.json", config.to_dict())
        write_json(
            run_dir / "run.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "run_kind": "image",
                "state": "trained",
                "z_init_seed": 2003,
            },
        )
        save_torch(run_dir / "data_bundle.pt", bundle)
        save_data_manifest(run_dir, config, bundle, run_kind="image")
        noise_generator = torch.Generator(device="cpu").manual_seed(2003)
        save_torch(
            run_dir / "z_init.pt",
            torch.randn(
                (
                    4,
                    int(config.data.image_channels),
                    int(config.data.image_size),
                    int(config.data.image_size),
                ),
                generator=noise_generator,
            ),
        )
        save_torch(
            run_dir / "checkpoints" / "step_000005.pt",
            {"step": 5, "model": {}},
        )
        return run_dir

    def test_one_saved_checkpoint_runs_and_reuses_every_expensive_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(Path(temporary))

            with (
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                result = evaluate_image_protocol(
                    run_dir,
                    checkpoints="5",
                    metrics="fid,f_mem,class_fd",
                    sample_count=4,
                    protocol_id="mnist-test-v1",
                    device="cpu",
                    feature_batch_size=2,
                    nn_batch_size=2,
                )
                repeated = evaluate_image_protocol(
                    run_dir,
                    checkpoints="5",
                    metrics="fid,f_mem,class_fd",
                    sample_count=4,
                    protocol_id="mnist-test-v1",
                    device="cpu",
                    feature_batch_size=2,
                    nn_batch_size=2,
                )

            self.assertEqual(result, repeated)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(
                [step for step, _ in resolve_checkpoint_selection(run_dir, "5")],
                [5],
            )
            manifest = read_json(result / "eval_manifest.json")
            metrics = read_json(result / "checkpoints" / "step_000005" / "metrics.json")
            neighbors = load_torch(
                result
                / "checkpoints"
                / "step_000005"
                / "nearest_neighbors"
                / "mnist_jointae.pt"
            )

            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(metrics["sample_reused"])
            self.assertTrue(metrics["features_reused"])
            self.assertTrue(metrics["mnist_fd_jointae32_v1"]["distance_reused"])
            self.assertTrue(metrics["f_mem_mnist_jointae32_v1"]["neighbors_reused"])
            self.assertEqual(
                [row["alpha"] for row in metrics["f_mem_mnist_jointae32_v1"]["curve"]],
                [0.005, 0.01, 0.02, 0.05],
            )
            for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
                self.assertIn(key, neighbors)
            self.assertTrue((result / "combined" / "checkpoint_metrics.csv").is_file())
            self.assertTrue((result / "combined" / "summary.json").is_file())

    def test_fd_only_skips_memorization_reference_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(Path(temporary))

            with (
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._load_or_neighbors",
                    side_effect=AssertionError("unexpected nearest-neighbor work"),
                ),
            ):
                result = evaluate_image_protocol(
                    run_dir,
                    checkpoints="5",
                    metrics="fid",
                    sample_count=4,
                    protocol_id="mnist-fd-only-v1",
                    device="cpu",
                    feature_batch_size=2,
                )

            self.assertTrue(
                (
                    result
                    / "checkpoints"
                    / "step_000005"
                    / "features"
                    / "mnist_jointae_stats.pt"
                ).is_file()
            )

    def test_protocol_id_cannot_escape_evaluations_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._write_run(Path(temporary))

            with self.assertRaisesRegex(ValueError, "path-safe"):
                evaluate_image_protocol(
                    run_dir,
                    checkpoints="5",
                    metrics="fid",
                    sample_count=2,
                    protocol_id="../outside",
                    device="cpu",
                )

    def test_full_two_checkpoint_smoke_reuses_sampling_and_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._write_run(root)
            save_torch(
                run_dir / "checkpoints" / "step_000010.pt",
                {"step": 10, "model": {}},
            )
            noise_artifact = root / "canonical_noise.pt"
            build_canonical_noise(
                [run_dir / "z_init.pt"],
                noise_artifact,
                count=4,
                seed=2003,
            )
            output_dir = root / "smoke" / "mnist32"

            with (
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                arguments = {
                    "checkpoints": "5,10",
                    "metrics": "full",
                    "sample_count": 4,
                    "loss_count": 4,
                    "output_dir": output_dir,
                    "noise_artifact": noise_artifact,
                    "write_plots": True,
                    "device": "cpu",
                    "feature_batch_size": 2,
                    "nn_batch_size": 2,
                }
                result = evaluate_image_protocol(run_dir, **arguments)
                repeated = evaluate_image_protocol(run_dir, **arguments)
                accepted = validate_full_suite(
                    result,
                    expected_checkpoints=2,
                    sample_count=4,
                )

            self.assertEqual(result, repeated)
            self.assertTrue(accepted["accepted"])
            self.assertEqual(sample.call_count, 2)
            manifest = read_json(result / "eval_manifest.json")
            self.assertTrue(manifest["development_only"])
            self.assertFalse(manifest["publication_ready"])
            final_metrics = read_json(
                result / "checkpoints" / "step_000010" / "metrics.json"
            )
            self.assertTrue(final_metrics["sample_reused"])
            self.assertTrue(final_metrics["features_reused"])
            self.assertTrue(final_metrics["heldout_loss"]["reused"])
            self.assertTrue(final_metrics["f_mem_raw_pixels"]["neighbors_reused"])
            samples = load_torch(result / "checkpoints" / "step_000010" / "samples.pt")
            self.assertEqual(tuple(samples.shape), (4, 1, 32, 32))
            self.assertTrue((result / "plots" / "quality_trajectories.png").is_file())
            self.assertTrue(
                (result / "plots" / "nearest_neighbors_mnist_jointae32.png").is_file()
            )

    def test_full_suite_streaming_retention_compacts_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._write_run(root)
            save_torch(
                run_dir / "checkpoints" / "step_000010.pt",
                {"step": 10, "model": {}},
            )
            noise_artifact = root / "canonical_noise.pt"
            build_canonical_noise(
                [run_dir / "z_init.pt"],
                noise_artifact,
                count=4,
                seed=2003,
            )
            output_dir = root / "streaming" / "mnist32"

            with (
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                arguments = {
                    "checkpoints": "5,10",
                    "metrics": "full",
                    "sample_count": 4,
                    "loss_count": 4,
                    "output_dir": output_dir,
                    "noise_artifact": noise_artifact,
                    "write_plots": True,
                    "streaming_retention": True,
                    "device": "cpu",
                    "feature_batch_size": 2,
                    "nn_batch_size": 2,
                }
                result = evaluate_image_protocol(run_dir, **arguments)
                repeated = evaluate_image_protocol(run_dir, **arguments)
                accepted = validate_full_suite(
                    result,
                    expected_checkpoints=2,
                    sample_count=4,
                )

            self.assertEqual(result, repeated)
            self.assertEqual(sample.call_count, 2)
            self.assertTrue(accepted["streaming_retention"])
            self.assertEqual(accepted["retained_checkpoint_count"], 2)
            early = result / "checkpoints" / "step_000005"
            final = result / "checkpoints" / "step_000010"
            self.assertFalse((early / "samples.pt").exists())
            self.assertFalse((early / "features" / "mnist_jointae.pt").exists())
            self.assertTrue((early / "features" / "mnist_jointae_stats.pt").is_file())
            self.assertTrue((early / "RETENTION_COMPLETE.json").is_file())
            self.assertTrue((final / "samples.pt").is_file())
            self.assertFalse((final / "features" / "mnist_jointae.pt").exists())
            self.assertTrue((final / "RETENTION_COMPLETE.json").is_file())

    def test_full_mnist64_path_preserves_native_shape_and_reuses_all_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _mnist_config()
            config.data.image_size = 64
            config.data.resize_size = 64
            run_dir = self._write_run(root, config=config)
            noise_artifact = root / "canonical_noise_64.pt"
            build_canonical_noise(
                [run_dir / "z_init.pt"],
                noise_artifact,
                count=4,
                seed=2003,
            )
            output_dir = root / "smoke" / "mnist64"

            with (
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                arguments = {
                    "checkpoints": "5",
                    "metrics": "full",
                    "sample_count": 4,
                    "loss_count": 4,
                    "output_dir": output_dir,
                    "noise_artifact": noise_artifact,
                    "write_plots": True,
                    "device": "cpu",
                    "feature_batch_size": 2,
                    "nn_batch_size": 2,
                }
                result = evaluate_image_protocol(run_dir, **arguments)
                repeated = evaluate_image_protocol(run_dir, **arguments)
                accepted = validate_full_suite(
                    result,
                    expected_checkpoints=1,
                    sample_count=4,
                )

            self.assertEqual(result, repeated)
            self.assertTrue(accepted["accepted"])
            self.assertEqual(sample.call_count, 1)
            samples = load_torch(result / "checkpoints" / "step_000005" / "samples.pt")
            self.assertEqual(tuple(samples.shape), (4, 1, 64, 64))
            metrics = read_json(result / "checkpoints" / "step_000005" / "metrics.json")
            self.assertTrue(metrics["sample_reused"])
            self.assertTrue(metrics["features_reused"])
            self.assertTrue(metrics["heldout_loss"]["reused"])

    def test_publication_evaluation_requires_ten_thousand_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._write_run(root)

            with self.assertRaisesRegex(ValueError, "sample_count=10000"):
                evaluate_image_protocol(
                    run_dir,
                    checkpoints="5",
                    metrics="full",
                    sample_count=4,
                    output_dir=root / "paper",
                    noise_artifact=root / "unused_noise.pt",
                    write_plots=True,
                    publication_evaluation=True,
                    device="cpu",
                )


class TestNaturalProtocolIntegration(unittest.TestCase):
    def _write_run(self, root, config):
        return TestMNISTProtocolIntegration()._write_run(root, config=config)

    def _write_assets(self, root):
        asset_root = root / "encoder_assets"
        (asset_root / "weights").mkdir(parents=True)
        inception = asset_root / "weights" / "inception.pt"
        sscd = asset_root / "weights" / "sscd.pt"
        inception.write_bytes(b"fixture inception")
        sscd.write_bytes(b"fixture sscd")
        manifest = {
            "schema_version": 3,
            "torch_home": "torch_home",
            "package_versions": {"torch": torch.__version__},
            "encoders": {
                "fid_inception_v3": {
                    "path": "weights/inception.pt",
                    "sha256": file_sha256(inception),
                },
                "sscd": {
                    "path": "weights/sscd.pt",
                    "sha256": file_sha256(sscd),
                },
            },
        }
        path = asset_root / "manifest.json"
        write_json(path, manifest)
        return path

    def test_natural_features_neighbors_and_samples_are_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _mnist_config()
            config.data.dataset = "celeba32_grey"
            run_dir = self._write_run(root, config=config)
            assets_manifest = self._write_assets(root)
            calls = []
            specifications = {
                "inception_v3": {
                    "dimension": 4,
                    "layer": "fixture",
                    "preprocessing": "fixture_inception",
                },
                "sscd": {
                    "dimension": 5,
                    "layer": "fixture",
                    "preprocessing": "fixture_sscd",
                },
            }

            def encoder(space, assets_root, assets, device):
                return _FakeFeatureEncoder(specifications[space]["dimension"], calls)

            with (
                patch.dict(
                    "mem_param_src.runners.image_protocol.NATURAL_SPACE_SPECS",
                    specifications,
                ),
                patch(
                    "mem_param_src.runners.image_protocol._natural_encoder",
                    side_effect=encoder,
                ),
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                arguments = {
                    "checkpoints": "5",
                    "metrics": "fid,f_mem",
                    "feature_spaces": "dataset_default,sscd",
                    "sample_count": 4,
                    "protocol_id": "natural-test-v1",
                    "assets_manifest": assets_manifest,
                    "device": "cpu",
                    "feature_batch_size": 16,
                    "nn_batch_size": 4,
                }
                result = evaluate_image_protocol(run_dir, **arguments)
                repeated = evaluate_image_protocol(run_dir, **arguments)

            self.assertEqual(result, repeated)
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(len(calls), 5)
            metrics = read_json(result / "checkpoints" / "step_000005" / "metrics.json")
            self.assertTrue(metrics["inception_fid"]["features_reused"])
            self.assertTrue(metrics["inception_fid"]["distance_reused"])
            self.assertTrue(metrics["f_mem_sscd"]["features_reused"])
            self.assertTrue(metrics["f_mem_sscd"]["neighbors_reused"])
            self.assertTrue(
                (
                    result
                    / "checkpoints"
                    / "step_000005"
                    / "features"
                    / "sscd.COMPLETE.json"
                ).is_file()
            )

    def test_full_two_checkpoint_natural_smoke_reuses_all_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _mnist_config()
            config.data.dataset = "celeba64_rgb"
            config.data.image_size = 64
            config.data.resize_size = 64
            config.data.image_channels = 3
            run_dir = self._write_run(root, config=config)
            save_torch(
                run_dir / "checkpoints" / "step_000010.pt",
                {"step": 10, "model": {}},
            )
            noise_artifact = root / "canonical_noise.pt"
            build_canonical_noise(
                [run_dir / "z_init.pt"],
                noise_artifact,
                count=4,
                seed=2003,
            )
            reference = load_torch(run_dir / "data_bundle.pt")["reference"]
            reference_manifest = {
                "artifact_sha256": "reference-artifact",
                "reference_tensor_sha256": tensor_sha256(reference),
            }
            specifications = {
                name: {
                    "dimension": dimension,
                    "layer": "fixture",
                    "preprocessing": f"fixture_{name}",
                }
                for name, dimension in {
                    "inception_v3": 4,
                    "sscd": 5,
                    "dinov2": 4,
                    "lpips": 6,
                }.items()
            }
            assets = {
                "package_versions": {"torch": torch.__version__},
                "encoders": {
                    "fid_inception_v3": {"sha256": "fid"},
                    "sscd": {"sha256": "sscd"},
                    "dinov2_weights": {"sha256": "dino"},
                    "lpips_calibration": {"sha256": "lpips"},
                    "lpips_alexnet": {"sha256": "alex"},
                },
            }
            calls = []

            def encoder(space, assets_root, manifest, device):
                del assets_root, manifest, device
                return _FakeFeatureEncoder(specifications[space]["dimension"], calls)

            with (
                patch.dict(
                    "mem_param_src.runners.image_protocol.NATURAL_SPACE_SPECS",
                    specifications,
                ),
                patch(
                    "mem_param_src.runners.image_protocol._load_encoder_assets",
                    return_value=(root, assets, {"sha256": "assets"}),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._natural_encoder",
                    side_effect=encoder,
                ),
                patch(
                    "mem_param_src.runners.image_protocol._load_reference",
                    return_value=(
                        reference,
                        {},
                        reference_manifest,
                        root / "reference.pt",
                    ),
                ),
                patch(
                    "mem_param_src.runners.image_protocol."
                    "_validate_reference_compatibility"
                ),
                patch(
                    "mem_param_src.runners.image_protocol.GDFM",
                    return_value=_FakeModel(),
                ),
                patch(
                    "mem_param_src.runners.image_protocol._sample_checkpoint",
                    side_effect=lambda model, noise, config, device: noise.clone(),
                ) as sample,
            ):
                arguments = {
                    "checkpoints": "5,10",
                    "metrics": "full",
                    "sample_count": 4,
                    "loss_count": 4,
                    "output_dir": root / "smoke" / "unet32",
                    "noise_artifact": noise_artifact,
                    "reference_artifact": root / "reference.pt",
                    "assets_manifest": root / "assets.json",
                    "write_plots": True,
                    "device": "cpu",
                    "feature_batch_size": 16,
                    "nn_batch_size": 4,
                }
                result = evaluate_image_protocol(run_dir, **arguments)
                call_count = len(calls)
                repeated = evaluate_image_protocol(run_dir, **arguments)
                accepted = validate_full_suite(
                    result,
                    expected_checkpoints=2,
                    sample_count=4,
                )

            self.assertEqual(result, repeated)
            self.assertTrue(accepted["accepted"])
            self.assertEqual(sample.call_count, 2)
            self.assertEqual(call_count, 15)
            self.assertEqual(len(calls), call_count)
            final_metrics = read_json(
                result / "checkpoints" / "step_000010" / "metrics.json"
            )
            self.assertTrue(final_metrics["dinov2_fd"]["features_reused"])
            self.assertTrue(final_metrics["dinov2_fd"]["distance_reused"])
            self.assertTrue(final_metrics["f_mem_lpips"]["features_reused"])
            self.assertTrue(final_metrics["heldout_loss"]["reused"])
            self.assertTrue(
                (result / "plots" / "memorization_raw_pixels.png").is_file()
            )
            self.assertTrue(
                (result / "plots" / "nearest_neighbors_dinov2.png").is_file()
            )

            manifest_path = result / "eval_manifest.json"
            manifest = read_json(manifest_path)
            manifest.update(
                {
                    "evaluation_tier": "paper",
                    "development_only": False,
                    "publication_ready": True,
                }
            )
            write_json(manifest_path, manifest)
            paper_acceptance = validate_full_suite(
                result,
                expected_checkpoints=2,
                sample_count=4,
                evaluation_tier="paper",
            )
            self.assertTrue(paper_acceptance["publication_ready"])


if __name__ == "__main__":
    unittest.main()
