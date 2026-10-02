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
    write_json,
)
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


def _mnist_bundle():
    generator = torch.Generator().manual_seed(73)
    train_source_id = "fixture:mnist_train"
    reference_source_id = "fixture:mnist_test"
    return {
        "train": torch.rand(6, 1, 32, 32, generator=generator).mul(2).sub(1),
        "reference": torch.rand(4, 1, 32, 32, generator=generator).mul(2).sub(1),
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
        "train_source_catalog_sha256": source_catalog_sha256(
            train_source_id, range(6)
        ),
        "reference_source_catalog_sha256": source_catalog_sha256(
            reference_source_id, range(4)
        ),
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_split": "train",
        "reference_split": "test",
        "train_role": "train",
        "reference_role": "heldout_reference",
        "display_shape": (1, 32, 32),
    }


class _FakeModel:
    def to(self, device):
        return self

    def load_state_dict(self, state, strict=True):
        return None

    def eval(self):
        return self


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
        self.assertEqual(
            protocol["memorization_spaces"], ["sscd", "dinov2", "lpips"]
        )


class TestMNISTProtocolIntegration(unittest.TestCase):
    def _write_run(self, root, config=None):
        run_dir = root / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "evaluations").mkdir()
        config = _mnist_config() if config is None else config
        bundle = _mnist_bundle()
        write_json(run_dir / "config.json", config.to_dict())
        write_json(
            run_dir / "run.json",
            {
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
                "run_kind": "image",
                "state": "trained",
            },
        )
        save_torch(run_dir / "data_bundle.pt", bundle)
        save_data_manifest(run_dir, config, bundle, run_kind="image")
        save_torch(run_dir / "z_init.pt", torch.randn(4, 1, 32, 32))
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
            metrics = read_json(
                result / "checkpoints" / "step_000005" / "metrics.json"
            )
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
            self.assertTrue(
                metrics["f_mem_mnist_jointae32_v1"]["neighbors_reused"]
            )
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
            metrics = read_json(
                result / "checkpoints" / "step_000005" / "metrics.json"
            )
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


if __name__ == "__main__":
    unittest.main()
