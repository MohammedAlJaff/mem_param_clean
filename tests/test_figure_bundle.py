"""Contract tests for portable paper-figure bundles."""

import csv
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from mem_param_src.runners.artifacts import (
    file_sha256,
    save_torch,
    tensor_sha256,
    write_json,
)
from mem_param_src.runners.checkpoint_metrics_benchmark import (
    _expected_generated_bank_identity,
)
from mem_param_src.runners.figure_bundle import build_figure_bundle


def _curve():
    """Return one complete held-out threshold curve."""
    return [
        {
            "heldout_quantile": alpha,
            "rho": 0.5 + alpha,
            "generated_f_mem": 0.25 + alpha,
            "calibration_rate": alpha,
            "validation_rate": alpha / 2,
        }
        for alpha in (0.005, 0.01, 0.02, 0.05)
    ]


def _neighbor_payload(count, training_count, metadata, offset=0.0):
    """Return one small valid top-two neighbor artifact."""
    d1 = torch.linspace(0.1 + offset, 0.9 + offset, count)
    d2 = torch.ones(count)
    return {
        "metadata": metadata,
        "d1": d1,
        "d2": d2,
        "r": d1 / d2,
        "nn_idx": torch.arange(count) % training_count,
        "second_nn_idx": (torch.arange(count) + 1) % training_count,
    }


def _make_evaluation(root, reference_split="test"):
    """Create one exact miniature checkpoint-gate output."""
    run_dir = root / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "step_000007.pt"
    checkpoint_path.write_bytes(b"checkpoint-seven")
    config = {
        "sampling": {
            "method": "heun",
            "n_steps": 50,
            "heun_final_euler": False,
        },
        "data": {"dataset": "celeba32_grey", "n_train": 6},
        "network": {"arch": "unet"},
        "training": {"seed": 3, "n_training_steps": 7},
        "framework": {
            "prediction_space": "x",
            "loss_space": "x",
            "loss_weighting": "constant",
        },
        "outputs": {"run_name": "fixture-run"},
        "study": {"name": "figure-bundle-test"},
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "run.json", {"z_init_seed": 2003})
    training = torch.linspace(-1.0, 1.0, 24).reshape(6, 1, 2, 2)
    save_torch(run_dir / "data_bundle.pt", {"train": training})
    write_json(run_dir / "data_manifest.json", {"schema_version": 1})

    sample_count = 40
    samples = torch.linspace(-1.2, 1.2, sample_count * 4).reshape(sample_count, 1, 2, 2)
    identity = _expected_generated_bank_identity(run_dir, 7, sample_count)
    generated_bank = root / "generated_samples.pt"
    save_torch(
        generated_bank,
        {
            "metadata": identity,
            "samples": samples,
            "sample_tensor_sha256": tensor_sha256(samples),
            "sampling_summary": {
                "checkpoint": {
                    "step": 7,
                    "sha256": identity["checkpoint_sha256"],
                    "saved_checkpoint_count": 1,
                },
                "timing": {
                    "measured_samples": sample_count,
                    "total_wall_seconds": 1.0,
                    "one_checkpoint_sampling_seconds": 0.8,
                },
            },
        },
    )

    reference_path = root / "reference.pt"
    save_torch(reference_path, {"images": torch.zeros(10, 1, 2, 2)})
    reference_hash = file_sha256(reference_path)
    reference_tensor_hash = tensor_sha256(torch.zeros(10, 1, 2, 2))
    training_hash = tensor_sha256(training)
    generated_hash = tensor_sha256(samples)
    evaluation_root = root / "evaluation"
    for space_index, space in enumerate(("raw_pixels", "sscd")):
        metric_root = evaluation_root / "metrics" / space
        common = {
            "space": space,
            "train_tensor_sha256": training_hash,
            "reference_tensor_sha256": reference_tensor_hash,
            "generated_tensor_sha256": generated_hash,
        }
        for partition, count in (
            ("generated", sample_count),
            ("calibration", 4),
            ("validation", 6),
        ):
            save_torch(
                metric_root / f"{partition}_neighbors.pt",
                _neighbor_payload(
                    count,
                    len(training),
                    {**common, "partition": partition},
                    offset=space_index * 0.01,
                ),
            )
        write_json(metric_root / "metrics.json", {"space": space})
    write_json(
        evaluation_root / "metrics" / "fid" / "metrics.json",
        {"fid_10k": 12.5},
    )

    curve = _curve()
    summary = {
        "schema_version": 2,
        "run_dir": str(run_dir.resolve()),
        "checkpoint_step": 7,
        "sample_count": sample_count,
        "training_count": len(training),
        "reference_artifact": str(reference_path.resolve()),
        "reference_artifact_sha256": reference_hash,
        "generated_bank": str(generated_bank.resolve()),
        "generated_tensor_sha256": generated_hash,
        "inception_fid": {
            "reference_split": reference_split,
            "fid_10k": 12.5,
        },
        "raw_pixel_f_mem": {
            "calibration_count": 4,
            "validation_count": 6,
            "fixed_rho": {"0.3": 0.7, "0.5": 0.4, "0.7": 0.2},
            "heldout_curve": curve,
            "headline_alpha_0.01": curve[1],
        },
        "sscd_f_mem": {
            "heldout_curve": curve,
            "headline_alpha_0.01": curve[1],
        },
    }
    write_json(evaluation_root / "summary.json", summary)
    return evaluation_root


class TestFigureBundle(unittest.TestCase):
    def test_strict_export_rejects_non_test_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _make_evaluation(root, reference_split="valid")

            with self.assertRaisesRegex(ValueError, "reference_split='test'"):
                build_figure_bundle(
                    [evaluation],
                    root / "bundle",
                    study_id="fixture",
                )

    def test_bundle_is_complete_portable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _make_evaluation(root)
            output = root / "bundle-one"

            with mock.patch(
                "mem_param_src.runners.checkpoint_metrics_benchmark."
                "benchmark_checkpoint_sampling"
            ) as sampler:
                bundle, archive = build_figure_bundle(
                    [evaluation],
                    output,
                    study_id="fixture",
                )

            sampler.assert_not_called()
            expected = (
                "manifest.json",
                "checksums.sha256",
                "tables/runs.csv",
                "tables/checkpoint_metrics.csv",
                "tables/calibration_curves.csv",
                "tables/distribution_quantiles.csv",
                "tables/examples.csv",
                "arrays/fixture-run/step_000007/raw_pixels.npz",
                "arrays/fixture-run/step_000007/sscd.npz",
                "arrays/fixture-run/reference/raw_pixels.npz",
                "arrays/fixture-run/reference/sscd.npz",
                "examples/fixture-run/step_000007/sscd/examples.npz",
                "examples/fixture-run/step_000007/sscd/examples.csv",
                "examples/fixture-run/step_000007/sscd/panel.png",
            )
            for relative in expected:
                self.assertTrue((bundle / relative).is_file(), relative)
            self.assertTrue(archive.is_file())
            self.assertFalse(list(bundle.rglob("*.pt")))

            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertTrue(manifest["publication_ready"])
            self.assertFalse(manifest["consumer_requires_model"])
            self.assertFalse(manifest["consumer_requires_torch"])
            self.assertEqual(manifest["row_counts"]["checkpoint_metrics"], 12)
            self.assertEqual(manifest["row_counts"]["calibration_curves"], 8)
            self.assertEqual(manifest["row_counts"]["distribution_quantiles"], 306)
            self.assertEqual(manifest["row_counts"]["examples"], 32)

            with (bundle / "tables/checkpoint_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 12)
            self.assertEqual(
                metric_rows[0]["checkpoint_sha256"],
                file_sha256(root / "run" / "checkpoints" / "step_000007.pt"),
            )

            neighbors = np.load(bundle / "arrays/fixture-run/step_000007/sscd.npz")
            self.assertEqual(neighbors["r"].shape, (40,))
            self.assertEqual(neighbors["nn_idx"].shape, (40,))
            examples = np.load(
                bundle / "examples/fixture-run/step_000007/sscd/examples.npz"
            )
            self.assertEqual(examples["generated"].shape, (32, 1, 2, 2))
            self.assertEqual(examples["nearest"].shape, (32, 1, 2, 2))
            with Image.open(
                bundle / "examples/fixture-run/step_000007/sscd/panel.png"
            ) as image:
                image.verify()

            for line in (bundle / "checksums.sha256").read_text().splitlines():
                digest, relative = line.split("  ", 1)
                data = (bundle / relative).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            with tarfile.open(archive, "r:gz") as handle:
                names = handle.getnames()
            self.assertIn("bundle-one/manifest.json", names)

            second, _ = build_figure_bundle(
                [evaluation],
                root / "bundle-two",
                study_id="fixture",
            )
            first_examples = np.load(
                bundle / "examples/fixture-run/step_000007/sscd/examples.npz"
            )
            second_examples = np.load(
                second / "examples/fixture-run/step_000007/sscd/examples.npz"
            )
            np.testing.assert_array_equal(
                first_examples["query_index"], second_examples["query_index"]
            )

            with self.assertRaises(FileExistsError):
                build_figure_bundle(
                    [evaluation],
                    output,
                    study_id="fixture",
                )

    def test_non_figure_checkpoint_does_not_require_generated_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _make_evaluation(root)
            config_path = root / "run" / "config.json"
            config = json.loads(config_path.read_text())
            config["training"]["n_training_steps"] = 8
            write_json(config_path, config)
            (root / "generated_samples.pt").unlink()

            bundle, _ = build_figure_bundle(
                [evaluation],
                root / "bundle",
                study_id="fixture",
            )

            manifest = json.loads((bundle / "manifest.json").read_text())
            self.assertFalse(manifest["evaluations"][0]["generated_bank_retained"])
            self.assertFalse(manifest["evaluations"][0]["examples_exported"])
            self.assertEqual(manifest["row_counts"]["examples"], 0)
            self.assertFalse((bundle / "tables/examples.csv").exists())

    def test_selected_checkpoint_requires_generated_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = _make_evaluation(root)
            config_path = root / "run" / "config.json"
            config = json.loads(config_path.read_text())
            config["training"]["n_training_steps"] = 8
            write_json(config_path, config)
            (root / "generated_samples.pt").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "must retain"):
                build_figure_bundle(
                    [evaluation],
                    root / "bundle",
                    study_id="fixture",
                    example_keys=["fixture-run:7"],
                )


if __name__ == "__main__":
    unittest.main()
