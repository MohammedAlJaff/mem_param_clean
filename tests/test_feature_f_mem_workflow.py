"""Acceptance tests for the feature-space f_mem job workflow."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "jobs"
    / "berzelius"
    / "eps_recovery_1m"
    / "feature_f_mem"
)
sys.path.insert(0, str(SCRIPT_DIR))

import combine_feature_f_mem as combine  # noqa: E402
import evaluate_feature_f_mem as evaluate  # noqa: E402


class TestFeatureFMemWorkflow(unittest.TestCase):
    def _write_seed0_run(self, root, directory, seed, prediction, loss):
        run_dir = root / directory
        (run_dir / "checkpoints").mkdir(parents=True)
        config = {
            "data": {"dataset": "celeba32_grey", "n_train": 256, "image_size": 32},
            "network": {"arch": "unet", "base_channels": 32},
            "training": {
                "seed": seed,
                "optimizer": "adam",
                "lr": 1.0e-4,
                "batch_cap": 256,
            },
            "framework": {
                "prediction_space": prediction,
                "loss_space": loss,
                "loss_weighting": "unit",
            },
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "data_bundle.pt").write_bytes(b"same-seed-zero-bundle")
        bundle_hash = evaluate.file_sha256(run_dir / "data_bundle.pt")
        (run_dir / "run.json").write_text(
            json.dumps({"data_bundle_sha256": bundle_hash})
        )
        for step in evaluate.CHECKPOINT_STEPS:
            (run_dir / "checkpoints" / f"step_{step:06d}.pt").write_bytes(b"checkpoint")

    def test_seed0_manifest_selects_matched_eps_v_x(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_seed0_run(root, "1_eps_direct_full_heun_s0", 0, "eps", "eps")
            self._write_seed0_run(root, "6_vv_reference_s0", 0, "v", "v")
            self._write_seed0_run(root, "7_xx_reference_s0", 0, "x", "x")

            manifest = evaluate.build_manifest(root, scope="seed0")

        self.assertEqual(len(manifest["tasks"]), 3)
        self.assertEqual(
            [task["parameterization"] for task in manifest["tasks"]],
            ["eps", "v", "x"],
        )
        self.assertEqual(manifest["checkpoint_steps"], list(evaluate.CHECKPOINT_STEPS))
        self.assertEqual(manifest["sampling"], evaluate.SAMPLING_SETTINGS)

    def test_pixel_feature_cache_rejects_metadata_or_tensor_changes(self):
        from mem_param_src.runners.artifacts import load_torch, save_torch

        images = torch.randn(4, 1, 32, 32)
        assets = {"encoders": {}, "package_versions": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.pt"
            first = evaluate.load_or_encode(
                path,
                images,
                space="full_pixels",
                source_hash="source-a",
                assets_root=Path(directory),
                assets=assets,
                device="cpu",
                batch_size=2,
            )
            second = evaluate.load_or_encode(
                path,
                images,
                space="full_pixels",
                source_hash="source-a",
                assets_root=Path(directory),
                assets=assets,
                device="cpu",
                batch_size=1,
            )
            self.assertTrue(torch.equal(first, second))
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                evaluate.load_or_encode(
                    path,
                    images,
                    space="full_pixels",
                    source_hash="source-b",
                    assets_root=Path(directory),
                    assets=assets,
                    device="cpu",
                    batch_size=1,
                )
            payload = load_torch(path)
            payload["features"][0, 0] += 1.0
            save_torch(path, payload)
            with self.assertRaisesRegex(RuntimeError, "tensor hash mismatch"):
                evaluate.load_or_encode(
                    path,
                    images,
                    space="full_pixels",
                    source_hash="source-a",
                    assets_root=Path(directory),
                    assets=assets,
                    device="cpu",
                    batch_size=1,
                )

    def test_combiner_requires_27_seed0_learned_records(self):
        manifest = {"tasks": [{}, {}, {}]}
        rows = []
        curve = [
            {
                "heldout_quantile": quantile,
                "rho": quantile,
                "calibration_rate": quantile,
                "validation_rate": quantile,
                "generated_f_mem": quantile,
            }
            for quantile in evaluate.QUANTILES
        ]
        for parameterization in ("eps", "v", "x"):
            for step in evaluate.CHECKPOINT_STEPS:
                for space in evaluate.PRIMARY_SPACES:
                    rows.append(
                        {
                            "space": space,
                            "reference_artifact_sha256": "reference",
                            "sample_count": 10_000,
                            "calibration_count": 1024,
                            "validation_count": 8976,
                            "raw_sample_stats": {"finite_fraction": 1.0},
                            "fid_10k": 10.0,
                            "heldout_curve": curve,
                        }
                    )
        for parameterization in ("eps", "v", "x"):
            for space in evaluate.SENSITIVITY_SPACES:
                rows.append(
                    {
                        "space": space,
                        "reference_artifact_sha256": "reference",
                        "sample_count": 10_000,
                        "calibration_count": 1024,
                        "validation_count": 8976,
                        "raw_sample_stats": {"finite_fraction": 1.0},
                        "fid_10k": 10.0,
                        "heldout_curve": curve,
                    }
                )

        summary = combine.validate_rows(rows, manifest)

        self.assertEqual(summary["learned_space_records"], 27)
        self.assertEqual(summary["primary_records"], 36)
        self.assertEqual(summary["sensitivity_records"], 9)


if __name__ == "__main__":
    unittest.main()
