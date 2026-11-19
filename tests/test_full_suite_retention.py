"""Tests for full-suite cache retention."""

import tempfile
import unittest
from pathlib import Path

from mem_param_src.runners.artifacts import write_json
from mem_param_src.runners.full_suite_retention import (
    apply_checkpoint_retention_manifest,
    apply_retention_manifest,
    build_checkpoint_retention_manifest,
    build_retention_manifest,
    ensure_checkpoint_retention,
    validate_retained_checkpoint,
    validate_retained_evaluation,
)


class TestFullSuiteRetention(unittest.TestCase):
    def _write_file(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _fixture(self, root):
        write_json(
            root / "ACCEPTED.json",
            {
                "accepted": True,
                "evaluation_tier": "paper",
                "publication_ready": True,
                "checkpoint_count": 3,
            },
        )
        write_json(
            root / "eval_manifest.json",
            {
                "status": "complete",
                "evaluation_tier": "paper",
                "publication_ready": True,
            },
        )
        self._write_file(root / "combined" / "checkpoint_metrics.csv", b"metric\n")
        write_json(root / "combined" / "summary.json", {"complete": True})
        self._write_file(root / "plots" / "quality.png", b"png")
        write_json(
            root / "plots" / "plot_manifest.json",
            {"files": {"plots/quality.png": "fixture"}},
        )
        for step in (0, 250, 1_000_000):
            checkpoint = root / "checkpoints" / f"step_{step:06d}"
            write_json(checkpoint / "COMPLETE.json", {"step": step})
            write_json(checkpoint / "metrics.json", {"step": step})
            write_json(checkpoint / "sample_manifest.json", {"step": step})
            self._write_file(checkpoint / "samples.pt", f"samples-{step}".encode())
            self._write_file(
                checkpoint / "features" / "inception_v3.pt",
                f"features-{step}".encode(),
            )
            self._write_file(
                checkpoint / "features" / "inception_v3_stats.pt",
                f"stats-{step}".encode(),
            )
            self._write_file(
                checkpoint / "nearest_neighbors" / "raw_pixels.pt",
                f"neighbors-{step}".encode(),
            )
        self._write_file(root / "reference" / "inception_v3_features.pt", b"fid-ref")
        self._write_file(
            root / "reference" / "dinov2" / "training_features.pt",
            b"train-ref",
        )
        self._write_file(
            root / "reference" / "dinov2" / "heldout_stats.pt",
            b"heldout-stats",
        )
        self._write_file(
            root / "reference" / "dinov2" / "calibration_neighbors.pt",
            b"calibration-neighbors",
        )
        write_json(
            root / "reference" / "dinov2" / "calibration.json",
            {"alpha": 0.01},
        )

    def test_dry_run_apply_and_validation_preserve_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            self._fixture(root)

            manifest_path, manifest = build_retention_manifest(root)

            self.assertEqual(manifest["delete_count"], 6)
            self.assertGreater(manifest["predicted_reclaimed_bytes"], 0)
            self.assertTrue(
                (root / "checkpoints" / "step_000250" / "samples.pt").is_file()
            )

            record = apply_retention_manifest(root, manifest_path)
            validated = validate_retained_evaluation(root)

            self.assertEqual(record["status"], "complete")
            self.assertEqual(validated["manifest_sha256"], record["manifest_sha256"])
            self.assertFalse(
                (root / "checkpoints" / "step_000250" / "samples.pt").exists()
            )
            self.assertTrue(
                (root / "checkpoints" / "step_000000" / "samples.pt").is_file()
            )
            self.assertTrue(
                (root / "checkpoints" / "step_1000000" / "samples.pt").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "checkpoints"
                    / "step_000250"
                    / "features"
                    / "inception_v3_stats.pt"
                ).is_file()
            )
            self.assertTrue((root / "combined" / "checkpoint_metrics.csv").is_file())
            self.assertTrue((root / "plots" / "quality.png").is_file())

    def test_checkpoint_retention_removes_dense_tensors_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            self._fixture(root)
            checkpoint = root / "checkpoints" / "step_000250"

            manifest_path, manifest = build_checkpoint_retention_manifest(checkpoint)
            self.assertEqual(manifest["delete_count"], 2)
            record = apply_checkpoint_retention_manifest(checkpoint, manifest_path)
            validated = validate_retained_checkpoint(checkpoint)
            repeated = ensure_checkpoint_retention(
                checkpoint,
                retain_samples=False,
            )

            self.assertEqual(record["manifest_sha256"], validated["manifest_sha256"])
            self.assertEqual(record["manifest_sha256"], repeated["manifest_sha256"])
            self.assertFalse((checkpoint / "samples.pt").exists())
            self.assertFalse((checkpoint / "features" / "inception_v3.pt").exists())
            self.assertTrue(
                (checkpoint / "features" / "inception_v3_stats.pt").is_file()
            )
            self.assertTrue(
                (checkpoint / "nearest_neighbors" / "raw_pixels.pt").is_file()
            )
            self.assertTrue((checkpoint / "metrics.json").is_file())

    def test_checkpoint_retention_validates_kept_files_before_unlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            self._fixture(root)
            checkpoint = root / "checkpoints" / "step_000250"
            manifest_path, _ = build_checkpoint_retention_manifest(checkpoint)
            write_json(checkpoint / "metrics.json", {"changed": True})

            with self.assertRaisesRegex(ValueError, "identity differs"):
                apply_checkpoint_retention_manifest(checkpoint, manifest_path)

            self.assertTrue((checkpoint / "samples.pt").is_file())
            self.assertTrue((checkpoint / "features" / "inception_v3.pt").is_file())

    def test_run_retention_accepts_stream_compacted_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            self._fixture(root)
            for step in (0, 250, 1_000_000):
                ensure_checkpoint_retention(
                    root / "checkpoints" / f"step_{step:06d}",
                    retain_samples=step in {0, 1_000_000},
                )

            manifest_path, manifest = build_retention_manifest(root)
            self.assertEqual(manifest["delete_count"], 2)
            apply_retention_manifest(root, manifest_path)
            validated = validate_retained_evaluation(root)

            self.assertEqual(validated["status"], "complete")
            self.assertFalse((root / "reference" / "inception_v3_features.pt").exists())
            self.assertTrue(
                (root / "checkpoints" / "step_000000" / "samples.pt").is_file()
            )
            self.assertFalse(
                (root / "checkpoints" / "step_000250" / "samples.pt").exists()
            )

    def test_apply_rejects_changed_candidate_before_any_unlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evaluation"
            self._fixture(root)
            manifest_path, manifest = build_retention_manifest(root)
            changed = root / manifest["delete_entries"][-1]["path"]
            changed.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "identity differs"):
                apply_retention_manifest(root, manifest_path)

            first = root / manifest["delete_entries"][0]["path"]
            self.assertTrue(first.is_file())


if __name__ == "__main__":
    unittest.main()
