"""Tests for canonical evaluation-noise artifacts."""

import tempfile
import unittest
from pathlib import Path

import torch

from mem_param_src.runners.artifacts import (
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    write_json,
)
from mem_param_src.runners.evaluation_noise import (
    build_canonical_noise,
    load_canonical_noise,
)


class TestCanonicalEvaluationNoise(unittest.TestCase):
    def _sources(self, root):
        generator = torch.Generator(device="cpu").manual_seed(2003)
        prefix = torch.randn(
            (128, 1, 32, 32),
            generator=generator,
            dtype=torch.float32,
        )
        paths = []
        for name in ("unet", "mnist", "dit"):
            path = root / f"{name}_z_init.pt"
            save_torch(path, prefix)
            paths.append(path)
        return prefix, paths

    def test_build_records_and_reuses_one_exact_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix, sources = self._sources(root)
            artifact = root / "canonical_noise.pt"

            artifact_path, manifest_path = build_canonical_noise(
                sources,
                artifact,
                count=512,
                seed=2003,
            )
            selected, identity = load_canonical_noise(
                artifact_path,
                sample_count=256,
                expected_shape=(1, 32, 32),
            )

            payload = load_torch(artifact_path)
            manifest = read_json(manifest_path)
            self.assertEqual(tuple(payload["tensor"].shape), (512, 1, 32, 32))
            self.assertTrue(torch.equal(payload["tensor"][:128], prefix))
            self.assertTrue(torch.equal(selected, payload["tensor"][:256]))
            self.assertEqual(identity["prefix_sha256"], tensor_sha256(selected))
            self.assertEqual(payload["seed"], 2003)
            self.assertEqual(payload["source_prefix_count"], 128)
            self.assertEqual(len(payload["source_prefixes"]), 3)
            self.assertEqual(payload["shape"], [512, 1, 32, 32])
            self.assertEqual(manifest["generator"], payload["generator"])
            self.assertIn("torch_version", manifest)
            self.assertIn("generator_state_sha256", manifest)

    def test_rejects_a_source_prefix_that_differs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, sources = self._sources(root)
            changed = load_torch(sources[-1])
            changed[0, 0, 0, 0] += 1
            save_torch(sources[-1], changed)

            with self.assertRaisesRegex(ValueError, "not identical"):
                build_canonical_noise(
                    sources,
                    root / "canonical_noise.pt",
                    count=512,
                    seed=2003,
                )

    def test_load_rejects_manifest_tampering_and_short_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, sources = self._sources(root)
            artifact = root / "canonical_noise.pt"
            _, manifest_path = build_canonical_noise(
                sources,
                artifact,
                count=256,
                seed=2003,
            )

            with self.assertRaisesRegex(ValueError, "fewer rows"):
                load_canonical_noise(
                    artifact,
                    sample_count=257,
                    expected_shape=(1, 32, 32),
                )

            manifest = read_json(manifest_path)
            manifest["tensor_sha256"] = "0" * 64
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                load_canonical_noise(
                    artifact,
                    sample_count=128,
                    expected_shape=(1, 32, 32),
                )


if __name__ == "__main__":
    unittest.main()

