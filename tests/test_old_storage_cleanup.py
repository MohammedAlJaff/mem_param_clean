"""Tests for the old-repository storage cleanup audit."""

import csv
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from mem_param_src.runners.old_storage_cleanup import (
    apply_audit,
    create_audit,
)


class TestOldStorageCleanup(unittest.TestCase):
    def _write(self, path, payload=b"checkpoint"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _read_manifest(self, path):
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _capacity_tree(self, old_root):
        main_name = "mem_gen_boundary_capacity_20260427"
        backup_name = f"{main_name}_BACKUP_20260501_023904"
        run_id = "capacity_run_s0"
        main_attempts = (
            old_root
            / "outputs/job_sets"
            / main_name
            / "runs"
            / run_id
            / "seeds/seed_0000/attempts"
        )
        source = main_attempts / "20260428_000000/checkpoints"
        done = main_attempts / "DONE_20260503_000000/checkpoints"
        paths = []
        for index, step in enumerate(range(0, 100, 10)):
            paths.append(
                self._write(source / f"step_{step:06d}.pt", f"s{index}".encode())
            )
        paths.append(self._write(done / "step_000100.pt", b"s10"))
        paths.append(self._write(done / "step_000110.pt", b"s11"))
        self._write(done.parent / "status.json", b'{"state":"trained"}')

        backup_base = (
            old_root
            / "outputs/job_sets"
            / backup_name
            / "runs"
            / run_id
            / "seeds/seed_0000/attempts/20260428_000000/checkpoints"
        )
        for path in paths[:10]:
            destination = backup_base / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
        return paths

    def _cancelled_tree(self, old_root):
        run = (
            old_root
            / "prototyping/intr_amb_experiments/outputs/job_sets"
            / "intr_amb_k_edge_test_20260606/runs/D100_x_s0_j8825"
        )
        checkpoint = self._write(run / "checkpoints/step_000000.pt")
        event = self._write(
            run / "logs/wandb/wandb/run-20260608-cancelled/run-cancelled.wandb",
            b"wandb",
        )
        metadata = self._write(run / "metrics/final.json", b"{}")
        return checkpoint, event, metadata

    def _temporary_copy(self, old_root):
        source = self._write(
            old_root
            / "outputs/job_sets/mem_gen_boundary_20260427/runs/source_run"
            / "seeds/seed_0000/attempts/DONE_20260427/checkpoints/step_000010.pt",
            b"same-copy",
        )
        copied_root = (
            old_root / "tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams"
        )
        copied_attempt = (
            copied_root
            / "mem_gen_boundary_20260427/runs/source_run"
            / "seeds/seed_0000/attempts/DONE_20260427"
        )
        copied = self._write(
            copied_attempt / "checkpoints/step_000010.pt",
            b"same-copy",
        )
        manifest = {
            "attempts": [
                {
                    "source_attempt_dir": str(source.parent.parent).replace(
                        str(old_root),
                        "/historical/path/mem_par",
                    ),
                    "copied_attempt_dir": str(copied_attempt).replace(
                        str(old_root),
                        "/historical/path/mem_par",
                    ),
                }
            ]
        }
        (copied_root / "SANDBOX_MANIFEST.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        return source, copied

    def _intrinsic_ambient_source(self, old_root):
        source_root = old_root / "prototyping/intr_amb_experiments"
        source = self._write(source_root / "intr_amb_core.py", b"source-code")
        bytecode = self._write(
            source_root / "__pycache__/intr_amb_core.cpython-310.pyc",
            b"bytecode",
        )
        output = self._write(source_root / "outputs/geometry/figure.png", b"plot")
        return source, bytecode, output

    def test_dry_run_merges_attempts_and_changes_no_old_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old_mem_par"
            clean_root = root / "mem_param_clean"
            clean_root.mkdir()
            capacity_paths = self._capacity_tree(old_root)
            cancelled_checkpoint, cancelled_event, metadata = self._cancelled_tree(
                old_root
            )
            source, copied = self._temporary_copy(old_root)
            old_source, old_bytecode, old_source_output = (
                self._intrinsic_ambient_source(old_root)
            )
            eval_result = self._write(
                old_root / "outputs/evals/historical/all_checkpoint_metrics.csv",
                b"step,metric\n",
            )
            audit_dir = root / "audit"

            summary = create_audit(old_root, clean_root, audit_dir)
            keep_rows = self._read_manifest(audit_dir / "KEEP_MANIFEST.tsv")
            delete_rows = self._read_manifest(audit_dir / "DELETE_MANIFEST.tsv")
            keep_paths = {row["path"] for row in keep_rows}
            delete_paths = {row["path"] for row in delete_rows}

            self.assertTrue(all(path.exists() for path in capacity_paths))
            self.assertTrue(cancelled_checkpoint.exists())
            self.assertTrue(cancelled_event.exists())
            self.assertTrue(old_source.exists())
            self.assertTrue(old_bytecode.exists())
            self.assertTrue(old_source_output.exists())
            self.assertTrue(metadata.exists())
            self.assertTrue(source.exists())
            self.assertTrue(copied.exists())
            self.assertTrue(eval_result.exists())
            self.assertEqual(summary["mode"], "dry_run")
            self.assertFalse(summary["deletion_applied"])
            self.assertGreater(summary["predicted_reclaimable_bytes"], 0)
            self.assertIn(
                cancelled_checkpoint.relative_to(old_root).as_posix(),
                delete_paths,
            )
            self.assertIn(
                cancelled_event.relative_to(old_root).as_posix(),
                delete_paths,
            )
            self.assertIn(metadata.relative_to(old_root).as_posix(), keep_paths)
            self.assertIn(copied.relative_to(old_root).as_posix(), delete_paths)
            self.assertIn(old_source.relative_to(old_root).as_posix(), delete_paths)
            self.assertIn(old_bytecode.relative_to(old_root).as_posix(), delete_paths)
            self.assertNotIn(
                old_source_output.relative_to(old_root).as_posix(),
                keep_paths | delete_paths,
            )
            self.assertNotIn(eval_result.relative_to(old_root).as_posix(), keep_paths)
            self.assertNotIn(eval_result.relative_to(old_root).as_posix(), delete_paths)

            kept_capacity = {
                Path(row["path"]).name
                for row in keep_rows
                if "capacity_run_s0" in row["path"] and "/checkpoints/" in row["path"]
            }
            archive_path = audit_dir / "INTR_AMB_SOURCE_SNAPSHOT.tar.gz"
            self.assertEqual(
                summary["intrinsic_ambient_source_archive"]["files"],
                2,
            )
            with tarfile.open(archive_path, mode="r:gz") as archive:
                self.assertEqual(
                    set(archive.getnames()),
                    {
                        old_source.relative_to(old_root).as_posix(),
                        old_bytecode.relative_to(old_root).as_posix(),
                    },
                )
            self.assertEqual(
                kept_capacity,
                {"step_000000.pt", "step_000090.pt", "step_000110.pt"},
            )

    def test_apply_validates_identity_and_deletes_only_selected_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old_mem_par"
            clean_root = root / "mem_param_clean"
            clean_root.mkdir()
            checkpoint, event, metadata = self._cancelled_tree(old_root)
            audit_dir = root / "audit"
            create_audit(old_root, clean_root, audit_dir)

            result = apply_audit(
                old_root,
                audit_dir,
                confirm_root=old_root,
                groups=["cancelled_j8825"],
            )

            self.assertEqual(result["deleted_files"], 2)
            self.assertFalse(checkpoint.exists())
            self.assertFalse(event.exists())
            self.assertTrue(metadata.exists())
            self.assertTrue((audit_dir / "APPLIED.cancelled_j8825.json").is_file())
            self.assertTrue(
                (audit_dir / "APPLY_PROGRESS.cancelled_j8825.jsonl").is_file()
            )

    def test_apply_rejects_a_changed_target_before_deleting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old_mem_par"
            clean_root = root / "mem_param_clean"
            clean_root.mkdir()
            checkpoint, event, metadata = self._cancelled_tree(old_root)
            audit_dir = root / "audit"
            create_audit(old_root, clean_root, audit_dir)
            event.write_bytes(b"changed-after-audit")

            with self.assertRaisesRegex(ValueError, "changed"):
                apply_audit(
                    old_root,
                    audit_dir,
                    confirm_root=old_root,
                    groups=["cancelled_j8825"],
                )

            self.assertTrue(checkpoint.exists())
            self.assertTrue(event.exists())
            self.assertTrue(metadata.exists())

    def test_source_apply_requires_the_preserved_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root = root / "old_mem_par"
            clean_root = root / "mem_param_clean"
            clean_root.mkdir()
            source, bytecode, output = self._intrinsic_ambient_source(old_root)
            audit_dir = root / "audit"
            create_audit(old_root, clean_root, audit_dir)
            archive_path = audit_dir / "INTR_AMB_SOURCE_SNAPSHOT.tar.gz"
            archive_path.write_bytes(b"changed archive")

            with self.assertRaisesRegex(ValueError, "archive hash changed"):
                apply_audit(
                    old_root,
                    audit_dir,
                    confirm_root=old_root,
                    groups=["obsolete_intrinsic_ambient_source"],
                )

            self.assertTrue(source.exists())
            self.assertTrue(bytecode.exists())
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
