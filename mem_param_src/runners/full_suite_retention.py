"""Safely remove reproducible caches from one accepted full-suite evaluation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from mem_param_src.runners.artifacts import file_sha256, read_json, write_json

RETENTION_SCHEMA_VERSION = 1
CHECKPOINT_RETENTION_SCHEMA_VERSION = 1
DEFAULT_PROTECTED_SAMPLE_STEPS = (0, 10_000, 100_000, 1_000_000)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _record(path, root):
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "nlink": int(stat.st_nlink),
        "sha256": file_sha256(path),
    }


def _step_from_root(path):
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError as error:
        raise ValueError(f"Invalid checkpoint directory: {path}") from error


def _accepted_contract(root):
    accepted_path = root / "ACCEPTED.json"
    manifest_path = root / "eval_manifest.json"
    accepted = read_json(accepted_path)
    manifest = read_json(manifest_path)
    if accepted.get("accepted") is not True:
        raise ValueError("Retention requires an accepted evaluation.")
    if accepted.get("evaluation_tier") != "paper":
        raise ValueError("Retention requires paper-tier acceptance.")
    if accepted.get("publication_ready") is not True:
        raise ValueError("Retention requires publication-ready acceptance.")
    if manifest.get("status") != "complete":
        raise ValueError("Evaluation manifest is not complete.")
    if manifest.get("evaluation_tier") != "paper":
        raise ValueError("Evaluation manifest is not paper-tier.")
    return accepted_path, manifest_path, accepted, manifest


def _checkpoint_retention_paths(checkpoint_root):
    return (
        checkpoint_root / "retention" / "DELETE_MANIFEST.json",
        checkpoint_root / "RETENTION_COMPLETE.json",
    )


def _checkpoint_candidate_paths(checkpoint_root, retain_samples):
    candidates = set()
    sample_path = checkpoint_root / "samples.pt"
    if not retain_samples:
        candidates.add(sample_path)
    feature_root = checkpoint_root / "features"
    if feature_root.is_dir():
        for path in feature_root.glob("*.pt"):
            if not path.name.endswith("_stats.pt"):
                candidates.add(path)
    return sorted(candidates)


def _checkpoint_keep_paths(checkpoint_root, candidates):
    manifest_path, completion_path = _checkpoint_retention_paths(checkpoint_root)
    excluded = set(candidates) | {manifest_path, completion_path}
    return sorted(
        path
        for path in checkpoint_root.rglob("*")
        if path.is_file() and not path.is_symlink() and path not in excluded
    )


def build_checkpoint_retention_manifest(checkpoint_root, *, retain_samples=False):
    """Record removable tensors immediately after one checkpoint completes."""
    root = Path(checkpoint_root).expanduser().resolve()
    manifest_path, completion_path = _checkpoint_retention_paths(root)
    if completion_path.exists():
        raise FileExistsError("Checkpoint caches were already retained and compacted.")
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace checkpoint retention manifest: {manifest_path}"
        )
    complete_path = root / "COMPLETE.json"
    metrics_path = root / "metrics.json"
    if not complete_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Checkpoint retention requires complete metrics.")

    candidates = _checkpoint_candidate_paths(root, bool(retain_samples))
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Expected removable cache is missing: {missing[0]}")
    required = _checkpoint_keep_paths(root, candidates)
    if complete_path not in required or metrics_path not in required:
        raise RuntimeError(
            "Checkpoint retention would not preserve completion metrics."
        )

    delete_entries = [_record(path, root) for path in candidates]
    keep_entries = [_record(path, root) for path in required]
    manifest = {
        "schema_version": CHECKPOINT_RETENTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "mode": "checkpoint_dry_run",
        "checkpoint_root": str(root),
        "checkpoint_complete_sha256": file_sha256(complete_path),
        "retain_samples": bool(retain_samples),
        "delete_count": len(delete_entries),
        "predicted_reclaimed_bytes": sum(item["size"] for item in delete_entries),
        "delete_entries": delete_entries,
        "keep_count": len(keep_entries),
        "keep_entries": keep_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def apply_checkpoint_retention_manifest(checkpoint_root, manifest_path=None):
    """Unlink the verified dense tensors for one completed checkpoint."""
    root = Path(checkpoint_root).expanduser().resolve()
    default_manifest, completion_path = _checkpoint_retention_paths(root)
    manifest_path = (
        default_manifest
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != CHECKPOINT_RETENTION_SCHEMA_VERSION:
        raise ValueError("Checkpoint retention manifest schema is unsupported.")
    if Path(manifest.get("checkpoint_root", "")).resolve() != root:
        raise ValueError("Checkpoint retention manifest targets a different root.")
    if manifest.get("checkpoint_complete_sha256") != file_sha256(
        root / "COMPLETE.json"
    ):
        raise ValueError("Checkpoint completion changed after the retention dry-run.")

    entries = manifest.get("delete_entries", [])
    if len(entries) != int(manifest.get("delete_count", -1)):
        raise ValueError("Checkpoint retention delete count differs from its entries.")
    resolved = []
    for entry in entries:
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Checkpoint retention entry escapes its root.") from error
        resolved.append((path, _validate_entry(path, entry, allow_missing=True)))
    for entry in manifest.get("keep_entries", []):
        _validate_entry(root / entry["path"], entry)

    removed_bytes = 0
    removed_count = 0
    for (path, exists), entry in zip(resolved, entries):
        if exists:
            path.unlink()
            removed_bytes += int(entry["size"])
            removed_count += 1
    for path, _ in resolved:
        if path.exists():
            raise RuntimeError(f"Checkpoint retention path still exists: {path}")

    record = {
        "schema_version": CHECKPOINT_RETENTION_SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "checkpoint_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "delete_count": len(entries),
        "removed_now_count": removed_count,
        "removed_now_bytes": removed_bytes,
        "predicted_reclaimed_bytes": int(manifest["predicted_reclaimed_bytes"]),
        "retain_samples": bool(manifest["retain_samples"]),
        "status": "complete",
    }
    write_json(completion_path, record)
    return record


def validate_retained_checkpoint(checkpoint_root):
    """Verify the immutable keep/delete identity for one compacted checkpoint."""
    root = Path(checkpoint_root).expanduser().resolve()
    _, completion_path = _checkpoint_retention_paths(root)
    record = read_json(completion_path)
    if record.get("status") != "complete":
        raise ValueError("Checkpoint retention completion record is not complete.")
    if Path(record.get("checkpoint_root", "")).resolve() != root:
        raise ValueError("Checkpoint retention completion targets a different root.")
    manifest_path = Path(record["manifest"]).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("Checkpoint retention manifest escapes its root.") from error
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError("Checkpoint retention manifest hash differs from its record.")
    manifest = read_json(manifest_path)
    if Path(manifest.get("checkpoint_root", "")).resolve() != root:
        raise ValueError("Checkpoint retention manifest targets a different root.")
    for entry in manifest["delete_entries"]:
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Checkpoint retention entry escapes its root.") from error
        if path.exists():
            raise ValueError(f"Deleted checkpoint cache still exists: {entry['path']}")
    for entry in manifest["keep_entries"]:
        _validate_entry(root / entry["path"], entry)
    return record


def ensure_checkpoint_retention(checkpoint_root, *, retain_samples=False):
    """Build, finish, or validate an idempotent checkpoint compaction."""
    root = Path(checkpoint_root).expanduser().resolve()
    manifest_path, completion_path = _checkpoint_retention_paths(root)
    if completion_path.exists():
        record = validate_retained_checkpoint(root)
    else:
        if not manifest_path.exists():
            build_checkpoint_retention_manifest(
                root,
                retain_samples=retain_samples,
            )
        record = apply_checkpoint_retention_manifest(root, manifest_path)
        validate_retained_checkpoint(root)
    if bool(record["retain_samples"]) != bool(retain_samples):
        raise ValueError("Checkpoint sample-retention policy differs from the request.")
    return record


def _checkpoint_roots(root, expected_count):
    checkpoint_roots = sorted(
        (root / "checkpoints").glob("step_*"),
        key=_step_from_root,
    )
    if len(checkpoint_roots) != int(expected_count):
        raise ValueError("Checkpoint count differs from the acceptance record.")
    return checkpoint_roots


def _candidate_paths(root, checkpoint_roots, protected_steps):
    candidates = set()
    for checkpoint_root in checkpoint_roots:
        step = _step_from_root(checkpoint_root)
        if (checkpoint_root / "RETENTION_COMPLETE.json").is_file():
            record = validate_retained_checkpoint(checkpoint_root)
            expected = step in protected_steps
            if bool(record["retain_samples"]) != expected:
                raise ValueError(
                    f"Checkpoint sample-retention policy differs at step {step}."
                )
            continue
        sample_path = checkpoint_root / "samples.pt"
        if step not in protected_steps:
            candidates.add(sample_path)
        feature_root = checkpoint_root / "features"
        if feature_root.is_dir():
            for path in feature_root.glob("*.pt"):
                if not path.name.endswith("_stats.pt"):
                    candidates.add(path)
    reference_root = root / "reference"
    if reference_root.is_dir():
        candidates.update(reference_root.rglob("*_features.pt"))
    return sorted(candidates)


def _required_keep_paths(root, checkpoint_roots, protected_steps):
    required = {
        root / "ACCEPTED.json",
        root / "eval_manifest.json",
        root / "combined" / "checkpoint_metrics.csv",
        root / "combined" / "summary.json",
        root / "plots" / "plot_manifest.json",
    }
    plot_manifest = read_json(root / "plots" / "plot_manifest.json")
    required.update(root / relative for relative in plot_manifest.get("files", {}))
    for checkpoint_root in checkpoint_roots:
        required.update(
            {
                checkpoint_root / "COMPLETE.json",
                checkpoint_root / "metrics.json",
                checkpoint_root / "sample_manifest.json",
            }
        )
        required.update((checkpoint_root / "nearest_neighbors").glob("*.pt"))
        required.update((checkpoint_root / "features").glob("*_stats.pt"))
        checkpoint_record = checkpoint_root / "RETENTION_COMPLETE.json"
        if checkpoint_record.is_file():
            record = read_json(checkpoint_record)
            required.update({checkpoint_record, Path(record["manifest"]).resolve()})
        if _step_from_root(checkpoint_root) in protected_steps:
            required.add(checkpoint_root / "samples.pt")
    reference_root = root / "reference"
    if reference_root.is_dir():
        required.update(reference_root.rglob("*neighbors.pt"))
        required.update(reference_root.rglob("*stats.pt"))
        required.update(reference_root.rglob("calibration.json"))
    return sorted(required)


def build_retention_manifest(
    evaluation_root,
    manifest_path=None,
    protected_sample_steps=DEFAULT_PROTECTED_SAMPLE_STEPS,
):
    """Write a dry-run manifest for reproducible cache files only."""
    root = Path(evaluation_root).expanduser().resolve()
    if (root / "RETENTION_COMPLETE.json").exists():
        raise FileExistsError("Evaluation caches were already retained and compacted.")
    accepted_path, eval_manifest_path, accepted, _ = _accepted_contract(root)
    checkpoint_roots = _checkpoint_roots(root, accepted["checkpoint_count"])
    available_steps = {_step_from_root(path) for path in checkpoint_roots}
    protected_steps = {
        int(step) for step in protected_sample_steps if int(step) in available_steps
    }
    protected_steps.add(max(available_steps))

    candidates = _candidate_paths(root, checkpoint_roots, protected_steps)
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Expected removable cache is missing: {missing[0]}")
    required = _required_keep_paths(root, checkpoint_roots, protected_steps)
    missing_required = [path for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(
            f"Required retained artifact is missing: {missing_required[0]}"
        )
    overlap = set(candidates).intersection(required)
    if overlap:
        raise RuntimeError(f"Retention candidate is also required: {min(overlap)}")

    delete_entries = [_record(path, root) for path in candidates]
    keep_entries = [_record(path, root) for path in required]
    manifest = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "mode": "dry_run",
        "evaluation_root": str(root),
        "accepted_sha256": file_sha256(accepted_path),
        "eval_manifest_sha256": file_sha256(eval_manifest_path),
        "protected_sample_steps": sorted(protected_steps),
        "delete_count": len(delete_entries),
        "predicted_reclaimed_bytes": sum(item["size"] for item in delete_entries),
        "delete_entries": delete_entries,
        "keep_count": len(keep_entries),
        "keep_entries": keep_entries,
    }
    if manifest_path is None:
        manifest_path = root / "retention" / "DELETE_MANIFEST.json"
    manifest_path = Path(manifest_path).expanduser().resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "Retention manifest must be inside the evaluation root."
        ) from error
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace retention manifest: {manifest_path}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def _validate_entry(path, entry, *, allow_missing=False):
    if not path.exists():
        if allow_missing:
            return False
        raise FileNotFoundError(f"Manifest path is missing: {path}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Manifest path is not a regular file: {path}")
    stat = path.stat()
    checks = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "nlink": int(stat.st_nlink),
        "sha256": file_sha256(path),
    }
    for key, actual in checks.items():
        if actual != entry[key]:
            raise ValueError(f"Manifest identity differs for {path}: {key}.")
    return True


def apply_retention_manifest(evaluation_root, manifest_path):
    """Unlink only verified paths from one reviewed retention manifest."""
    root = Path(evaluation_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != RETENTION_SCHEMA_VERSION:
        raise ValueError("Retention manifest schema version is unsupported.")
    if Path(manifest.get("evaluation_root", "")).resolve() != root:
        raise ValueError("Retention manifest targets a different evaluation root.")
    _accepted_contract(root)
    if manifest.get("accepted_sha256") != file_sha256(root / "ACCEPTED.json"):
        raise ValueError("Acceptance record changed after the retention dry-run.")
    if manifest.get("eval_manifest_sha256") != file_sha256(root / "eval_manifest.json"):
        raise ValueError("Evaluation manifest changed after the retention dry-run.")

    entries = manifest.get("delete_entries", [])
    if len(entries) != int(manifest.get("delete_count", -1)):
        raise ValueError("Retention delete count differs from its entries.")
    resolved = []
    for entry in entries:
        path = (root / entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Retention entry escapes the evaluation root.") from error
        exists = _validate_entry(path, entry, allow_missing=True)
        resolved.append((path, exists))

    for entry in manifest.get("keep_entries", []):
        _validate_entry(root / entry["path"], entry)
    removed_bytes = 0
    removed_count = 0
    for (path, exists), entry in zip(resolved, entries):
        if exists:
            path.unlink()
            removed_bytes += int(entry["size"])
            removed_count += 1
    for path, _ in resolved:
        if path.exists():
            raise RuntimeError(f"Retention path still exists after unlink: {path}")

    record = {
        "schema_version": RETENTION_SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "evaluation_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "delete_count": len(entries),
        "removed_now_count": removed_count,
        "removed_now_bytes": removed_bytes,
        "predicted_reclaimed_bytes": int(manifest["predicted_reclaimed_bytes"]),
        "protected_sample_steps": manifest["protected_sample_steps"],
        "status": "complete",
    }
    write_json(root / "RETENTION_COMPLETE.json", record)
    return record


def validate_retained_evaluation(evaluation_root):
    """Verify every retained and removed path against the saved manifest."""
    root = Path(evaluation_root).expanduser().resolve()
    record = read_json(root / "RETENTION_COMPLETE.json")
    if record.get("status") != "complete":
        raise ValueError("Retention completion record is not complete.")
    manifest_path = Path(record["manifest"]).resolve()
    if file_sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError("Retention manifest hash differs from completion record.")
    manifest = read_json(manifest_path)
    if Path(manifest["evaluation_root"]).resolve() != root:
        raise ValueError("Retention completion record targets a different root.")
    for entry in manifest["delete_entries"]:
        if (root / entry["path"]).exists():
            raise ValueError(f"Deleted cache still exists: {entry['path']}")
    for entry in manifest["keep_entries"]:
        _validate_entry(root / entry["path"], entry)
    for checkpoint_root in sorted((root / "checkpoints").glob("step_*")):
        if (checkpoint_root / "RETENTION_COMPLETE.json").is_file():
            validate_retained_checkpoint(checkpoint_root)
    return record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.evaluation_root).expanduser().resolve()
    if args.validate:
        record = validate_retained_evaluation(root)
        print(f"Validated retained evaluation: {record['evaluation_root']}")
        return 0
    if args.apply:
        if args.manifest is None:
            raise ValueError("--apply requires --manifest.")
        record = apply_retention_manifest(root, args.manifest)
        print(
            f"Removed {record['removed_now_count']} files and "
            f"{record['removed_now_bytes']} bytes."
        )
        return 0
    manifest_path, manifest = build_retention_manifest(root, args.manifest)
    print(f"Dry-run manifest: {manifest_path}")
    print(f"Delete files: {manifest['delete_count']}")
    print(f"Predicted reclaimed bytes: {manifest['predicted_reclaimed_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
