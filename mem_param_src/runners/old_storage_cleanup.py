"""Audit and apply explicit cleanup manifests for the old repository."""

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

AUDIT_SCHEMA_VERSION = 1
CHECKPOINT_PATTERN = re.compile(r"^step_(\d+)\.pt$")
MANIFEST_FIELDS = (
    "action",
    "group",
    "reason",
    "path",
    "size_bytes",
    "allocated_bytes",
    "mtime_ns",
    "mtime_utc",
    "device",
    "inode",
    "hardlink_count",
)
SCAN_ROOTS = (
    ("image_job_sets", Path("outputs/job_sets")),
    (
        "prototype_job_sets",
        Path("prototyping/intr_amb_experiments/outputs/job_sets"),
    ),
    ("temporary_eval_copy", Path("tmp/eval_prototype_runs")),
)
INTR_AMB_SOURCE_ROOT = Path("prototyping/intr_amb_experiments")
INTR_AMB_SOURCE_ARCHIVE = "INTR_AMB_SOURCE_SNAPSHOT.tar.gz"
CAPACITY_JOB_SET = "mem_gen_boundary_capacity_20260427"
CAPACITY_BACKUP_JOB_SET = "mem_gen_boundary_capacity_20260427_BACKUP_20260501_023904"
CANCELLED_LAUNCH_SUFFIX = "_j8825"
TEMP_COPY_MANIFEST = Path(
    "tmp/eval_prototype_runs/celeba_adam_N16384_s0_allparams/SANDBOX_MANIFEST.json"
)
PROTECTED_RUN_STEPS = {
    (
        "mem_gen_boundary_20260427_celeba32_N256_B256_"
        "eps_eps_steps1000000_s0-bd48e9dab3"
    ): {25000},
}


def _utc_timestamp():
    """Return one stable UTC timestamp for saved audit records."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mtime_utc(mtime_ns):
    """Format a nanosecond modification time in UTC."""
    seconds = mtime_ns / 1_000_000_000
    return datetime.fromtimestamp(seconds, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _file_sha256(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    """Write JSON through a temporary file in the destination directory."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_tsv(path, rows):
    """Write a tab-separated manifest atomically."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in MANIFEST_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _record_for_file(path, old_root, scan_name):
    """Build one immutable filesystem record without following symlinks."""
    file_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode):
        return None
    relative = path.relative_to(old_root)
    relative_text = relative.as_posix()
    if "\t" in relative_text or "\n" in relative_text:
        raise ValueError(f"Unsupported whitespace in path: {path}")
    return {
        "path_object": path,
        "relative": relative,
        "path": relative_text,
        "scan_name": scan_name,
        "size_bytes": int(file_stat.st_size),
        "allocated_bytes": int(file_stat.st_blocks * 512),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "mtime_utc": _mtime_utc(file_stat.st_mtime_ns),
        "device": int(file_stat.st_dev),
        "inode": int(file_stat.st_ino),
        "hardlink_count": int(file_stat.st_nlink),
        "action": "keep",
        "group": "preserved",
        "reason": "preserve_unmanaged_file",
    }


def _scan_files(old_root):
    """Scan only the old output roots selected by the cleanup plan."""
    records = []
    roots = []
    for scan_name, relative_root in SCAN_ROOTS:
        root = old_root / relative_root
        if not root.is_dir():
            continue
        roots.append(relative_root.as_posix())
        for directory, _, filenames in os.walk(root, followlinks=False):
            directory = Path(directory)
            for filename in filenames:
                record = _record_for_file(
                    directory / filename,
                    old_root,
                    scan_name,
                )
                if record is not None:
                    records.append(record)
    source_root = old_root / INTR_AMB_SOURCE_ROOT
    if source_root.is_dir():
        roots.append(f"{INTR_AMB_SOURCE_ROOT.as_posix()} (excluding outputs)")
        for directory, dirnames, filenames in os.walk(
            source_root,
            followlinks=False,
        ):
            dirnames[:] = [name for name in dirnames if name != "outputs"]
            directory = Path(directory)
            for filename in filenames:
                record = _record_for_file(
                    directory / filename,
                    old_root,
                    "intrinsic_ambient_source",
                )
                if record is not None:
                    records.append(record)
    records.sort(key=lambda row: row["path"])
    return records, roots


def _path_part_after(parts, name):
    """Return the path component after the first matching component."""
    try:
        index = parts.index(name)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _checkpoint_identity(record):
    """Return grouping fields for one checkpoint file."""
    relative = record["relative"]
    match = CHECKPOINT_PATTERN.fullmatch(relative.name)
    if match is None or relative.parent.name != "checkpoints":
        return None
    parts = relative.parts
    run_id = _path_part_after(parts, "runs")
    if run_id is None:
        return None
    job_set = _path_part_after(parts, "job_sets")
    if job_set == CAPACITY_BACKUP_JOB_SET:
        normalized_job_set = CAPACITY_JOB_SET
    else:
        normalized_job_set = job_set
    attempt = _path_part_after(parts, "attempts")
    return {
        "step": int(match.group(1)),
        "run_id": run_id,
        "job_set": job_set,
        "normalized_job_set": normalized_job_set,
        "attempt": attempt,
        "is_backup": job_set == CAPACITY_BACKUP_JOB_SET,
        "is_cancelled_launch": run_id.endswith(CANCELLED_LAUNCH_SUFFIX),
        "is_temporary_copy": record["scan_name"] == "temporary_eval_copy",
    }


def _checkpoint_group_key(identity):
    """Return the scientific-run key used across attempts and backups."""
    return (
        identity["scan_name"],
        identity["normalized_job_set"],
        identity["run_id"],
    )


def _checkpoint_preference(item):
    """Sort the preferred copy of a checkpoint before duplicates."""
    record, identity = item
    attempt = identity["attempt"] or ""
    return (
        int(identity["is_backup"]),
        -int(attempt.startswith("DONE_")),
        -record["mtime_ns"],
        record["path"],
    )


def _retained_steps(steps):
    """Keep step zero, each tenth saved checkpoint, and the final step."""
    steps = sorted(set(steps))
    if not steps:
        return set()
    retained = {steps[0], steps[-1]}
    retained.update(steps[index] for index in range(9, len(steps), 10))
    return retained


def _mark(record, action, group, reason):
    """Set one manifest decision."""
    record["action"] = action
    record["group"] = group
    record["reason"] = reason


def _classify_intrinsic_ambient_source(records):
    """Select the obsolete live source files without selecting run output."""
    for record in records:
        if record["scan_name"] == "intrinsic_ambient_source":
            _mark(
                record,
                "delete",
                "obsolete_intrinsic_ambient_source",
                "obsolete_intrinsic_ambient_source_file",
            )


def _write_intrinsic_ambient_source_archive(records, output_dir):
    """Preserve the exact source files before they become deletion candidates."""
    source_records = [
        record
        for record in records
        if record["scan_name"] == "intrinsic_ambient_source"
    ]
    if not source_records:
        return None
    archive_path = output_dir / INTR_AMB_SOURCE_ARCHIVE
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for record in source_records:
            archive.add(
                record["path_object"],
                arcname=record["path"],
                recursive=False,
            )
    return {
        "path": archive_path.name,
        "sha256": _file_sha256(archive_path),
        "files": len(source_records),
        "logical_bytes": sum(record["size_bytes"] for record in source_records),
    }


def _preservation_reason(record):
    """Explain why a non-target file remains."""
    parts = set(record["relative"].parts)
    if "metrics" in parts:
        return "preserve_metrics"
    if "figures" in parts or record["relative"].suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".svg",
    }:
        return "preserve_figures"
    if "artifacts" in parts:
        return "preserve_reproducibility_artifact"
    if "logs" in parts or "slurm" in parts:
        return "preserve_logs"
    if record["relative"].suffix.lower() in {
        ".json",
        ".csv",
        ".tsv",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".sh",
        ".py",
        ".ipynb",
    }:
        return "preserve_metadata_or_source"
    return "preserve_unmanaged_file"


def _classify_checkpoint_groups(records, protected_paths):
    """Apply the checkpoint retention policy across attempts and backups."""
    groups = defaultdict(list)
    identities = {}
    for record in records:
        identity = _checkpoint_identity(record)
        if identity is None or identity["is_temporary_copy"]:
            continue
        identity["scan_name"] = record["scan_name"]
        identities[record["path"]] = identity
        groups[_checkpoint_group_key(identity)].append((record, identity))

    for items in groups.values():
        first_identity = items[0][1]
        if first_identity["is_cancelled_launch"]:
            for record, _ in items:
                if record["path"] in protected_paths:
                    _mark(
                        record,
                        "keep",
                        "protected",
                        "protected_explicit_checkpoint",
                    )
                else:
                    _mark(
                        record,
                        "delete",
                        "cancelled_j8825",
                        "cancelled_launch_checkpoint",
                    )
            continue

        attempts = {identity["attempt"] for _, identity in items}
        uses_attempts = any(attempt is not None for attempt in attempts)
        complete = not uses_attempts or any(
            attempt and attempt.startswith("DONE_") for attempt in attempts
        )
        if not complete:
            for record, _ in items:
                if record["path"] in protected_paths:
                    _mark(
                        record,
                        "keep",
                        "protected",
                        "protected_explicit_checkpoint",
                    )
                else:
                    _mark(
                        record,
                        "delete",
                        "failed_or_incomplete_runs",
                        "incomplete_run_checkpoint",
                    )
            continue

        by_step = defaultdict(list)
        for item in items:
            by_step[item[1]["step"]].append(item)
        retained = _retained_steps(by_step)
        retained.update(PROTECTED_RUN_STEPS.get(first_identity["run_id"], set()))
        for step, step_items in by_step.items():
            ordered = sorted(step_items, key=_checkpoint_preference)
            protected_items = [
                item for item in ordered if item[0]["path"] in protected_paths
            ]
            chosen = protected_items[0] if protected_items else ordered[0]
            for item in ordered:
                record, identity = item
                if item == chosen and step in retained:
                    reason = "retained_checkpoint_schedule"
                    if step in PROTECTED_RUN_STEPS.get(identity["run_id"], set()):
                        reason = "protected_documented_checkpoint"
                    if record["path"] in protected_paths:
                        reason = "protected_explicit_checkpoint"
                    _mark(record, "keep", "retained_checkpoints", reason)
                elif identity["is_backup"]:
                    _mark(
                        record,
                        "delete",
                        "capacity_checkpoints",
                        "capacity_backup_checkpoint_link",
                    )
                elif step in retained:
                    _mark(
                        record,
                        "delete",
                        _checkpoint_cleanup_group(identity),
                        "duplicate_checkpoint_step",
                    )
                else:
                    _mark(
                        record,
                        "delete",
                        _checkpoint_cleanup_group(identity),
                        "checkpoint_retention_thinning",
                    )


def _checkpoint_cleanup_group(identity):
    """Return the cleanup group for one checkpoint."""
    if identity["normalized_job_set"] == CAPACITY_JOB_SET:
        return "capacity_checkpoints"
    if identity["scan_name"] == "prototype_job_sets":
        return "prototype_checkpoints"
    return "old_image_checkpoints"


def _remap_old_path(raw_path, old_root):
    """Map the historical mem_par absolute prefix to the present old root."""
    raw = str(raw_path)
    for marker in ("/outputs/", "/tmp/"):
        if marker in raw:
            suffix = raw.split(marker, 1)[1]
            return old_root / marker.strip("/") / suffix
    return Path(raw).expanduser().resolve()


def _classify_verified_temp_copies(records, old_root):
    """Delete copied sandbox checkpoints only after a SHA-256 match."""
    manifest_path = old_root / TEMP_COPY_MANIFEST
    if not manifest_path.is_file():
        return {"matched_files": 0, "matched_bytes": 0, "manifest": None}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    copy_pairs = []
    for item in payload.get("attempts", []):
        source = _remap_old_path(item["source_attempt_dir"], old_root)
        copied = _remap_old_path(item["copied_attempt_dir"], old_root)
        copy_pairs.append((copied.resolve(), source.resolve()))

    matched_files = 0
    matched_bytes = 0
    hash_cache = {}
    for record in records:
        identity = _checkpoint_identity(record)
        if identity is None or not identity["is_temporary_copy"]:
            continue
        copied_path = record["path_object"].resolve()
        source_path = None
        for copied_root, source_root in copy_pairs:
            try:
                suffix = copied_path.relative_to(copied_root)
            except ValueError:
                continue
            source_path = source_root / suffix
            break
        if source_path is None or not source_path.is_file():
            _mark(
                record,
                "keep",
                "temporary_unverified",
                "temporary_copy_source_missing",
            )
            continue
        source_stat = source_path.stat()
        if source_stat.st_size != record["size_bytes"]:
            _mark(
                record,
                "keep",
                "temporary_unverified",
                "temporary_copy_size_differs",
            )
            continue
        for path in (copied_path, source_path):
            cache_key = (path.stat().st_dev, path.stat().st_ino)
            if cache_key not in hash_cache:
                hash_cache[cache_key] = _file_sha256(path)
        copied_key = (copied_path.stat().st_dev, copied_path.stat().st_ino)
        source_key = (source_stat.st_dev, source_stat.st_ino)
        if hash_cache[copied_key] != hash_cache[source_key]:
            _mark(
                record,
                "keep",
                "temporary_unverified",
                "temporary_copy_hash_differs",
            )
            continue
        _mark(
            record,
            "delete",
            "verified_temporary_copies",
            "verified_temporary_checkpoint_copy",
        )
        matched_files += 1
        matched_bytes += record["size_bytes"]
    return {
        "matched_files": matched_files,
        "matched_bytes": matched_bytes,
        "manifest": TEMP_COPY_MANIFEST.as_posix(),
    }


def _load_synced_wandb_ids(path):
    """Load explicitly verified W&B run ids, one per line."""
    if path is None:
        return set()
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.add(value)
    return ids


def _wandb_run_id(record):
    """Return the run id encoded in one W&B event filename."""
    name = record["relative"].name
    if not (name.startswith("run-") and name.endswith(".wandb")):
        return None
    return name[len("run-") : -len(".wandb")]


def _classify_wandb_events(records, synced_ids, protected_paths):
    """Delete only cancelled or explicitly verified W&B event files."""
    for record in records:
        run_id = _wandb_run_id(record)
        if run_id is None or record["path"] in protected_paths:
            continue
        path_text = record["path"]
        if CANCELLED_LAUNCH_SUFFIX.strip("_") in path_text:
            _mark(
                record,
                "delete",
                "cancelled_j8825",
                "cancelled_launch_wandb_event",
            )
        elif run_id in synced_ids:
            _mark(
                record,
                "delete",
                "verified_synced_wandb",
                "verified_synced_wandb_event",
            )
        else:
            _mark(
                record,
                "keep",
                "wandb_unverified",
                "wandb_sync_not_verified",
            )


def _normalize_protected_paths(paths, old_root):
    """Validate user-protected paths and return old-root-relative strings."""
    protected = set()
    for raw_path in paths or []:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = old_root / path
        path = path.resolve()
        try:
            relative = path.relative_to(old_root)
        except ValueError as error:
            raise ValueError(f"Protected path is outside old root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"Protected checkpoint does not exist: {path}")
        protected.add(relative.as_posix())
    return protected


def _manifest_row(record):
    """Return the public fields for one manifest row."""
    return {key: record[key] for key in MANIFEST_FIELDS}


def _sum_by(rows, key):
    """Summarize file counts and byte counts by one row field."""
    summary = defaultdict(lambda: {"files": 0, "logical_bytes": 0})
    for row in rows:
        value = row[key]
        summary[value]["files"] += 1
        summary[value]["logical_bytes"] += row["size_bytes"]
    return dict(sorted(summary.items()))


def _predicted_reclaimable_bytes(records):
    """Count allocated bytes only when every hard link will be removed."""
    by_inode = defaultdict(list)
    for record in records:
        by_inode[(record["device"], record["inode"])].append(record)
    reclaimable = 0
    uncertain = 0
    by_group = defaultdict(int)
    for inode_records in by_inode.values():
        hardlink_count = inode_records[0]["hardlink_count"]
        delete_records = [
            record for record in inode_records if record["action"] == "delete"
        ]
        if len(inode_records) < hardlink_count and delete_records:
            uncertain += inode_records[0]["allocated_bytes"]
            continue
        if len(delete_records) != len(inode_records):
            continue
        allocated = inode_records[0]["allocated_bytes"]
        reclaimable += allocated
        groups = {record["group"] for record in inode_records}
        group = next(iter(groups)) if len(groups) == 1 else "multiple_groups"
        by_group[group] += allocated
    return reclaimable, uncertain, dict(sorted(by_group.items()))


def create_audit(
    old_root,
    clean_root,
    output_dir,
    *,
    synced_wandb_ids_path=None,
    protected_checkpoint_paths=None,
):
    """Create keep/delete manifests without changing old experiment files."""
    old_root = Path(old_root).expanduser().resolve()
    clean_root = Path(clean_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not old_root.is_dir():
        raise NotADirectoryError(f"Old repository does not exist: {old_root}")
    if not clean_root.is_dir():
        raise NotADirectoryError(f"Clean repository does not exist: {clean_root}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an audit directory: {output_dir}")
    output_dir.mkdir(parents=True)

    protected_paths = _normalize_protected_paths(
        protected_checkpoint_paths,
        old_root,
    )
    records, scan_roots = _scan_files(old_root)
    for record in records:
        record["reason"] = _preservation_reason(record)
    source_archive = _write_intrinsic_ambient_source_archive(records, output_dir)
    _classify_intrinsic_ambient_source(records)
    _classify_checkpoint_groups(records, protected_paths)
    temp_summary = _classify_verified_temp_copies(records, old_root)
    synced_ids = _load_synced_wandb_ids(synced_wandb_ids_path)
    _classify_wandb_events(records, synced_ids, protected_paths)

    keep_rows = [
        _manifest_row(record) for record in records if record["action"] == "keep"
    ]
    delete_rows = [
        _manifest_row(record) for record in records if record["action"] == "delete"
    ]
    keep_path = output_dir / "KEEP_MANIFEST.tsv"
    delete_path = output_dir / "DELETE_MANIFEST.tsv"
    _atomic_tsv(keep_path, keep_rows)
    _atomic_tsv(delete_path, delete_rows)
    reclaimable, uncertain, reclaimable_by_group = _predicted_reclaimable_bytes(records)
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": _utc_timestamp(),
        "mode": "dry_run",
        "old_root": str(old_root),
        "clean_root": str(clean_root),
        "output_dir": str(output_dir),
        "scan_roots": scan_roots,
        "explicitly_untouched": [
            "mem_param_clean current outputs",
            "old_mem_par/outputs/evals",
            "old_mem_par/data",
            "old repository source code outside the intrinsic/ambient prototype",
        ],
        "retention_policy": {
            "keep_first_checkpoint": True,
            "keep_every_nth_saved_checkpoint": 10,
            "keep_final_checkpoint": True,
            "merge_resumed_attempts_by_step": True,
            "known_cancelled_launch": "j8825",
        },
        "protected_run_steps": {
            key: sorted(value) for key, value in PROTECTED_RUN_STEPS.items()
        },
        "explicit_protected_paths": sorted(protected_paths),
        "verified_synced_wandb_ids": sorted(synced_ids),
        "temporary_copy_verification": temp_summary,
        "files_scanned": len(records),
        "files_kept": len(keep_rows),
        "files_marked_for_deletion": len(delete_rows),
        "intrinsic_ambient_source_archive": source_archive,
        "delete_logical_bytes": sum(row["size_bytes"] for row in delete_rows),
        "predicted_reclaimable_bytes": reclaimable,
        "predicted_reclaimable_tib": reclaimable / (1024**4),
        "unresolved_external_hardlink_bytes": uncertain,
        "keep_by_reason": _sum_by(keep_rows, "reason"),
        "delete_by_reason": _sum_by(delete_rows, "reason"),
        "delete_by_group": _sum_by(delete_rows, "group"),
        "predicted_reclaimable_bytes_by_group": reclaimable_by_group,
        "keep_manifest": keep_path.name,
        "delete_manifest": delete_path.name,
        "keep_manifest_sha256": _file_sha256(keep_path),
        "delete_manifest_sha256": _file_sha256(delete_path),
        "deletion_applied": False,
    }
    _atomic_json(output_dir / "SUMMARY.json", summary)
    return summary


def _read_manifest(path):
    """Read one audit TSV and restore numeric fields."""
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Unexpected manifest columns: {path}")
        for row in reader:
            for key in (
                "size_bytes",
                "allocated_bytes",
                "mtime_ns",
                "device",
                "inode",
                "hardlink_count",
            ):
                row[key] = int(row[key])
            rows.append(row)
    return rows


def _validate_apply_row(row, old_root):
    """Validate one deletion target against its recorded filesystem identity."""
    relative = Path(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {row['path']}")
    path = old_root / relative
    resolved_parent = path.parent.resolve()
    try:
        resolved_parent.relative_to(old_root)
    except ValueError as error:
        raise ValueError(f"Deletion target escapes old root: {path}") from error
    file_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"Deletion target is not a regular file: {path}")
    expected = {
        "size_bytes": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "hardlink_count": file_stat.st_nlink,
    }
    for key, current in expected.items():
        if row[key] != current:
            raise ValueError(
                f"Deletion target changed for {row['path']}: "
                f"{key} was {row[key]}, now {current}"
            )
    return path


def apply_audit(old_root, audit_dir, *, confirm_root, groups=None):
    """Apply validated file deletions from one immutable audit manifest."""
    old_root = Path(old_root).expanduser().resolve()
    audit_dir = Path(audit_dir).expanduser().resolve()
    confirmed = Path(confirm_root).expanduser().resolve()
    if confirmed != old_root:
        raise ValueError("--confirm-root must resolve to the exact old root.")
    summary_path = audit_dir / "SUMMARY.json"
    delete_path = audit_dir / "DELETE_MANIFEST.tsv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("Unsupported cleanup audit schema.")
    if Path(summary["old_root"]).resolve() != old_root:
        raise ValueError("Audit old root does not match the requested old root.")
    if summary.get("delete_manifest_sha256") != _file_sha256(delete_path):
        raise ValueError("Delete manifest hash does not match SUMMARY.json.")
    rows = _read_manifest(delete_path)
    selected_groups = set(groups or [])
    if selected_groups:
        available = {row["group"] for row in rows}
        unknown = selected_groups - available
        if unknown:
            raise ValueError(f"Unknown cleanup groups: {sorted(unknown)}")
        rows = [row for row in rows if row["group"] in selected_groups]
    if not rows:
        raise ValueError("No deletion rows were selected.")
    selected_row_groups = {row["group"] for row in rows}
    if "obsolete_intrinsic_ambient_source" in selected_row_groups:
        archive_record = summary.get("intrinsic_ambient_source_archive")
        if not archive_record:
            raise ValueError("The intrinsic/ambient source archive is missing.")
        archive_path = audit_dir / archive_record["path"]
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"The intrinsic/ambient source archive is missing: {archive_path}"
            )
        if _file_sha256(archive_path) != archive_record["sha256"]:
            raise ValueError("The intrinsic/ambient source archive hash changed.")
    targets = [_validate_apply_row(row, old_root) for row in rows]

    apply_label = "all"
    if selected_groups:
        apply_label = "--".join(sorted(selected_groups))
    if re.fullmatch(r"[a-z0-9_-]+(?:--[a-z0-9_-]+)*", apply_label) is None:
        raise ValueError(f"Unsafe cleanup group label: {apply_label}")
    progress_path = audit_dir / f"APPLY_PROGRESS.{apply_label}.jsonl"
    if progress_path.exists():
        raise FileExistsError(
            f"Apply progress already exists for {apply_label}. Review it first."
        )
    applied = []
    with progress_path.open("x", encoding="utf-8") as progress:
        for row, path in zip(rows, targets):
            path.unlink()
            event = {
                "deleted_at": _utc_timestamp(),
                "path": row["path"],
                "group": row["group"],
                "size_bytes": row["size_bytes"],
                "inode": row["inode"],
            }
            progress.write(json.dumps(event, sort_keys=True) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            applied.append(event)
    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "completed_at": _utc_timestamp(),
        "old_root": str(old_root),
        "audit_dir": str(audit_dir),
        "groups": sorted(selected_groups) if selected_groups else "all",
        "deleted_files": len(applied),
        "deleted_logical_bytes": sum(item["size_bytes"] for item in applied),
        "delete_manifest_sha256": summary["delete_manifest_sha256"],
    }
    _atomic_json(audit_dir / f"APPLIED.{apply_label}.json", result)
    return result


def build_parser():
    """Build the old-storage cleanup command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--clean-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verified-synced-wandb-ids")
    parser.add_argument("--protect-checkpoint", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-root")
    parser.add_argument("--group", action="append", default=[])
    return parser


def main(argv=None):
    """Create a dry-run audit or apply one previously reviewed manifest."""
    args = build_parser().parse_args(argv)
    if args.apply:
        if not args.confirm_root:
            raise ValueError("--apply requires --confirm-root.")
        result = apply_audit(
            args.old_root,
            args.output_dir,
            confirm_root=args.confirm_root,
            groups=args.group,
        )
    else:
        if args.group:
            raise ValueError("--group is valid only with --apply.")
        if args.confirm_root:
            raise ValueError("--confirm-root is valid only with --apply.")
        result = create_audit(
            args.old_root,
            args.clean_root,
            args.output_dir,
            synced_wandb_ids_path=args.verified_synced_wandb_ids,
            protected_checkpoint_paths=args.protect_checkpoint,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
