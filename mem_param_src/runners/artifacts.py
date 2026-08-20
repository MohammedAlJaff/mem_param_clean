"""Saved-run artifact operations for training and checkpoint replay."""

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch


ARTIFACT_FORMAT_VERSION = 2
_CHECKPOINT_PATTERN = re.compile(r"^step_(\d+)\.pt$")
_EVALUATION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now():
    """Return one UTC timestamp for run records."""
    return datetime.now(timezone.utc).isoformat()


def _cpu_value(value):
    """Move tensors in one artifact payload to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    if isinstance(value, list):
        return [_cpu_value(item) for item in value]
    return value


def write_json(path, payload):
    """Write one JSON artifact through a temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    """Read one JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(path, payload):
    """Save one CPU tensor payload through a temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(_cpu_value(payload), temporary)
    temporary.replace(path)


def load_torch(path):
    """Load one trusted saved-run tensor payload onto CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def file_sha256(path):
    """Hash one saved artifact without loading it into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    """Hash one tensor with its dtype, shape, and exact CPU bytes."""
    if not torch.is_tensor(tensor):
        raise TypeError("tensor_sha256 expects a torch.Tensor.")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(metadata)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_sequence(value):
    """Convert one tensor or sequence into JSON-compatible scalar values."""
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    return list(value)


def _tensor_manifest_record(bundle_key, tensor):
    """Describe one exact tensor stored inside the saved data bundle."""
    if tensor is None:
        return None
    return {
        "bundle_key": bundle_key,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_sha256(tensor),
    }


def _source_manifest_record(bundle, partition, finite_catalog):
    """Describe one complete source pool before subset selection."""
    tensor = bundle[partition]
    return {
        "source_id": str(bundle[f"{partition}_source_id"]),
        "split": str(bundle.get(f"{partition}_split", "")),
        "size": int(bundle.get(f"{partition}_source_size", tensor.shape[0])),
        "finite_catalog": bool(finite_catalog),
        "sample_id_kind": bundle.get(f"{partition}_sample_id_kind"),
        "catalog_sha256": bundle.get(f"{partition}_source_catalog_sha256"),
    }


def _partition_manifest_record(bundle, partition, source_name):
    """Describe selected members in exact tensor order."""
    tensor = bundle[partition]
    labels = bundle.get(f"{partition}_labels")
    return {
        "role": str(bundle.get(f"{partition}_role", partition)),
        "source": source_name,
        "sample_count": int(tensor.shape[0]),
        "sample_ids": _json_sequence(bundle.get(f"{partition}_sample_ids")),
        "source_indices": _json_sequence(bundle.get(f"{partition}_indices")),
        "tensor": _tensor_manifest_record(partition, tensor),
        "labels": _tensor_manifest_record(f"{partition}_labels", labels),
    }


def build_data_manifest(config, bundle, *, run_kind, bundle_sha256):
    """Build the readable membership and tensor-integrity record for one run."""
    finite_catalog = run_kind == "image"
    preprocessing = {
        "exact_post_transform_tensors_saved": True,
        "training_augmentation": False,
        "flatten": bool(config.data.flatten),
    }
    if run_kind == "image":
        preprocessing.update(
            {
                "image_channels": int(config.data.image_channels),
                "image_size": int(config.data.image_size),
                "resize_size": int(config.data.resize_size),
                "value_range": [-1.0, 1.0],
            }
        )

    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "run_kind": str(run_kind),
        "dataset": str(config.data.dataset),
        "selection_seed": int(config.training.seed),
        "resolved_config_path": "config.json",
        "unused_tensors_saved": False,
        "bundle": {
            "path": "data_bundle.pt",
            "sha256": str(bundle_sha256),
        },
        "preprocessing": preprocessing,
        "sources": {
            "train_pool": _source_manifest_record(bundle, "train", finite_catalog),
            "reference_pool": _source_manifest_record(
                bundle, "reference", finite_catalog
            ),
        },
        "partitions": {
            "train": _partition_manifest_record(bundle, "train", "train_pool"),
            "reference": _partition_manifest_record(
                bundle, "reference", "reference_pool"
            ),
        },
    }


def save_data_manifest(run_dir, config, bundle, *, run_kind):
    """Write one data manifest beside its already-saved exact data bundle."""
    run_dir = Path(run_dir)
    bundle_path = run_dir / "data_bundle.pt"
    manifest = build_data_manifest(
        config,
        bundle,
        run_kind=run_kind,
        bundle_sha256=file_sha256(bundle_path),
    )
    manifest_path = run_dir / "data_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def validate_data_manifest(run_dir, bundle, *, run_kind, verify_hashes=False):
    """Validate manifest alignment and optionally recompute artifact hashes."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "data_manifest.json")
    if int(manifest.get("artifact_format_version", -1)) != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Saved data manifest uses an unsupported artifact version.")
    if str(manifest.get("run_kind")) != str(run_kind):
        raise ValueError("Saved data manifest has the wrong run kind.")
    if str(manifest.get("bundle", {}).get("path")) != "data_bundle.pt":
        raise ValueError("Saved data manifest has an invalid bundle path.")

    if verify_hashes:
        actual_bundle_hash = file_sha256(run_dir / "data_bundle.pt")
        if actual_bundle_hash != manifest["bundle"]["sha256"]:
            raise ValueError("Saved data bundle SHA-256 does not match its manifest.")

    for partition in ("train", "reference"):
        record = manifest["partitions"][partition]
        tensor = bundle[partition]
        source = manifest["sources"][record["source"]]
        expected_source = _source_manifest_record(
            bundle,
            partition,
            run_kind == "image",
        )
        if source != expected_source:
            raise ValueError(f"Saved {partition} source does not match its manifest.")
        if int(record["sample_count"]) != int(tensor.shape[0]):
            raise ValueError(f"Saved {partition} count does not match its manifest.")
        if record["tensor"]["shape"] != list(tensor.shape) or record["tensor"][
            "dtype"
        ] != str(tensor.dtype):
            raise ValueError(f"Saved {partition} tensor metadata is incorrect.")
        if record["sample_ids"] != _json_sequence(
            bundle.get(f"{partition}_sample_ids")
        ):
            raise ValueError(f"Saved {partition} sample IDs do not match its manifest.")
        if record["source_indices"] != _json_sequence(
            bundle.get(f"{partition}_indices")
        ):
            raise ValueError(f"Saved {partition} indices do not match its manifest.")
        if verify_hashes and record["tensor"]["sha256"] != tensor_sha256(tensor):
            raise ValueError(f"Saved {partition} tensor does not match its manifest.")

        labels = bundle.get(f"{partition}_labels")
        label_record = record["labels"]
        if (labels is None) != (label_record is None):
            raise ValueError(f"Saved {partition} labels do not match its manifest.")
        if labels is not None and (
            label_record["shape"] != list(labels.shape)
            or label_record["dtype"] != str(labels.dtype)
        ):
            raise ValueError(f"Saved {partition} label metadata is incorrect.")
        if (
            verify_hashes
            and labels is not None
            and label_record["sha256"] != tensor_sha256(labels)
        ):
            raise ValueError(f"Saved {partition} labels do not match their manifest.")

    return manifest


def prepare_run_dir(run_dir):
    """Create an empty run directory and its two fixed subdirectories."""
    run_dir = Path(run_dir).expanduser().resolve()
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"Run path is not a directory: {run_dir}")
        if any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
    else:
        run_dir.mkdir(parents=True)

    (run_dir / "checkpoints").mkdir()
    (run_dir / "evaluations").mkdir()
    return run_dir


def create_run_record(run_dir, *, run_kind, seed, device):
    """Create the single run-level state record."""
    now = _utc_now()
    record = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "run_kind": str(run_kind),
        "state": "building_data",
        "seed": int(seed),
        "z_init_seed": int(seed) + 2003,
        "resolved_device": str(device),
        "created_at": now,
        "updated_at": now,
    }
    if run_kind == "synthetic":
        record["sliced_direction_seed"] = int(seed) + 1009
    write_json(Path(run_dir) / "run.json", record)
    return record


def update_run_record(run_dir, **changes):
    """Update the single run-level state record."""
    path = Path(run_dir) / "run.json"
    record = read_json(path)
    record.update(changes)
    record["updated_at"] = _utc_now()
    write_json(path, record)
    return record


def checkpoint_path(run_dir, step):
    """Return the canonical checkpoint path for one optimizer step."""
    return Path(run_dir) / "checkpoints" / f"step_{int(step):06d}.pt"


def save_checkpoint(run_dir, step, model):
    """Save one model checkpoint for deterministic replay."""
    path = checkpoint_path(run_dir, step)
    save_torch(path, {"step": int(step), "model": model.state_dict()})
    return path


def list_checkpoints(run_dir):
    """Return saved checkpoints ordered by optimizer step."""
    checkpoint_dir = Path(run_dir) / "checkpoints"
    found = []
    for path in checkpoint_dir.glob("step_*.pt"):
        match = _CHECKPOINT_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"No checkpoints exist under: {checkpoint_dir}")
    return sorted(found, key=lambda item: item[0])


def prepare_evaluation_dir(run_dir, eval_name):
    """Create one new named evaluation directory without overwriting results."""
    eval_name = str(eval_name)
    if not _EVALUATION_NAME_PATTERN.fullmatch(eval_name):
        raise ValueError(
            "Evaluation name must be one path-safe name containing letters, "
            "numbers, dots, underscores, or hyphens."
        )

    eval_dir = Path(run_dir) / "evaluations" / eval_name
    if eval_dir.exists():
        raise FileExistsError(f"Evaluation already exists: {eval_dir}")
    eval_dir.mkdir(parents=True)
    return eval_dir


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "build_data_manifest",
    "checkpoint_path",
    "create_run_record",
    "file_sha256",
    "list_checkpoints",
    "load_torch",
    "prepare_evaluation_dir",
    "prepare_run_dir",
    "read_json",
    "save_checkpoint",
    "save_data_manifest",
    "save_torch",
    "tensor_sha256",
    "update_run_record",
    "validate_data_manifest",
    "write_json",
]
