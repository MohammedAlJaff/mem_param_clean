#!/usr/bin/env python3
"""Persist one canonical CelebA validation tensor for post-hoc FID."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ml_collections import ConfigDict

from mem_param_src.datasets import build_data_bundle
from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    write_json,
)


ARTIFACT_FORMAT_VERSION = 1


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    return list(value)


def _verify_equal(name, candidate, saved):
    import torch

    if torch.is_tensor(candidate) and torch.is_tensor(saved):
        equal = candidate.shape == saved.shape and torch.equal(candidate, saved)
    else:
        equal = _as_list(candidate) == _as_list(saved)
    if not equal:
        raise RuntimeError(f"Rebuilt FID reference does not match saved {name} prefix.")


def build_reference(source_run, output_artifact, reference_count):
    """Rebuild once, verify against training provenance, then freeze the tensor."""
    source_run = Path(source_run).expanduser().resolve()
    output_artifact = Path(output_artifact).expanduser().resolve()
    manifest_path = output_artifact.with_suffix(".json")
    if output_artifact.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace an existing FID reference: {output_artifact}"
        )

    config = ConfigDict(read_json(source_run / "config.json"))
    if str(config.data.dataset) != "celeba32_grey":
        raise ValueError("This reference builder is restricted to CelebA32.")
    if str(config.data.reference_split) != "valid":
        raise ValueError("Expected the training reference split to be CelebA valid.")
    config.data.n_test = int(reference_count)

    saved_bundle = load_torch(source_run / "data_bundle.pt")
    rebuilt = build_data_bundle(config)
    reference = rebuilt["reference"].contiguous()
    saved_count = int(saved_bundle["reference"].shape[0])
    if int(reference.shape[0]) != int(reference_count):
        raise RuntimeError("The rebuilt reference count does not match the request.")
    if int(reference_count) < saved_count:
        raise ValueError(
            f"FID reference count {reference_count} is smaller than saved prefix "
            f"count {saved_count}."
        )

    _verify_equal(
        "reference tensor",
        reference[:saved_count],
        saved_bundle["reference"],
    )
    for suffix in ("labels", "indices", "sample_ids"):
        key = f"reference_{suffix}"
        if key in saved_bundle:
            _verify_equal(key, rebuilt[key][:saved_count], saved_bundle[key])
    for suffix in (
        "sample_id_kind",
        "source_size",
        "source_catalog_sha256",
        "source_id",
        "split",
    ):
        key = f"reference_{suffix}"
        if rebuilt.get(key) != saved_bundle.get(key):
            raise RuntimeError(f"Rebuilt FID reference disagrees on {key}.")

    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "reference": reference,
        "reference_labels": rebuilt.get("reference_labels"),
        "reference_indices": rebuilt.get("reference_indices"),
        "reference_sample_ids": rebuilt.get("reference_sample_ids"),
        "reference_sample_id_kind": rebuilt.get("reference_sample_id_kind"),
        "reference_source_size": rebuilt.get("reference_source_size"),
        "reference_source_catalog_sha256": rebuilt.get(
            "reference_source_catalog_sha256"
        ),
        "reference_source_id": rebuilt.get("reference_source_id"),
        "reference_split": rebuilt.get("reference_split"),
        "selection_seed": int(config.training.seed) + 1,
        "source_run": str(source_run),
        "saved_prefix_count": saved_count,
        "saved_prefix_tensor_sha256": tensor_sha256(saved_bundle["reference"]),
        "preprocessing": {
            "crop": "center_140",
            "resize_size": int(config.data.resize_size),
            "image_size": int(config.data.image_size),
            "image_channels": int(config.data.image_channels),
            "value_range": [-1.0, 1.0],
            "training_augmentation": False,
        },
    }
    save_torch(output_artifact, payload)
    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(output_artifact),
        "artifact_sha256": file_sha256(output_artifact),
        "reference_tensor_sha256": tensor_sha256(reference),
        "reference_shape": list(reference.shape),
        "reference_count": int(reference_count),
        "reference_source_id": payload["reference_source_id"],
        "reference_split": payload["reference_split"],
        "reference_source_size": payload["reference_source_size"],
        "reference_source_catalog_sha256": payload[
            "reference_source_catalog_sha256"
        ],
        "selection_seed": payload["selection_seed"],
        "source_run": str(source_run),
        "saved_prefix_count": saved_count,
        "saved_prefix_tensor_sha256": payload["saved_prefix_tensor_sha256"],
        "saved_prefix_verified": True,
        "preprocessing": payload["preprocessing"],
    }
    write_json(manifest_path, manifest)
    print(
        f"reference={output_artifact} count={reference_count} "
        f"saved_prefix_verified={saved_count}",
        flush=True,
    )
    return output_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--reference-count", type=int, default=10_000)
    args = parser.parse_args(argv)
    if int(args.reference_count) < 1:
        parser.error("--reference-count must be positive")
    build_reference(args.source_run, args.output_artifact, args.reference_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
