"""Build one clean CelebA test reference from saved paper training banks."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import datasets

from mem_param_src.datasets.celeba32 import _build_transform
from mem_param_src.datasets.common import source_catalog_sha256
from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    write_json,
)

REFERENCE_SCHEMA_VERSION = 2
EXPECTED_REFERENCE_COUNT = 10000
EXPECTED_SELECTION_SEED = 1
EXPECTED_PREPROCESSING_IMPLEMENTATION_SHA256 = (
    "47e9f6a66b2312dc8bdd335c4e661cea856cfc0a23e77a7931b0b38df5d2b46b"
)


def _preprocessing_specification(*, image_size, image_channels, resize_size):
    """Describe one exact saved-run CelebA preprocessing transform."""
    image_size = int(image_size)
    image_channels = int(image_channels)
    resize_size = int(resize_size)
    color_operation = (
        "Grayscale(num_output_channels=1)" if image_channels == 1 else 'Convert("RGB")'
    )
    return {
        "operations": [
            "CenterCrop(140)",
            f"Resize({resize_size})",
            f"CenterCrop({image_size})",
            color_operation,
            "ToTensor()",
            f"Normalize(mean={(0.5,) * image_channels}, std={(0.5,) * image_channels})",
        ],
        "image_channels": image_channels,
        "image_size": image_size,
        "resize_size": resize_size,
        "training_augmentation": False,
        "value_range": [-1.0, 1.0],
        "implementation": "mem_param_src/datasets/celeba32.py",
        "implementation_sha256": EXPECTED_PREPROCESSING_IMPLEMENTATION_SHA256,
    }


PREPROCESSING_SPECIFICATION = _preprocessing_specification(
    image_size=32,
    image_channels=1,
    resize_size=32,
)


def _json_sha256(value):
    """Hash one JSON-compatible value with a canonical encoding."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sequence_sha256(name, values):
    """Hash one ordered string sequence with a named source identity."""
    return source_catalog_sha256(name, values)


def _celeba_path(dataset, filename):
    """Return the official aligned JPEG path for one catalog filename."""
    return Path(dataset.root) / dataset.base_folder / "img_align_celeba" / filename


def map_split_indices_to_filenames(indices, split_filenames):
    """Map split-local indices through the ordered official split catalog."""
    mapped = []
    for value in indices:
        index = int(value)
        if index < 0 or index >= len(split_filenames):
            raise ValueError(f"CelebA split-local index is out of range: {index}.")
        mapped.append(str(split_filenames[index]))
    return mapped


def _discover_data_bundles(source_roots):
    """Find every saved data bundle below the declared run roots."""
    bundles = set()
    for source in source_roots:
        path = Path(source).expanduser().resolve()
        if path.is_file():
            if path.name != "data_bundle.pt":
                raise ValueError(f"Expected a data_bundle.pt file, got: {path}")
            bundles.add(path)
        elif (path / "data_bundle.pt").is_file():
            bundles.add(path / "data_bundle.pt")
        elif path.is_dir():
            bundles.update(path.rglob("data_bundle.pt"))
        else:
            raise FileNotFoundError(f"Missing source run root: {path}")
    if not bundles:
        raise ValueError("No saved CelebA data_bundle.pt files were found.")
    return sorted(path.resolve() for path in bundles)


def _validated_train_record(bundle_path, train_dataset):
    """Verify one saved bundle and map its train indices to global filenames."""
    run_dir = bundle_path.parent
    manifest_path = run_dir / "data_manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Saved run lacks config.json or data_manifest.json: {run_dir}"
        )

    manifest = read_json(manifest_path)
    config = read_json(config_path)
    if file_sha256(bundle_path) != manifest["bundle"]["sha256"]:
        raise ValueError(f"Saved data bundle hash does not match: {bundle_path}")
    if not str(config["data"]["dataset"]).startswith("celeba"):
        raise ValueError(f"Declared source is not a CelebA run: {run_dir}")

    bundle = load_torch(bundle_path)
    training = bundle.get("train")
    indices = bundle.get("train_indices")
    if not torch.is_tensor(training) or not torch.is_tensor(indices):
        raise ValueError(f"Saved CelebA run lacks train tensor or indices: {run_dir}")
    training = training.contiguous()
    indices = indices.to(torch.int64).flatten()
    if len(training) != len(indices):
        raise ValueError(f"Train tensor and source indices are misaligned: {run_dir}")
    if str(bundle.get("train_source_id")) != "torchvision_celeba:train":
        raise ValueError(f"Unexpected CelebA training source in {run_dir}.")
    if str(bundle.get("train_split")) != "train":
        raise ValueError(f"Unexpected CelebA training split in {run_dir}.")
    if int(bundle.get("train_source_size", -1)) != len(train_dataset):
        raise ValueError(
            f"Saved and official CelebA train catalog sizes differ: {run_dir}"
        )

    expected_catalog_hash = source_catalog_sha256(
        "torchvision_celeba:train", range(len(train_dataset))
    )
    if bundle.get("train_source_catalog_sha256") != expected_catalog_hash:
        raise ValueError(f"Saved CelebA train catalog hash differs: {run_dir}")
    manifest_tensor_hash = manifest["partitions"]["train"]["tensor"]["sha256"]
    train_tensor_hash = tensor_sha256(training)
    if train_tensor_hash != manifest_tensor_hash:
        raise ValueError(f"Saved CelebA train tensor hash differs: {run_dir}")

    index_values = indices.tolist()
    filenames = map_split_indices_to_filenames(index_values, train_dataset.filename)
    return {
        "run_dir": str(run_dir),
        "data_bundle": str(bundle_path),
        "data_bundle_sha256": file_sha256(bundle_path),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": file_sha256(manifest_path),
        "dataset": str(config["data"]["dataset"]),
        "image_channels": int(config["data"]["image_channels"]),
        "image_size": int(config["data"]["image_size"]),
        "train_count": len(training),
        "train_shape": list(training.shape),
        "train_tensor_sha256": train_tensor_hash,
        "train_indices": index_values,
        "train_indices_sha256": _sequence_sha256(
            "celeba:train:split-local-indices", [str(value) for value in index_values]
        ),
        "train_filenames": filenames,
        "train_filenames_sha256": _sequence_sha256(
            "celeba:train:global-filenames", filenames
        ),
        "training": training,
    }


def build_training_union(
    source_roots, train_dataset, expected_tensor_shape=(1, 32, 32)
):
    """Build raw-JPEG and matching transformed-tensor training unions."""
    expected_tensor_shape = tuple(int(value) for value in expected_tensor_shape)
    bundle_paths = _discover_data_bundles(source_roots)
    jpeg_cache = {}
    per_tensor_cache = {}
    source_runs = []
    global_filenames = set()
    jpeg_hashes = set()
    celeba32_tensor_hashes = set()

    for position, bundle_path in enumerate(bundle_paths, start=1):
        record = _validated_train_record(bundle_path, train_dataset)
        filenames = record.pop("train_filenames")
        training = record.pop("training")
        run_jpeg_hashes = []
        for filename in filenames:
            if filename not in jpeg_cache:
                jpeg_cache[filename] = file_sha256(
                    _celeba_path(train_dataset, filename)
                )
            jpeg_hash = jpeg_cache[filename]
            global_filenames.add(filename)
            jpeg_hashes.add(jpeg_hash)
            run_jpeg_hashes.append(jpeg_hash)

        record["train_jpeg_sha256s_sha256"] = _sequence_sha256(
            "celeba:train:jpeg-sha256s", run_jpeg_hashes
        )
        if tuple(training.shape[1:]) == expected_tensor_shape:
            whole_hash = record["train_tensor_sha256"]
            if whole_hash not in per_tensor_cache:
                per_tensor_cache[whole_hash] = [
                    tensor_sha256(image) for image in training
                ]
            per_image_hashes = per_tensor_cache[whole_hash]
            celeba32_tensor_hashes.update(per_image_hashes)
            record["train_per_image_tensor_sha256s_sha256"] = _sequence_sha256(
                "celeba32:train:per-image-tensor-sha256s", per_image_hashes
            )
        else:
            record["train_per_image_tensor_sha256s_sha256"] = None
        source_runs.append(record)
        if position == 1 or position % 25 == 0 or position == len(bundle_paths):
            print(
                f"[training-union] {position}/{len(bundle_paths)} saved runs verified.",
                flush=True,
            )

    filename_records = [
        {"filename": filename, "jpeg_sha256": jpeg_cache[filename]}
        for filename in sorted(global_filenames)
    ]
    manifest = {
        "source_roots": [
            str(Path(path).expanduser().resolve()) for path in source_roots
        ],
        "source_run_count": len(source_runs),
        "source_runs": source_runs,
        "global_filename_count": len(global_filenames),
        "global_filename_records": filename_records,
        "global_filenames_sha256": _sequence_sha256(
            "celeba:training-union:global-filenames", sorted(global_filenames)
        ),
        "unique_jpeg_sha256_count": len(jpeg_hashes),
        "jpeg_sha256s_sha256": _sequence_sha256(
            "celeba:training-union:jpeg-sha256s", sorted(jpeg_hashes)
        ),
        "celeba32_tensor_sha256_count": len(celeba32_tensor_hashes),
        "celeba32_tensor_sha256s": sorted(celeba32_tensor_hashes),
        "celeba32_tensor_sha256s_sha256": _sequence_sha256(
            "celeba32:training-union:tensor-sha256s",
            sorted(celeba32_tensor_hashes),
        ),
    }
    return manifest, jpeg_hashes, celeba32_tensor_hashes


def select_clean_reference(
    candidate_indices,
    count,
    *,
    filename_for_index,
    jpeg_hash_for_index,
    tensor_and_label_for_index,
    training_jpeg_hashes,
    training_tensor_hashes,
    progress=False,
):
    """Walk one candidate order and refill after every detected collision."""
    accepted = {
        "indices": [],
        "filenames": [],
        "jpeg_sha256s": [],
        "tensor_sha256s": [],
        "labels": [],
        "tensors": [],
    }
    accepted_filenames = set()
    accepted_jpeg_hashes = set()
    accepted_tensor_hashes = set()
    rejections = {
        "duplicate_filename": 0,
        "training_jpeg_overlap": 0,
        "internal_jpeg_duplicate": 0,
        "training_tensor_overlap": 0,
        "internal_tensor_duplicate": 0,
    }
    candidates_examined = 0

    for raw_index in candidate_indices:
        index = int(raw_index)
        candidates_examined += 1
        filename = str(filename_for_index(index))
        if filename in accepted_filenames:
            rejections["duplicate_filename"] += 1
            continue
        jpeg_hash = str(jpeg_hash_for_index(index))
        if jpeg_hash in training_jpeg_hashes:
            rejections["training_jpeg_overlap"] += 1
            continue
        if jpeg_hash in accepted_jpeg_hashes:
            rejections["internal_jpeg_duplicate"] += 1
            continue

        tensor, label = tensor_and_label_for_index(index)
        tensor = tensor.contiguous()
        transformed_hash = tensor_sha256(tensor)
        if transformed_hash in training_tensor_hashes:
            rejections["training_tensor_overlap"] += 1
            continue
        if transformed_hash in accepted_tensor_hashes:
            rejections["internal_tensor_duplicate"] += 1
            continue

        accepted["indices"].append(index)
        accepted["filenames"].append(filename)
        accepted["jpeg_sha256s"].append(jpeg_hash)
        accepted["tensor_sha256s"].append(transformed_hash)
        accepted["labels"].append(int(torch.as_tensor(label).item()))
        accepted["tensors"].append(tensor)
        accepted_filenames.add(filename)
        accepted_jpeg_hashes.add(jpeg_hash)
        accepted_tensor_hashes.add(transformed_hash)
        if progress and (len(accepted["indices"]) % 1000 == 0):
            print(
                f"[test-reference] {len(accepted['indices']):,}/{count:,} "
                f"images accepted after {candidates_examined:,} candidates.",
                flush=True,
            )
        if len(accepted["indices"]) == int(count):
            break

    if len(accepted["indices"]) != int(count):
        raise RuntimeError(
            f"Only {len(accepted['indices'])} clean test images were available; "
            f"{count} were required."
        )
    accepted["candidates_examined"] = candidates_examined
    accepted["rejection_counts"] = rejections
    return accepted


def _build_transform_and_verify_source(
    *, image_size=32, image_channels=1, resize_size=None
):
    """Construct the training transform after pinning its implementation hash."""
    source_path = Path(__file__).resolve().parents[1] / "datasets" / "celeba32.py"
    actual_hash = file_sha256(source_path)
    if actual_hash != EXPECTED_PREPROCESSING_IMPLEMENTATION_SHA256:
        raise ValueError(
            "CelebA preprocessing implementation SHA-256 differs from the "
            "training-time source."
        )
    image_size = int(image_size)
    image_channels = int(image_channels)
    resize_size = image_size if resize_size is None else int(resize_size)
    config = SimpleNamespace(
        data=SimpleNamespace(
            image_size=image_size,
            image_channels=image_channels,
            resize_size=resize_size,
        )
    )
    return _build_transform(config)


def build_reference(
    celeba_root,
    source_roots,
    output_artifact,
    count=10000,
    seed=1,
    *,
    image_size=32,
    image_channels=1,
    resize_size=None,
):
    """Build, verify, and save the immutable clean CelebA test reference."""
    count = int(count)
    seed = int(seed)
    image_size = int(image_size)
    image_channels = int(image_channels)
    resize_size = image_size if resize_size is None else int(resize_size)
    if count != EXPECTED_REFERENCE_COUNT:
        raise ValueError("The canonical CelebA test reference requires 10,000 images.")
    if seed != EXPECTED_SELECTION_SEED:
        raise ValueError(
            "The canonical CelebA test reference requires selection seed 1."
        )

    output_artifact = Path(output_artifact).expanduser().resolve()
    manifest_path = output_artifact.with_suffix(".json")
    if output_artifact.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace an existing CelebA reference: {output_artifact}"
        )

    preprocessing = _preprocessing_specification(
        image_size=image_size,
        image_channels=image_channels,
        resize_size=resize_size,
    )
    transform = _build_transform_and_verify_source(
        image_size=image_size,
        image_channels=image_channels,
        resize_size=resize_size,
    )
    celeba_root = Path(celeba_root).expanduser().resolve()
    train_dataset = datasets.CelebA(
        root=celeba_root,
        split="train",
        target_type="identity",
        download=False,
    )
    test_dataset = datasets.CelebA(
        root=celeba_root,
        split="test",
        target_type="identity",
        transform=transform,
        download=False,
    )
    partition_path = celeba_root / test_dataset.base_folder / "list_eval_partition.txt"
    partition_hash = file_sha256(partition_path)

    training_union, training_jpeg_hashes, training_tensor_hashes = build_training_union(
        source_roots,
        train_dataset,
        expected_tensor_shape=(image_channels, image_size, image_size),
    )
    training_union_hash = _json_sha256(training_union)
    order = torch.randperm(
        len(test_dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    selected = select_clean_reference(
        order,
        count,
        filename_for_index=lambda index: test_dataset.filename[index],
        jpeg_hash_for_index=lambda index: file_sha256(
            _celeba_path(test_dataset, test_dataset.filename[index])
        ),
        tensor_and_label_for_index=lambda index: test_dataset[index],
        training_jpeg_hashes=training_jpeg_hashes,
        training_tensor_hashes=training_tensor_hashes,
        progress=True,
    )

    reference = torch.stack(selected["tensors"]).contiguous()
    reference_indices = torch.as_tensor(selected["indices"], dtype=torch.int64)
    reference_labels = torch.as_tensor(selected["labels"], dtype=torch.int64)
    reference_hash = tensor_sha256(reference)
    test_catalog_hash = source_catalog_sha256(
        "torchvision_celeba:test", [str(name) for name in test_dataset.filename]
    )
    payload = {
        "artifact_format_version": REFERENCE_SCHEMA_VERSION,
        "reference": reference,
        "reference_tensor_sha256": reference_hash,
        "reference_indices": reference_indices,
        "reference_filenames": selected["filenames"],
        "reference_sample_ids": selected["filenames"],
        "reference_jpeg_sha256s": selected["jpeg_sha256s"],
        "reference_tensor_sha256s": selected["tensor_sha256s"],
        "reference_labels": reference_labels,
        "reference_split": "test",
        "reference_source_id": "torchvision_celeba:test",
        "reference_source_size": len(test_dataset),
        "reference_source_catalog_sha256": test_catalog_hash,
        "selection_seed": seed,
        "selection": {
            "method": "torch.randperm_then_collision_filter_and_refill",
            "seed": seed,
            "requested_count": count,
            "candidates_examined": selected["candidates_examined"],
            "rejection_counts": selected["rejection_counts"],
        },
        "preprocessing": dict(preprocessing),
        "partition_file": str(partition_path),
        "partition_file_sha256": partition_hash,
        "training_union": training_union,
        "training_union_manifest_sha256": training_union_hash,
    }
    save_torch(output_artifact, payload)

    manifest = {
        "artifact_format_version": REFERENCE_SCHEMA_VERSION,
        "artifact": str(output_artifact),
        "artifact_sha256": file_sha256(output_artifact),
        "reference_count": count,
        "reference_shape": list(reference.shape),
        "reference_tensor_sha256": reference_hash,
        "reference_indices_sha256": tensor_sha256(reference_indices),
        "reference_labels_sha256": tensor_sha256(reference_labels),
        "reference_filenames_sha256": _sequence_sha256(
            "celeba:test:accepted-filenames", selected["filenames"]
        ),
        "reference_jpeg_sha256s_sha256": _sequence_sha256(
            "celeba:test:accepted-jpeg-sha256s", selected["jpeg_sha256s"]
        ),
        "reference_tensor_sha256s_sha256": _sequence_sha256(
            "celeba32:test:accepted-tensor-sha256s", selected["tensor_sha256s"]
        ),
        "reference_split": "test",
        "reference_source_id": "torchvision_celeba:test",
        "reference_source_size": len(test_dataset),
        "reference_source_catalog_sha256": test_catalog_hash,
        "selection_seed": seed,
        "selection": payload["selection"],
        "preprocessing": dict(preprocessing),
        "partition_file": str(partition_path),
        "partition_file_sha256": partition_hash,
        "training_union_manifest_sha256": training_union_hash,
        "training_union": training_union,
    }
    write_json(manifest_path, manifest)
    print(f"[complete] Reference: {output_artifact}", flush=True)
    print(f"[complete] Manifest: {manifest_path}", flush=True)
    print(f"[complete] Artifact SHA-256: {manifest['artifact_sha256']}", flush=True)
    print(f"[complete] Tensor SHA-256: {reference_hash}", flush=True)
    return output_artifact, manifest_path


def build_parser():
    """Build the clean CelebA test-reference command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba-root", required=True)
    parser.add_argument("--source-run-root", action="append", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--count", type=int, default=EXPECTED_REFERENCE_COUNT)
    parser.add_argument("--seed", type=int, default=EXPECTED_SELECTION_SEED)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-channels", type=int, choices=(1, 3), default=1)
    parser.add_argument("--resize-size", type=int)
    return parser


def main(argv=None):
    """Build one clean, fixed CelebA test reference artifact."""
    args = build_parser().parse_args(argv)
    build_reference(
        args.celeba_root,
        args.source_run_root,
        args.output_artifact,
        count=args.count,
        seed=args.seed,
        image_size=args.image_size,
        image_channels=args.image_channels,
        resize_size=args.resize_size,
    )
    return 0


if __name__ == "__main__":
    main()
