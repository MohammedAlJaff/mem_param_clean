"""
Canonical MNIST subset loader for first-party studies.

This loader intentionally keeps the data path simple:
- sample deterministic subsets from the torchvision train and test splits
- normalize pixels to `[-1, 1]`
- optionally resize the images to the configured square resolution
- optionally flatten the images when a vector-valued backbone is requested
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.datasets.mnist import read_image_file, read_label_file

from mem_param_src.datasets.common import (
    maybe_flatten,
    seeded_indices,
    source_catalog_sha256,
)

MNIST_NUM_CLASSES = 10


def _maybe_resize_images(tensor, config):
    """
    Resize image tensors to `config.data.image_size` when needed.

    MNIST ships as 28x28 images, but the live study surface sometimes wants a
    32x32 image backbone. We keep that conversion here so the rest of the
    canonical data-bundle contract stays the same.
    """
    target_size = int(config.data.image_size)
    if target_size <= 0:
        raise ValueError(
            f"config.data.image_size must be positive for MNIST, got {target_size}"
        )

    if tensor.shape[-1] == target_size and tensor.shape[-2] == target_size:
        return tensor

    return F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def _require_divisible_by_class_count(count, split_name, num_classes=MNIST_NUM_CLASSES):
    """
    Require that a balanced subset can allocate the same count to every class.
    """
    if count % num_classes != 0:
        raise ValueError(
            f"Balanced {split_name} selection requires count divisible by "
            f"{num_classes}, got {count}"
        )
    return count // num_classes


def _balanced_seeded_indices(targets, count, seed, split_name):
    """
    Draw an equal number of examples from every MNIST class.

    The returned indices are shuffled after per-class sampling so the saved
    subset remains class balanced without being grouped by label order.
    """
    per_class = _require_divisible_by_class_count(count, split_name)
    generator = torch.Generator().manual_seed(int(seed))
    selected = []

    for class_id in range(MNIST_NUM_CLASSES):
        class_indices = torch.nonzero(targets == class_id, as_tuple=False).flatten()
        if class_indices.numel() < per_class:
            raise ValueError(
                f"Balanced {split_name} selection requested {per_class} examples "
                f"for class {class_id}, but only {class_indices.numel()} are available."
            )
        perm = torch.randperm(class_indices.numel(), generator=generator)[:per_class]
        selected.append(class_indices[perm])

    indices = torch.cat(selected)
    shuffle = torch.randperm(indices.numel(), generator=generator)
    return indices[shuffle]


def _resolve_class_balance(balance_classes, count, split_name):
    """
    Disable impossible balanced MNIST sampling while preserving subset size.
    """
    requested = bool(balance_classes)
    count = int(count)
    if not requested or count % MNIST_NUM_CLASSES == 0:
        return requested, None

    return False, {
        "split": split_name,
        "requested_count": count,
        "num_classes": MNIST_NUM_CLASSES,
        "requested_balance": True,
        "actual_balance": False,
        "reason": f"count is not divisible by {MNIST_NUM_CLASSES}",
    }


def _select_indices(targets, count, seed, balance_classes, split_name):
    """
    Pick deterministic MNIST subset indices, optionally enforcing class balance.
    """
    if balance_classes:
        return _balanced_seeded_indices(targets, count, seed, split_name)
    return seeded_indices(len(targets), count, seed)


def _raw_mnist_paths(root, train):
    """
    Return raw IDX file paths for the requested MNIST split.
    """
    raw_dir = Path(root) / "MNIST" / "raw"
    if train:
        return raw_dir / "train-images-idx3-ubyte", raw_dir / "train-labels-idx1-ubyte"
    return raw_dir / "t10k-images-idx3-ubyte", raw_dir / "t10k-labels-idx1-ubyte"


def _load_raw_mnist_split(root, train):
    """
    Load one MNIST split directly from raw IDX files.
    """
    image_path, label_path = _raw_mnist_paths(root, train)
    missing = [str(path) for path in (image_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return read_image_file(str(image_path)), read_label_file(str(label_path))


def _load_mnist_split(root, train, download):
    """
    Load one MNIST split from torchvision, with a raw-file fallback.
    """
    try:
        dataset = datasets.MNIST(root=root, train=train, download=download)
        return dataset.data, dataset.targets
    except RuntimeError:
        if download:
            raise
        return _load_raw_mnist_split(root, train)


def _class_count_summary(labels, num_classes=MNIST_NUM_CLASSES):
    """
    Convert label counts into a JSON-friendly `{class_id: count}` dict.
    """
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes)
    return {
        str(class_id): int(counts[class_id].item()) for class_id in range(num_classes)
    }


def load_mnist_bundle(config):
    """
    Return a deterministic MNIST train/reference bundle plus labels and indices.

    The training subset comes from the torchvision training split. The
    reference subset comes from the test split so memorization sanity checks can
    compare generated samples against held-out real data as well.

    This is a compact example of the shared data-bundle contract in practice:
    MNIST still returns the same train/reference/labels/indices structure as
    the higher-dimensional studies.
    """
    root = config.data.root
    download = bool(config.data.download)

    # Moved caches often keep raw IDX files without torchvision's processed cache.
    train_data, train_targets = _load_mnist_split(root, train=True, download=download)
    test_data, test_targets = _load_mnist_split(root, train=False, download=download)

    train_count = int(config.data.n_train)
    reference_count = int(config.data.n_test)
    requested_balance_train = bool(getattr(config.data, "balance_train_classes", False))
    requested_balance_reference = bool(
        getattr(config.data, "balance_reference_classes", False)
    )
    balance_train, train_balance_adjustment = _resolve_class_balance(
        requested_balance_train,
        train_count,
        "train",
    )
    balance_reference, reference_balance_adjustment = _resolve_class_balance(
        requested_balance_reference,
        reference_count,
        "reference",
    )
    class_balance_adjustments = [
        adjustment
        for adjustment in (train_balance_adjustment, reference_balance_adjustment)
        if adjustment is not None
    ]

    train_indices = _select_indices(
        train_targets,
        train_count,
        int(config.training.seed),
        balance_train,
        "training subset",
    )
    reference_indices = _select_indices(
        test_targets,
        reference_count,
        int(config.training.seed) + 1,
        balance_reference,
        "reference subset",
    )

    train_tensor = train_data[train_indices].float().div(127.5).sub(1.0).unsqueeze(1)
    reference_tensor = (
        test_data[reference_indices].float().div(127.5).sub(1.0).unsqueeze(1)
    )
    train_labels = train_targets[train_indices].clone()
    reference_labels = test_targets[reference_indices].clone()

    train_tensor = _maybe_resize_images(train_tensor, config)
    reference_tensor = _maybe_resize_images(reference_tensor, config)
    train_tensor = maybe_flatten(train_tensor, config)
    reference_tensor = maybe_flatten(reference_tensor, config)
    train_source_id = "torchvision_mnist:train"
    reference_source_id = "torchvision_mnist:test"
    train_sample_ids = [int(item) for item in train_indices.tolist()]
    reference_sample_ids = [int(item) for item in reference_indices.tolist()]

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "train_class_counts": _class_count_summary(train_labels),
        "requested_balanced_train_classes": requested_balance_train,
        "balanced_train_classes": balance_train,
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "reference_class_counts": _class_count_summary(reference_labels),
        "requested_balanced_reference_classes": requested_balance_reference,
        "balanced_reference_classes": balance_reference,
        "class_balance_adjustments": class_balance_adjustments,
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "train_sample_ids": train_sample_ids,
        "reference_sample_ids": reference_sample_ids,
        "train_sample_id_kind": "index",
        "reference_sample_id_kind": "index",
        "train_source_size": int(train_data.shape[0]),
        "reference_source_size": int(test_data.shape[0]),
        "train_source_catalog_sha256": source_catalog_sha256(
            train_source_id, range(int(train_data.shape[0]))
        ),
        "reference_source_catalog_sha256": source_catalog_sha256(
            reference_source_id, range(int(test_data.shape[0]))
        ),
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_split": "train",
        "reference_split": "test",
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
    }


if __name__ == "__main__":
    from pathlib import Path

    from mem_param_src.config import get_base_config

    print("mem_param_src.datasets.mnist example")

    # Build the smallest MNIST config that still shows the loader contract.
    config = get_base_config()
    config.data.dataset = "mnist"
    config.data.root = "data"
    config.data.n_train = 8
    config.data.n_test = 6
    config.data.image_size = 32
    config.data.image_channels = 1
    config.data.flatten = False
    config.data.download = False

    # Show the direct call pattern before attempting any local filesystem use.
    print("Usage:")
    print("  config = get_base_config()")
    print("  config.data.dataset = 'mnist'")
    print("  bundle = load_mnist_bundle(config)")

    # Skip cleanly when local data is absent. This example should stay safe to
    # run on a fresh machine without triggering a download.
    if not Path(config.data.root).exists():
        print(f"Skipping runnable demo: {config.data.root!r} does not exist.")
    else:
        try:
            # Load one tiny bundle and print the tensor shapes we expect.
            bundle = load_mnist_bundle(config)
            print(f"train shape     : {tuple(bundle['train'].shape)}")
            print(f"reference shape : {tuple(bundle['reference'].shape)}")
        except Exception as exc:
            print(f"Skipping runnable demo: {exc}")
