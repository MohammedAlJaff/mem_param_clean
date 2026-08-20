"""
Canonical CelebA-32 grayscale loader for first-party studies.

This module mirrors the image preprocessing used by the active CelebA32 study:
center crop, resize to 32x32, convert to grayscale, and normalize to `[-1, 1]`.
The returned bundle keeps the same train/reference contract as the other
dataset loaders so the training and evaluation code can stay generic.
"""

import torch
from torchvision import datasets, transforms

from mem_param_src.datasets.common import (
    maybe_flatten,
    seeded_indices,
    source_catalog_sha256,
)


def _build_transform():
    """
    Build the exact image transform used by the canonical CelebA32 study.
    """
    return transforms.Compose(
        [
            transforms.CenterCrop(140),
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _stack_subset(dataset, indices):
    """
    Materialize a tensor batch and label vector from a list of dataset indices.

    CelebA examples are transformed lazily by torchvision, so we explicitly
    iterate through the selected subset and stack the results into a dense
    tensor for downstream training code.
    """
    images = []
    labels = []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.as_tensor(labels)


def load_celeba32_bundle(config):
    """
    Return a deterministic CelebA32 train/reference bundle.

    The training split is always `train`; the reference split is configurable
    so a study can compare against validation or test images without changing
    the rest of the data-loading logic. Even though the raw dataset is more
    complex than MNIST or GMM, the output bundle deliberately has the same
    shape so the rest of the study pipeline can stay dataset-independent.
    """
    root = config.data.root
    download = bool(config.data.download)
    transform = _build_transform()

    train_set = datasets.CelebA(
        root=root,
        split="train",
        target_type="identity",
        transform=transform,
        download=download,
    )
    reference_set = datasets.CelebA(
        root=root,
        split=config.data.reference_split,
        target_type="identity",
        transform=transform,
        download=download,
    )

    train_indices = seeded_indices(
        len(train_set), int(config.data.n_train), int(config.training.seed)
    )
    reference_indices = seeded_indices(
        len(reference_set), int(config.data.n_test), int(config.training.seed) + 1
    )

    train_tensor, train_labels = _stack_subset(train_set, train_indices)
    reference_tensor, reference_labels = _stack_subset(reference_set, reference_indices)

    train_tensor = maybe_flatten(train_tensor, config)
    reference_tensor = maybe_flatten(reference_tensor, config)
    train_source_id = "torchvision_celeba:train"
    reference_source_id = f"torchvision_celeba:{config.data.reference_split}"
    train_sample_ids = [int(item) for item in train_indices.tolist()]
    reference_sample_ids = [int(item) for item in reference_indices.tolist()]

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "train_sample_ids": train_sample_ids,
        "reference_sample_ids": reference_sample_ids,
        "train_sample_id_kind": "index",
        "reference_sample_id_kind": "index",
        "train_source_size": len(train_set),
        "reference_source_size": len(reference_set),
        "train_source_catalog_sha256": source_catalog_sha256(
            train_source_id, range(len(train_set))
        ),
        "reference_source_catalog_sha256": source_catalog_sha256(
            reference_source_id, range(len(reference_set))
        ),
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_split": "train",
        "reference_split": str(config.data.reference_split),
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
    }


if __name__ == "__main__":
    from pathlib import Path

    from mem_param_src.config import get_base_config

    print("mem_param_src.datasets.celeba32 example")

    # Build a tiny grayscale CelebA32 config with the same fields a real run uses.
    config = get_base_config()
    config.data.dataset = "celeba32_grey"
    config.data.root = "data"
    config.data.n_train = 4
    config.data.n_test = 4
    config.data.image_size = 32
    config.data.image_channels = 1
    config.data.reference_split = "valid"
    config.data.download = False

    # Show the direct loader call before we check whether local data exists.
    print("Usage:")
    print("  config = get_base_config()")
    print("  config.data.dataset = 'celeba32_grey'")
    print("  bundle = load_celeba32_bundle(config)")

    # Skip cleanly when the local dataset is absent. This keeps the example
    # readable without forcing a download.
    if not Path(config.data.root).exists():
        print(f"Skipping runnable demo: {config.data.root!r} does not exist.")
    else:
        try:
            # Load one tiny bundle and print the image-tensor contract.
            bundle = load_celeba32_bundle(config)
            print(f"train shape     : {tuple(bundle['train'].shape)}")
            print(f"reference shape : {tuple(bundle['reference'].shape)}")
        except Exception as exc:
            print(f"Skipping runnable demo: {exc}")
