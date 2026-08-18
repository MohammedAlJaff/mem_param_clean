"""
Canonical loaders for external image collections used in research runs.

The datasets here share the same downstream tensor contract as MNIST and
CelebA32: deterministic train/reference subsets, optional labels, source
indices, display shape metadata, and image tensors normalized to `[-1, 1]`.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from mem_param_src.datasets.common import maybe_flatten, seeded_indices


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_channels(config):
    """
    Read and validate the configured image channel count.
    """
    channels = int(config.data.image_channels)
    if channels not in (1, 3):
        raise ValueError(
            "Image collection loaders support 1 or 3 channels, "
            f"got config.data.image_channels={channels}"
        )
    return channels


def _build_transform(config):
    """
    Build a square image transform for external image datasets.

    The transform returns tensors in `[-1, 1]`, matching the model input range
    used by the image runner.
    """
    image_size = int(config.data.image_size)
    if image_size <= 0:
        raise ValueError(
            "config.data.image_size must be positive for image collections, "
            f"got {image_size}"
        )

    channels = _image_channels(config)
    resize_size = int(getattr(config.data, "resize_size", 0) or image_size)
    mean = (0.5,) * channels
    std = (0.5,) * channels

    ops = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
    ]
    if channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    else:
        ops.append(transforms.Lambda(lambda image: image.convert("RGB")))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


def _stack_subset(dataset, indices):
    """
    Materialize transformed samples selected by integer indices.

    The training loop expects a dense tensor, not a lazy torchvision dataset.
    This also makes the exact selected subset available as a saved artifact.
    """
    images = []
    labels = []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), torch.as_tensor(labels, dtype=torch.long)


def _class_count_summary(labels):
    """
    Convert arbitrary integer label counts into a JSON-friendly dict.
    """
    if labels.numel() == 0:
        return {}
    counts = torch.bincount(labels.to(torch.int64))
    return {
        str(class_id): int(counts[class_id].item())
        for class_id in range(counts.numel())
        if int(counts[class_id].item()) > 0
    }


def _resolve_existing_dir(root, candidates, dataset_name):
    """
    Return the first existing candidate directory under `root`.
    """
    root = Path(root)
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    formatted = "\n".join(f"  - {root / candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find local files for {dataset_name}. Checked:\n{formatted}"
    )


class RecursiveImageDataset(Dataset):
    """
    Minimal image dataset for folders without class subdirectories.

    FFHQ thumbnails are treated as one unlabeled image collection. Labels are
    fixed to zero because class labels are not part of that study.
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


def load_flowers102_bundle(config):
    """
    Return a deterministic Oxford Flowers102 train/reference bundle.

    The runner profiles use the official large test split for training and the
    official train split as held-out reference data. This gives enough training
    images while keeping train/reference sources distinct.
    """
    root = config.data.root
    transform = _build_transform(config)
    train_split = str(getattr(config.data, "train_split", "train"))
    reference_split = str(getattr(config.data, "reference_split", "val"))
    if reference_split == "valid":
        reference_split = "val"

    train_set = datasets.Flowers102(
        root=root,
        split=train_split,
        transform=transform,
        download=bool(config.data.download),
    )
    reference_set = datasets.Flowers102(
        root=root,
        split=reference_split,
        transform=transform,
        download=bool(config.data.download),
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

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "train_class_counts": _class_count_summary(train_labels),
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "reference_class_counts": _class_count_summary(reference_labels),
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": f"torchvision_flowers102:{train_split}",
        "reference_source_id": f"torchvision_flowers102:{reference_split}",
        "train_split": train_split,
        "reference_split": reference_split,
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
    }


def load_afhq_v2_bundle(config):
    """
    Return a deterministic AFHQ-v2 train/reference bundle.

    AFHQ-v2 is read from local ImageFolder-style splits. The returned bundle
    follows the same tensor/label/index contract as torchvision datasets.
    """
    root = config.data.root
    subdir = str(getattr(config.data, "dataset_subdir", "") or "afhq_v2")
    base_dir = _resolve_existing_dir(
        root,
        (
            subdir,
            "afhq_v2",
            "external/stargan-v2/data",
        ),
        "AFHQ-v2",
    )
    train_split = str(getattr(config.data, "train_split", "train"))
    reference_split = str(getattr(config.data, "reference_split", "test"))
    if reference_split in ("valid", "val"):
        reference_split = "test"
    transform = _build_transform(config)

    train_set = datasets.ImageFolder(base_dir / train_split, transform=transform)
    reference_set = datasets.ImageFolder(base_dir / reference_split, transform=transform)

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

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "train_class_counts": _class_count_summary(train_labels),
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "reference_class_counts": _class_count_summary(reference_labels),
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "train_role": "train",
        "reference_role": "heldout_reference",
        "train_source_id": f"afhq_v2:{train_split}",
        "reference_source_id": f"afhq_v2:{reference_split}",
        "train_split": train_split,
        "reference_split": reference_split,
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
        "class_to_idx": dict(train_set.class_to_idx),
    }


def load_ffhq128_bundle(config):
    """
    Return a deterministic FFHQ thumbnail bundle.

    FFHQ does not ship with train/validation splits. We therefore draw one
    disjoint seeded partition from the full image collection and mark the
    bundle as a same-source holdout through matching source ids. The common
    leakage guard then verifies that the saved train/reference indices are
    disjoint.
    """
    root = config.data.root
    subdir = str(
        getattr(config.data, "dataset_subdir", "")
        or "ffhq/thumbnails128x128"
    )
    base_dir = _resolve_existing_dir(
        root,
        (
            subdir,
            "ffhq/thumbnails128x128",
            "ffhq-dataset/thumbnails128x128",
            "external/ffhq-dataset/thumbnails128x128",
        ),
        "FFHQ thumbnails",
    )
    dataset = RecursiveImageDataset(base_dir, transform=_build_transform(config))
    count = int(config.data.n_train) + int(config.data.n_test)
    combined = seeded_indices(len(dataset), count, int(config.training.seed))
    train_indices = combined[: int(config.data.n_train)]
    reference_indices = combined[int(config.data.n_train) :]

    train_tensor, train_labels = _stack_subset(dataset, train_indices)
    reference_tensor, reference_labels = _stack_subset(dataset, reference_indices)

    train_tensor = maybe_flatten(train_tensor, config)
    reference_tensor = maybe_flatten(reference_tensor, config)

    return {
        "train": train_tensor.contiguous(),
        "train_labels": train_labels,
        "train_class_counts": _class_count_summary(train_labels),
        "reference": reference_tensor.contiguous(),
        "reference_labels": reference_labels,
        "reference_class_counts": _class_count_summary(reference_labels),
        "train_indices": train_indices,
        "reference_indices": reference_indices,
        "train_role": "train",
        "reference_role": "same_corpus_holdout",
        "train_source_id": "ffhq128:all",
        "reference_source_id": "ffhq128:all",
        "train_split": "seeded_partition",
        "reference_split": "seeded_partition",
        "display_shape": (
            int(config.data.image_channels),
            int(config.data.image_size),
            int(config.data.image_size),
        ),
    }


if __name__ == "__main__":
    print("mem_param_src.datasets.image_collections example")
    print("supported loaders: flowers102, afhq_v2, ffhq128")
