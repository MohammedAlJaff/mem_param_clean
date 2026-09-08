"""Source adapters for Flowers102, AFHQ-v2, and FFHQ128."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from mem_param_src.datasets.common import relative_path_sample_ids
from mem_param_src.datasets.image import build_image_bundle


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_transform(config):
    """Build the shared deterministic image-collection transform."""
    image_size = int(config.data.image_size)
    if image_size <= 0:
        raise ValueError(
            "config.data.image_size must be positive for image collections, "
            f"got {image_size}"
        )

    channels = int(config.data.image_channels)
    if channels not in (1, 3):
        raise ValueError(
            "Image collection loaders support 1 or 3 channels, "
            f"got config.data.image_channels={channels}"
        )

    resize_size = int(getattr(config.data, "resize_size", 0) or image_size)
    operations = [
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
    ]
    if channels == 1:
        operations.append(transforms.Grayscale(num_output_channels=1))
    else:
        operations.append(transforms.Lambda(lambda image: image.convert("RGB")))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels),
        ]
    )
    return transforms.Compose(operations)


def _resolve_existing_dir(root, candidates, dataset_name):
    """Return the first existing source directory."""
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
    """Read one unlabeled recursive image directory in stable path order."""

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
    """Open the configured Flowers102 splits and use the shared pipeline."""
    train_split = str(getattr(config.data, "train_split", "train"))
    reference_split = str(getattr(config.data, "reference_split", "val"))
    if reference_split == "valid":
        reference_split = "val"
    transform = _build_transform(config)
    train_set = datasets.Flowers102(
        root=config.data.root,
        split=train_split,
        transform=transform,
        download=bool(config.data.download),
    )
    reference_set = datasets.Flowers102(
        root=config.data.root,
        split=reference_split,
        transform=transform,
        download=bool(config.data.download),
    )

    return build_image_bundle(
        config,
        {
            "train": {
                "dataset": train_set,
                "source_id": f"torchvision_flowers102:{train_split}",
            },
            "reference": {
                "dataset": reference_set,
                "source_id": f"torchvision_flowers102:{reference_split}",
            },
            "train_split": train_split,
            "reference_split": reference_split,
            "num_classes": 102,
            "include_empty_classes": False,
        },
    )


def load_afhq_v2_bundle(config):
    """Open local AFHQ-v2 folders and use the shared pipeline."""
    subdir = str(getattr(config.data, "dataset_subdir", "") or "afhq_v2")
    base_dir = _resolve_existing_dir(
        config.data.root,
        (subdir, "afhq_v2", "external/stargan-v2/data"),
        "AFHQ-v2",
    )
    train_split = str(getattr(config.data, "train_split", "train"))
    reference_split = str(getattr(config.data, "reference_split", "test"))
    if reference_split in ("valid", "val"):
        reference_split = "test"
    transform = _build_transform(config)
    train_set = datasets.ImageFolder(base_dir / train_split, transform=transform)
    reference_set = datasets.ImageFolder(
        base_dir / reference_split,
        transform=transform,
    )
    train_catalog = relative_path_sample_ids(
        (path for path, _ in train_set.samples),
        base_dir / train_split,
    )
    reference_catalog = relative_path_sample_ids(
        (path for path, _ in reference_set.samples),
        base_dir / reference_split,
    )

    return build_image_bundle(
        config,
        {
            "train": {
                "dataset": train_set,
                "catalog": train_catalog,
                "sample_id_kind": "relative_path",
                "source_id": f"afhq_v2:{train_split}",
            },
            "reference": {
                "dataset": reference_set,
                "catalog": reference_catalog,
                "sample_id_kind": "relative_path",
                "source_id": f"afhq_v2:{reference_split}",
            },
            "train_split": train_split,
            "reference_split": reference_split,
            "num_classes": len(train_set.classes),
            "include_empty_classes": False,
            "class_to_idx": dict(train_set.class_to_idx),
        },
    )


def load_ffhq128_bundle(config):
    """Open FFHQ thumbnails and create one disjoint same-source holdout."""
    subdir = str(getattr(config.data, "dataset_subdir", "") or "ffhq/thumbnails128x128")
    base_dir = _resolve_existing_dir(
        config.data.root,
        (
            subdir,
            "ffhq/thumbnails128x128",
            "ffhq-dataset/thumbnails128x128",
            "external/ffhq-dataset/thumbnails128x128",
        ),
        "FFHQ thumbnails",
    )
    dataset = RecursiveImageDataset(base_dir, transform=_build_transform(config))
    catalog = relative_path_sample_ids(dataset.files, base_dir)
    source = {
        "dataset": dataset,
        "catalog": catalog,
        "sample_id_kind": "relative_path",
        "source_id": "ffhq128:all",
    }

    return build_image_bundle(
        config,
        {
            "train": source,
            "reference": source,
            "train_split": "seeded_partition",
            "reference_split": "seeded_partition",
            "reference_role": "same_corpus_holdout",
            "shared_partition": True,
            "num_classes": 1,
            "include_empty_classes": False,
        },
    )
