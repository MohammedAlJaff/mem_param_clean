"""Aligned CelebA source adapter for the shared image-bundle pipeline."""

from torchvision import datasets, transforms

from mem_param_src.datasets.image import build_image_bundle


def _build_transform(config):
    """Preserve the historical crop before configured image conversion."""
    image_size = int(config.data.image_size)
    if image_size < 1:
        raise ValueError("config.data.image_size must be positive for CelebA.")

    channels = int(config.data.image_channels)
    if channels not in (1, 3):
        raise ValueError("CelebA supports 1 or 3 image channels.")

    resize_size = int(getattr(config.data, "resize_size", 0) or image_size)
    if resize_size < image_size:
        raise ValueError("config.data.resize_size must be at least data.image_size.")

    operations = [
        # This crop keeps new studies comparable with historical CelebA32 data.
        transforms.CenterCrop(140),
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


def load_celeba32_bundle(config):
    """Open CelebA and delegate shared selection and provenance work."""
    root = config.data.root
    reference_split = str(config.data.reference_split)
    transform = _build_transform(config)
    train_set = datasets.CelebA(
        root=root,
        split="train",
        target_type="identity",
        transform=transform,
        download=bool(config.data.download),
    )
    reference_set = datasets.CelebA(
        root=root,
        split=reference_split,
        target_type="identity",
        transform=transform,
        download=bool(config.data.download),
    )

    return build_image_bundle(
        config,
        {
            "train": {
                "dataset": train_set,
                "source_id": "torchvision_celeba:train",
            },
            "reference": {
                "dataset": reference_set,
                "source_id": f"torchvision_celeba:{reference_split}",
            },
            "train_split": "train",
            "reference_split": reference_split,
            "include_class_counts": False,
        },
    )
