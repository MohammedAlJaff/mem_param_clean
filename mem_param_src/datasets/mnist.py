"""MNIST source adapter for the shared image-bundle pipeline."""

from pathlib import Path

import torch.nn.functional as F
from torchvision import datasets
from torchvision.datasets.mnist import read_image_file, read_label_file

from mem_param_src.datasets.image import build_image_bundle


MNIST_NUM_CLASSES = 10


def _maybe_resize_images(tensor, config):
    """Resize MNIST tensors to the configured square resolution."""
    target_size = int(config.data.image_size)
    if target_size <= 0:
        raise ValueError(
            f"config.data.image_size must be positive for MNIST, got {target_size}"
        )
    if tuple(tensor.shape[-2:]) == (target_size, target_size):
        return tensor
    return F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def _match_image_channels(tensor, config):
    """Return one or three configured MNIST image channels."""
    channels = int(config.data.image_channels)
    if channels == 1:
        return tensor
    if channels == 3:
        # Replication changes tensor shape but adds no color information.
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError("MNIST supports 1 or 3 image channels.")


def _raw_mnist_paths(root, train):
    """Return raw IDX paths for one MNIST split."""
    raw_dir = Path(root) / "MNIST" / "raw"
    if train:
        return raw_dir / "train-images-idx3-ubyte", raw_dir / "train-labels-idx1-ubyte"
    return raw_dir / "t10k-images-idx3-ubyte", raw_dir / "t10k-labels-idx1-ubyte"


def _load_raw_mnist_split(root, train):
    """Load one MNIST split from raw IDX files."""
    image_path, label_path = _raw_mnist_paths(root, train)
    missing = [str(path) for path in (image_path, label_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return read_image_file(str(image_path)), read_label_file(str(label_path))


def _load_mnist_split(root, train, download):
    """Use Torchvision first and support moved raw-only caches."""
    try:
        dataset = datasets.MNIST(root=root, train=train, download=download)
        return dataset.data, dataset.targets
    except RuntimeError:
        if download:
            raise
        return _load_raw_mnist_split(root, train)


def _materialize_subset(data, targets, indices, config):
    """Convert one fixed MNIST subset into configured image tensors."""
    tensor = data[indices].float().div(127.5).sub(1.0).unsqueeze(1)
    tensor = _maybe_resize_images(tensor, config)
    tensor = _match_image_channels(tensor, config)
    return tensor, targets[indices].clone()


def load_mnist_bundle(config):
    """Open MNIST and delegate shared selection and provenance work."""
    root = config.data.root
    download = bool(config.data.download)
    train_data, train_targets = _load_mnist_split(root, train=True, download=download)
    test_data, test_targets = _load_mnist_split(root, train=False, download=download)

    return build_image_bundle(
        config,
        {
            "train": {
                "dataset": train_data,
                "targets": train_targets,
                "source_id": "torchvision_mnist:train",
            },
            "reference": {
                "dataset": test_data,
                "targets": test_targets,
                "source_id": "torchvision_mnist:test",
            },
            "train_split": "train",
            "reference_split": "test",
            "num_classes": MNIST_NUM_CLASSES,
            "balance_supported": True,
            "materialize_subset": _materialize_subset,
        },
    )
