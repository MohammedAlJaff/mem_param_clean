"""CIFAR-10 source adapter for the shared image-bundle pipeline."""

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms.functional import rgb_to_grayscale

from mem_param_src.datasets.image import build_image_bundle


CIFAR10_NUM_CLASSES = 10


def _resize_images(tensor, config):
    """Resize CIFAR-10 tensors to the configured square resolution."""
    image_size = int(config.data.image_size)
    if image_size < 1:
        raise ValueError("config.data.image_size must be positive for CIFAR-10.")
    if tuple(tensor.shape[-2:]) == (image_size, image_size):
        return tensor
    return F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _match_image_channels(tensor, config):
    """Return configured RGB or grayscale CIFAR-10 tensors."""
    channels = int(config.data.image_channels)
    if channels == 3:
        return tensor
    if channels == 1:
        return rgb_to_grayscale(tensor, num_output_channels=1)
    raise ValueError("CIFAR-10 supports 1 or 3 image channels.")


def _materialize_subset(dataset, targets, indices, config):
    """Convert one fixed CIFAR-10 subset into configured image tensors."""
    images = torch.from_numpy(dataset.data[indices.numpy()])
    images = images.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    images = _resize_images(images, config)
    images = _match_image_channels(images, config)
    return images, targets[indices]


def load_cifar10_bundle(config):
    """Open CIFAR-10 and delegate shared selection and provenance work."""
    root = config.data.root
    download = bool(config.data.download)
    train_set = datasets.CIFAR10(root=root, train=True, download=download)
    reference_set = datasets.CIFAR10(root=root, train=False, download=download)
    train_targets = torch.as_tensor(train_set.targets, dtype=torch.long)
    reference_targets = torch.as_tensor(reference_set.targets, dtype=torch.long)

    return build_image_bundle(
        config,
        {
            "train": {
                "dataset": train_set,
                "targets": train_targets,
                "source_id": "torchvision_cifar10:train",
            },
            "reference": {
                "dataset": reference_set,
                "targets": reference_targets,
                "source_id": "torchvision_cifar10:test",
            },
            "train_split": "train",
            "reference_split": "test",
            "num_classes": CIFAR10_NUM_CLASSES,
            "balance_supported": True,
            "materialize_subset": _materialize_subset,
            "class_to_idx": dict(train_set.class_to_idx),
        },
    )
