"""Image tensor layout, range, and device conversions."""

import warnings

import torch


def as_image_batch(images):
    """Normalize common image tensor layouts to `[B, C, H, W]`."""
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    images = images.detach()

    if images.dim() == 2:
        return images.unsqueeze(0).unsqueeze(0)
    if images.dim() == 3:
        if images.shape[0] in (1, 3, 4):
            return images.unsqueeze(0)
        if images.shape[-1] in (1, 3, 4):
            return images.permute(2, 0, 1).unsqueeze(0)
        return images.unsqueeze(1)
    if images.dim() == 4:
        if images.shape[1] in (1, 3, 4):
            return images
        if images.shape[-1] in (1, 3, 4):
            return images.permute(0, 3, 1, 2)

    raise ValueError(
        "Expected image tensor shaped [B, C, H, W], [B, H, W], [C, H, W], "
        f"or [H, W], got {tuple(images.shape)}"
    )


def normalize_images(images, image_range="auto"):
    """Convert image tensors to float `[0, 1]`."""
    images = as_image_batch(images).float()
    mode = str(image_range).lower().replace("-", "_")

    if mode in {"0_1", "zero_one", "unit"}:
        return images.clamp(0.0, 1.0)
    if mode in {"minus_one_one", "neg_one_one", "_1_1"}:
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    if mode in {"0_255", "uint8", "byte"}:
        return (images / 255.0).clamp(0.0, 1.0)
    if mode != "auto":
        raise ValueError(
            f"image_range must be one of auto, 0_1, -1_1, or 0_255; got {image_range!r}"
        )

    if images.numel() == 0:
        raise ValueError("Image metrics require at least one image.")

    min_value = float(images.min().item())
    max_value = float(images.max().item())
    warnings.warn(
        "image_range='auto' inferred range from data "
        f"(min={min_value:.4g}, max={max_value:.4g}). "
        "Set image_range explicitly for reproducibility.",
        stacklevel=2,
    )
    if min_value >= 0.0 and max_value <= 1.0:
        return images.clamp(0.0, 1.0)
    if min_value >= -1.0 and max_value <= 1.0:
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    if min_value >= 0.0 and max_value <= 255.0:
        return (images / 255.0).clamp(0.0, 1.0)

    raise ValueError(
        "Could not infer image range. Pass image_range='0_1', '-1_1', "
        f"or '0_255'. Observed min={min_value:.4g}, max={max_value:.4g}."
    )


def to_uint8_rgb(images, image_range="auto"):
    """Convert images to uint8 RGB `[B, 3, H, W]`."""
    images = normalize_images(images, image_range=image_range)
    channels = int(images.shape[1])
    if channels == 1:
        images = images.repeat(1, 3, 1, 1)
    elif channels == 4:
        images = images[:, :3]
    elif channels != 3:
        raise ValueError(f"Expected 1, 3, or 4 image channels, got {channels}")

    return torch.round(images * 255.0).clamp(0, 255).to(torch.uint8).cpu()


def device_from_arg(device):
    """Convert a device argument into `torch.device`."""
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def uses_cuda(device):
    """Return whether a device argument selects CUDA."""
    return device_from_arg(device).type == "cuda"


def select_cuda_device(device):
    """Select an indexed CUDA device before torch-fidelity calls."""
    torch_device = device_from_arg(device)
    if torch_device.type == "cuda" and torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)
    return torch_device


__all__ = ["as_image_batch", "normalize_images", "to_uint8_rgb"]
