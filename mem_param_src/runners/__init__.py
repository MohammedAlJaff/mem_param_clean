"""Shared boundaries for the clean image and synthetic runners."""

import torch


IMAGE_DATASETS = (
    "mnist",
    "cifar10",
    "celeba32_grey",
    "flowers102",
    "afhq_v2",
    "ffhq128",
)
SYNTHETIC_DATASETS = ("gmm_subspace", "torus3d")


def validate_run_kind(config, run_kind):
    """Check that a domain runner received a dataset from its domain."""
    dataset = str(config.data.dataset)
    choices = IMAGE_DATASETS if run_kind == "image" else SYNTHETIC_DATASETS
    if run_kind not in ("image", "synthetic"):
        raise ValueError(f"Unknown runner kind: {run_kind!r}")
    if dataset not in choices:
        raise ValueError(
            f"The {run_kind} runner does not accept dataset {dataset!r}. "
            f"Expected one of: {', '.join(choices)}."
        )


def resolve_device(name):
    """Resolve ``auto`` or one explicit PyTorch device."""
    name = str(name)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA is not available for requested device {name!r}.")
    if device.type == "mps":
        available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not available:
            raise ValueError("MPS is not available on this host.")
    return device


def configure_runtime(config, device):
    """Validate the first runner runtime and apply its CUDA TF32 setting."""
    if str(config.runtime.precision) != "fp32":
        raise ValueError("The first runner supports runtime.precision='fp32' only.")
    if str(config.runtime.autocast_dtype) != "none":
        raise ValueError("The first runner does not use autocast.")
    if device.type == "cuda":
        tf32_enabled = bool(config.runtime.tf32_enabled)
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled


__all__ = [
    "IMAGE_DATASETS",
    "SYNTHETIC_DATASETS",
    "configure_runtime",
    "resolve_device",
    "validate_run_kind",
]
