"""Fréchet Inception Distance for in-memory image tensors."""

import torch
from torch.utils.data import Dataset

from mem_param_src.metrics.image.preprocessing import (
    select_cuda_device,
    to_uint8_rgb,
    uses_cuda,
)

DEFAULT_FID_FEATURE_EXTRACTOR = "inception-v3-compat"


class _TensorImageDataset(Dataset):
    """Adapt an in-memory uint8 RGB batch to torch-fidelity's input contract."""

    def __init__(self, images):
        if not torch.is_tensor(images):
            raise TypeError("images must be a torch.Tensor")
        if images.dtype != torch.uint8:
            raise TypeError(f"FID images must be uint8, got {images.dtype}")
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"FID images must have shape [B, 3, H, W], got {tuple(images.shape)}"
            )
        self.images = images.cpu().contiguous()

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, index):
        return self.images[index]


def compute_image_fid(
    samples,
    reference=None,
    *,
    reference_stats_path=None,
    batch_size=64,
    device="cuda",
    image_range="auto",
    feature_extractor=DEFAULT_FID_FEATURE_EXTRACTOR,
    cache=False,
    verbose=False,
):
    """Compute FID against a reference tensor or saved reference statistics."""
    if reference is None and reference_stats_path is None:
        raise ValueError("Pass either reference or reference_stats_path for FID.")
    if reference is not None and reference_stats_path is not None:
        raise ValueError("Pass only one of reference or reference_stats_path for FID.")

    try:
        import torch_fidelity
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch-fidelity is required for FID metrics."
        ) from exc

    select_cuda_device(device)
    input1 = _TensorImageDataset(to_uint8_rgb(samples, image_range=image_range))
    input2 = None
    reference_source = None
    n_reference = None

    kwargs = {
        "input1": input1,
        "input2": None,
        "fid": True,
        "cuda": uses_cuda(device),
        "batch_size": int(batch_size),
        "feature_extractor": str(feature_extractor),
        "cache": bool(cache),
        "verbose": bool(verbose),
    }

    if reference is not None:
        input2 = _TensorImageDataset(to_uint8_rgb(reference, image_range=image_range))
        kwargs["input2"] = input2
        reference_source = "tensor"
        n_reference = len(input2)
    else:
        # torch-fidelity reads fid_statistics_file only when another metric
        # prevents its FID-only shortcut. ISC is computed and ignored here.
        kwargs["fid_statistics_file"] = str(reference_stats_path)
        kwargs["isc"] = True
        reference_source = f"stats:{reference_stats_path}"

    raw_metrics = torch_fidelity.calculate_metrics(**kwargs)
    fid = float(raw_metrics["frechet_inception_distance"])
    return {
        "fid": fid,
        "frechet_inception_distance": fid,
        "n_samples": len(input1),
        "n_reference": n_reference,
        "feature_extractor": str(feature_extractor),
        "reference_source": reference_source,
    }


__all__ = ["DEFAULT_FID_FEATURE_EXTRACTOR", "compute_image_fid"]
