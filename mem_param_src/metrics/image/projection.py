"""Random-projected pixel features and memorization metrics."""

import torch
import torch.nn.functional as F

from mem_param_src.metrics.image.preprocessing import (
    as_image_batch,
    device_from_arg,
    normalize_images,
)
from mem_param_src.metrics.nearest_neighbors import compute_memorization_metrics


def make_random_projection_matrix(input_dim, projection_dim, projection_seed):
    """Draw a fixed Gaussian `[input_dim, projection_dim]` projection."""
    input_dim = int(input_dim)
    projection_dim = int(projection_dim)
    if input_dim < 1:
        raise ValueError(f"input_dim must be at least 1, got {input_dim}")
    if projection_dim < 1:
        raise ValueError(f"projection_dim must be at least 1, got {projection_dim}")
    generator = torch.Generator().manual_seed(int(projection_seed))
    matrix = torch.randn(input_dim, projection_dim, generator=generator)
    return matrix / float(projection_dim) ** 0.5


def _resolve_batch_size(value, name):
    """Return a positive batch size or `None`."""
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _resize_flatten(images, *, resize_size, image_range, device):
    """Return flattened, resized `[0, 1]` image pixels."""
    resize_size = int(resize_size)
    if resize_size < 1:
        raise ValueError(f"resize_size must be at least 1, got {resize_size}")
    images = normalize_images(images, image_range=image_range)
    images = images.to(device_from_arg(device))
    if int(images.shape[-2]) != resize_size or int(images.shape[-1]) != resize_size:
        images = F.interpolate(
            images,
            size=(resize_size, resize_size),
            mode="bilinear",
            align_corners=False,
        )
    return images.flatten(1).float()


def _resolve_projection_matrix(
    input_dim,
    projection_dim,
    projection_seed,
    projection_matrix,
    device,
):
    """Validate and place a projection matrix on the metric device."""
    if projection_matrix is None:
        if projection_seed is None:
            raise ValueError(
                "projection_seed is required when projection_matrix is not provided."
            )
        projection_matrix = make_random_projection_matrix(
            input_dim,
            projection_dim,
            projection_seed,
        )
    if not torch.is_tensor(projection_matrix):
        projection_matrix = torch.as_tensor(projection_matrix, dtype=torch.float32)
    projection_matrix = projection_matrix.float()
    if projection_matrix.dim() != 2:
        raise ValueError(
            "projection_matrix must have shape [input_dim, projection_dim], "
            f"got {tuple(projection_matrix.shape)}"
        )
    if int(projection_matrix.shape[0]) != int(input_dim):
        raise ValueError(
            "projection_matrix input dimension does not match resized pixels: "
            f"{int(projection_matrix.shape[0])} != {int(input_dim)}"
        )
    if projection_dim is not None and int(projection_matrix.shape[1]) != int(
        projection_dim
    ):
        raise ValueError(
            "projection_matrix output dimension does not match projection_dim: "
            f"{int(projection_matrix.shape[1])} != {int(projection_dim)}"
        )
    return projection_matrix.to(device_from_arg(device))


def compute_projected_image_features(
    images,
    *,
    resize_size=16,
    projection_dim=256,
    projection_seed=None,
    projection_matrix=None,
    image_range="auto",
    device="cpu",
    batch_size=None,
):
    """Resize images, flatten pixels, and apply one fixed projection."""
    image_batch = as_image_batch(images)
    if int(image_batch.shape[0]) < 1:
        raise ValueError("Projected image features require at least one image.")

    batch_size = _resolve_batch_size(batch_size, "batch_size")
    if batch_size is None:
        pixels = _resize_flatten(
            image_batch,
            resize_size=resize_size,
            image_range=image_range,
            device=device,
        )
        projection = _resolve_projection_matrix(
            int(pixels.shape[1]),
            projection_dim,
            projection_seed,
            projection_matrix,
            device,
        )
        return pixels @ projection

    features = []
    projection = None
    input_dim = None
    for start in range(0, int(image_batch.shape[0]), batch_size):
        pixels = _resize_flatten(
            image_batch[start : start + batch_size],
            resize_size=resize_size,
            image_range=image_range,
            device=device,
        )
        if projection is None:
            input_dim = int(pixels.shape[1])
            projection = _resolve_projection_matrix(
                input_dim,
                projection_dim,
                projection_seed,
                projection_matrix,
                device,
            )
        elif int(pixels.shape[1]) != input_dim:
            raise ValueError(
                "Projected image feature chunks produced inconsistent dimensions: "
                f"{int(pixels.shape[1])} != {input_dim}"
            )
        features.append(pixels @ projection)
    return torch.cat(features, dim=0)


def _resolve_reference_vectors(vectors, feature_dim, device):
    """Validate cached projected reference vectors."""
    if not torch.is_tensor(vectors):
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
    vectors = vectors.detach().float()
    if vectors.dim() != 2:
        raise ValueError(
            "projected_reference_vectors must have shape [N, feature_dim], "
            f"got {tuple(vectors.shape)}"
        )
    if int(vectors.shape[1]) != int(feature_dim):
        raise ValueError(
            "projected_reference_vectors feature dimension does not match samples: "
            f"{int(vectors.shape[1])} != {int(feature_dim)}"
        )
    return vectors.to(device_from_arg(device))


def compute_projected_image_memorization_metrics(
    samples,
    reference=None,
    *,
    projected_reference_vectors=None,
    resize_size=16,
    projection_dim=256,
    projection_seed=None,
    projection_matrix=None,
    rho=None,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """Compute memorization in fixed random-projected resized pixels."""
    query_batch_size = int(query_batch_size)
    if query_batch_size < 1:
        raise ValueError(f"query_batch_size must be at least 1, got {query_batch_size}")
    sample_vectors = compute_projected_image_features(
        samples,
        resize_size=resize_size,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        projection_matrix=projection_matrix,
        image_range=image_range,
        device=device,
    )
    if projected_reference_vectors is None:
        if reference is None:
            raise ValueError(
                "reference is required when projected_reference_vectors is not provided."
            )
        reference_vectors = compute_projected_image_features(
            reference,
            resize_size=resize_size,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            projection_matrix=projection_matrix,
            image_range=image_range,
            device=device,
        )
        projected_reference_cached = False
    else:
        reference_vectors = _resolve_reference_vectors(
            projected_reference_vectors,
            int(sample_vectors.shape[1]),
            device,
        )
        projected_reference_cached = True

    metrics = compute_memorization_metrics(
        sample_vectors,
        reference_vectors,
        rho=rho,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    metrics.update(
        {
            "space": "random_projected_pixels",
            "resize_size": int(resize_size),
            "projection_dim": int(sample_vectors.shape[1]),
            "projection_seed": (
                None if projection_seed is None else int(projection_seed)
            ),
            "image_range": str(image_range),
            "query_batch_size": query_batch_size,
            "reference_batch_size": (
                None if reference_batch_size is None else int(reference_batch_size)
            ),
            "feature_dim": int(sample_vectors.shape[1]),
            "input_dim": (
                int(as_image_batch(samples).shape[1])
                * int(resize_size)
                * int(resize_size)
            ),
            "projected_reference_cached": bool(projected_reference_cached),
        }
    )
    return metrics


__all__ = [
    "compute_projected_image_features",
    "compute_projected_image_memorization_metrics",
    "make_random_projection_matrix",
]
