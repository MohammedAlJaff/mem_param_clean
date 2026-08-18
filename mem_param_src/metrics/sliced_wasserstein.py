"""
Sliced Wasserstein metrics for vector data.

This module does not know about images or synthetic geometries. Callers must
choose the metric space first, then pass two `[N, D]` vector batches here.
"""

import torch


def _as_matrix(vectors, name):
    """Convert input to a float matrix with shape `[N, D]`."""
    vectors = torch.as_tensor(vectors).float()
    if vectors.dim() == 1:
        return vectors[:, None]
    if vectors.dim() == 2:
        return vectors
    if vectors.dim() > 2:
        return vectors.flatten(1)
    raise ValueError(f"{name} must have at least one dimension.")


def normalize_directions(directions):
    """Return unit projection directions."""
    directions = _as_matrix(directions, "directions")
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1.0e-12)


def sliced_w2(samples, reference, directions):
    """
    Compute mean squared 1D Wasserstein distance over fixed directions.

    The two sample counts can differ. Each projection uses the largest shared
    sorted prefix.
    """
    samples = _as_matrix(samples, "samples")
    reference = _as_matrix(reference, "reference").to(samples.device)
    directions = normalize_directions(directions).to(samples.device)
    if samples.shape[1] != reference.shape[1]:
        raise ValueError(
            "samples and reference must have the same vector dimension, "
            f"got {samples.shape[1]} and {reference.shape[1]}"
        )
    if directions.shape[1] != samples.shape[1]:
        raise ValueError(
            "directions must match the vector dimension, "
            f"got {directions.shape[1]} and {samples.shape[1]}"
        )

    sample_proj = samples @ directions.T
    reference_proj = reference @ directions.T
    sample_sorted, _ = sample_proj.sort(dim=0)
    reference_sorted, _ = reference_proj.sort(dim=0)
    count = min(sample_sorted.shape[0], reference_sorted.shape[0])
    return float(
        (sample_sorted[:count] - reference_sorted[:count]).pow(2).mean().item()
    )


__all__ = [
    "normalize_directions",
    "sliced_w2",
]
