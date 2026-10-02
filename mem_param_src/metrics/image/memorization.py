"""Pixel- and encoder-space image memorization metrics."""

import torch

from mem_param_src.metrics.image.features import build_metric_vectors
from mem_param_src.metrics.nearest_neighbors import (
    compute_f_mem,
    compute_memorization_metrics,
)


REFERENCE_QUANTILES = (0.005, 0.01, 0.02, 0.05)


def compute_f_mem_by_rho(r, rho_values):
    """Compute `f_mem` for each nearest-neighbor ratio threshold."""
    r = torch.as_tensor(r, dtype=torch.float32)
    if isinstance(rho_values, (int, float)):
        rho_values = [rho_values]
    return {float(rho): compute_f_mem(r, rho=float(rho)) for rho in rho_values}


def compute_f_mem_reference_curve(
    generated_r,
    calibration_r,
    validation_r=None,
    quantiles=REFERENCE_QUANTILES,
):
    """Derive f_mem at thresholds fixed by held-out ratio quantiles."""
    generated_r = torch.as_tensor(generated_r, dtype=torch.float32).flatten()
    calibration_r = torch.as_tensor(calibration_r, dtype=torch.float32).flatten()
    if generated_r.numel() == 0 or calibration_r.numel() == 0:
        raise ValueError("Generated and calibration ratio tensors must be nonempty.")
    if not bool(torch.isfinite(generated_r).all()) or not bool(
        torch.isfinite(calibration_r).all()
    ):
        raise ValueError("Generated and calibration ratios must be finite.")

    if validation_r is not None:
        validation_r = torch.as_tensor(validation_r, dtype=torch.float32).flatten()
        if validation_r.numel() == 0 or not bool(torch.isfinite(validation_r).all()):
            raise ValueError("Validation ratios must be nonempty and finite.")

    records = []
    for quantile in quantiles:
        quantile = float(quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"Held-out quantiles must be between zero and one, got {quantile}."
            )
        rho = float(torch.quantile(calibration_r, quantile).item())
        record = {
            "heldout_quantile": quantile,
            "rho": rho,
            "calibration_rate": compute_f_mem(calibration_r, rho=rho),
            "generated_f_mem": compute_f_mem(generated_r, rho=rho),
        }
        if validation_r is not None:
            record["validation_rate"] = compute_f_mem(validation_r, rho=rho)
        records.append(record)
    return records


def compute_image_memorization_metrics(
    samples,
    reference,
    *,
    encoder=None,
    rho=None,
    query_batch_size=256,
    reference_batch_size=None,
    image_range="auto",
    device="cpu",
):
    """Compute image memorization in normalized pixels or encoder features."""
    query_batch_size = int(query_batch_size)
    if query_batch_size < 1:
        raise ValueError(f"query_batch_size must be at least 1, got {query_batch_size}")

    sample_vectors, reference_vectors, space, encoder_name = build_metric_vectors(
        samples,
        reference,
        encoder=encoder,
        batch_size=query_batch_size,
        image_range=image_range,
        device=device,
    )
    if sample_vectors.shape[1] != reference_vectors.shape[1]:
        raise ValueError(
            "Sample and reference vectors must have matching feature dimensions, "
            f"got {sample_vectors.shape[1]} and {reference_vectors.shape[1]}."
        )

    metrics = compute_memorization_metrics(
        sample_vectors,
        reference_vectors,
        rho=rho,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    metrics.update(
        {
            "space": space,
            "encoder": encoder_name,
            "image_range": str(image_range),
            "query_batch_size": query_batch_size,
            "reference_batch_size": (
                None if reference_batch_size is None else int(reference_batch_size)
            ),
            "feature_dim": int(sample_vectors.shape[1]),
        }
    )
    return metrics


__all__ = [
    "REFERENCE_QUANTILES",
    "compute_f_mem_by_rho",
    "compute_f_mem_reference_curve",
    "compute_image_memorization_metrics",
]
