"""
Shared synthetic-data metric entrypoints.
"""

from mem_param_src.metrics.nearest_neighbors import compute_nn_stats
from mem_param_src.metrics.synthetic.gmm_subspace import (
    gmm_intrinsic_metrics,
    gmm_subspace_parts,
)
from mem_param_src.metrics.synthetic.torus3d import torus3d_metrics, torus3d_parts


def geometry_kind(geometry, config=None):
    """Return the controlled synthetic geometry name."""
    if geometry is None:
        if config is None:
            return ""
        return str(getattr(config.data, "dataset", ""))
    kind = geometry.get("kind") if isinstance(geometry, dict) else None
    if kind is not None:
        return str(kind)
    if "basis" in geometry:
        return "gmm_subspace"
    if config is not None:
        return str(getattr(config.data, "dataset", ""))
    return ""


def intrinsic_geometry_parts(samples, geometry, config=None):
    """Return intrinsic coordinates for a controlled synthetic dataset."""
    kind = geometry_kind(geometry, config)
    if kind == "gmm_subspace":
        return gmm_subspace_parts(samples, geometry)
    if kind == "torus3d":
        return torus3d_parts(samples, geometry)
    raise ValueError(f"Unsupported controlled synthetic geometry: {kind!r}")


def memorization_metrics_in_spaces(
    samples,
    train_x,
    geometry,
    train_intrinsic,
    *,
    chunk_size,
    config,
):
    """Measure nearest-neighbor ratios in ambient and intrinsic spaces."""
    ambient = compute_nn_stats(
        samples,
        train_x,
        chunk_size=chunk_size,
    )
    intrinsic = intrinsic_geometry_parts(samples, geometry, config=config)["intrinsic"]
    intrinsic_stats = compute_nn_stats(
        intrinsic,
        train_intrinsic,
        chunk_size=chunk_size,
    )
    return {
        "ambient_nearest_neighbor_ratio_d1_over_d2": float(ambient["r"].mean().item()),
        "intrinsic_nearest_neighbor_ratio_d1_over_d2": float(
            intrinsic_stats["r"].mean().item()
        ),
    }


def synthetic_distribution_metrics(
    samples,
    geometry,
    reference_intrinsic,
    directions,
    config,
):
    """Dispatch dataset-specific synthetic distribution metrics."""
    kind = geometry_kind(geometry, config)
    if kind == "gmm_subspace":
        return gmm_intrinsic_metrics(
            samples,
            geometry["basis"],
            geometry["means"],
            float(config.data.sigma_z),
            float(config.data.sigma_eta),
            reference_intrinsic,
            directions,
        )
    if kind == "torus3d":
        return torus3d_metrics(samples, geometry, reference_intrinsic, directions)
    raise ValueError(f"Unsupported controlled synthetic geometry: {kind!r}")


def collect_controlled_synthetic_metric_record(
    *,
    step,
    samples,
    baseline_samples,
    train_tensor,
    train_intrinsic,
    geometry,
    reference_intrinsic,
    sliced_directions,
    chunk_size,
    config,
):
    """Collect generated and held-out synthetic metrics in one record."""
    generated_mem = memorization_metrics_in_spaces(
        samples,
        train_tensor,
        geometry,
        train_intrinsic,
        chunk_size=chunk_size,
        config=config,
    )
    generated_distribution = synthetic_distribution_metrics(
        samples,
        geometry,
        reference_intrinsic,
        sliced_directions,
        config,
    )
    holdout_mem = memorization_metrics_in_spaces(
        baseline_samples,
        train_tensor,
        geometry,
        train_intrinsic,
        chunk_size=chunk_size,
        config=config,
    )
    holdout_intrinsic = intrinsic_geometry_parts(
        baseline_samples,
        geometry,
        config=config,
    )["intrinsic"]
    holdout_distribution = synthetic_distribution_metrics(
        baseline_samples,
        geometry,
        holdout_intrinsic,
        sliced_directions,
        config,
    )

    record = {"step": int(step)}
    record.update(generated_mem)
    record.update(generated_distribution)
    record.update({f"holdout_{key}": value for key, value in holdout_mem.items()})
    record.update(
        {f"holdout_{key}": value for key, value in holdout_distribution.items()}
    )
    return record


__all__ = [
    "collect_controlled_synthetic_metric_record",
    "geometry_kind",
    "intrinsic_geometry_parts",
    "memorization_metrics_in_spaces",
    "synthetic_distribution_metrics",
]
