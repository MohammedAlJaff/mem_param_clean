"""
Metrics for the synthetic 3D torus dataset.
"""

import torch

from mem_param_src.datasets.torus import (
    torus_angles_to_xyz,
    torus_periodic_features,
    torus_xyz_to_angles,
)
from mem_param_src.metrics.sliced_wasserstein import sliced_w2


def torus3d_parts(samples, geometry):
    """Map ambient samples to torus angle features."""
    major_radius = float(geometry["major_radius"])
    minor_radius = float(geometry["minor_radius"])
    basis = geometry.get("basis")
    torus_xyz = samples @ basis if basis is not None else samples
    theta, phi = torus_xyz_to_angles(torus_xyz, major_radius, minor_radius)
    intrinsic = torus_periodic_features(theta, phi)
    nearest_torus_xyz = torus_angles_to_xyz(theta, phi, major_radius, minor_radius)

    if basis is not None:
        projected = nearest_torus_xyz @ basis.T
    else:
        projected = nearest_torus_xyz

    return {
        "kind": "torus3d",
        "intrinsic": intrinsic,
        "geometry_error": torch.linalg.norm(torus_xyz - nearest_torus_xyz, dim=1),
        "projected": projected,
        "torus_xyz": torus_xyz,
        "nearest_torus_xyz": nearest_torus_xyz,
        "theta": theta,
        "phi": phi,
    }


def torus3d_metrics(samples, geometry, reference_intrinsic, directions):
    """Measure samples in torus periodic angle-feature space."""
    parts = torus3d_parts(samples, geometry)
    return {
        "intrinsic_sliced_w2": sliced_w2(
            parts["intrinsic"],
            reference_intrinsic,
            directions,
        ),
    }


__all__ = [
    "torus3d_metrics",
    "torus3d_parts",
]
