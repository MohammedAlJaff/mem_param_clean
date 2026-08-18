"""
Metrics for the synthetic GMM subspace dataset.
"""

import math

import torch

from mem_param_src.metrics.sliced_wasserstein import sliced_w2


def project_to_subspace(samples, basis):
    """Project ambient samples onto the known GMM subspace."""
    intrinsic = samples @ basis
    projected = intrinsic @ basis.T
    return {
        "intrinsic": intrinsic,
        "projected": projected,
    }


def gmm_subspace_parts(samples, geometry):
    """Return intrinsic coordinates and subspace residuals."""
    parts = project_to_subspace(samples, geometry["basis"])
    return {
        "kind": "gmm_subspace",
        "intrinsic": parts["intrinsic"],
        "geometry_error": (samples - parts["projected"]).norm(dim=1),
        "projected": parts["projected"],
    }


def gmm_intrinsic_log_prob(intrinsic, means, variance):
    """Evaluate the known equal-weight GMM density."""
    if variance <= 0.0:
        raise ValueError("Intrinsic GMM variance must be positive.")

    n_components = int(means.shape[0])
    intrinsic_dim = int(means.shape[1])
    delta = intrinsic[:, None, :] - means[None, :, :]
    squared_distance = delta.pow(2).sum(dim=2)
    log_components = -0.5 * (
        squared_distance / variance
        + intrinsic_dim * math.log(2.0 * math.pi * variance)
    ) - math.log(float(n_components))
    return torch.logsumexp(log_components, dim=1)


def gmm_intrinsic_metrics(
    samples,
    basis,
    means,
    sigma_z,
    sigma_eta,
    reference_intrinsic,
    directions,
):
    """Measure generated samples in known GMM intrinsic coordinates."""
    geometry = {"kind": "gmm_subspace", "basis": basis, "means": means}
    intrinsic = gmm_subspace_parts(samples, geometry)["intrinsic"]
    intrinsic_var = float(sigma_z) ** 2 + float(sigma_eta) ** 2
    nll = -gmm_intrinsic_log_prob(intrinsic, means, intrinsic_var)
    return {
        "intrinsic_gmm_nll_mean": float(nll.mean().item()),
        "intrinsic_sliced_w2": sliced_w2(
            intrinsic,
            reference_intrinsic,
            directions,
        ),
    }


__all__ = [
    "gmm_intrinsic_log_prob",
    "gmm_intrinsic_metrics",
    "gmm_subspace_parts",
    "project_to_subspace",
]
