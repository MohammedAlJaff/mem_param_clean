"""Synthetic-data metric entrypoints."""

from mem_param_src.metrics.synthetic.controlled import (
    collect_controlled_synthetic_metric_record,
    geometry_kind,
    intrinsic_geometry_parts,
    memorization_metrics_in_spaces,
    synthetic_distribution_metrics,
)
from mem_param_src.metrics.synthetic.gmm_subspace import (
    gmm_intrinsic_log_prob,
    gmm_intrinsic_metrics,
    gmm_subspace_parts,
    project_to_subspace,
)
from mem_param_src.metrics.synthetic.torus3d import torus3d_metrics, torus3d_parts

__all__ = [
    "collect_controlled_synthetic_metric_record",
    "geometry_kind",
    "gmm_intrinsic_log_prob",
    "gmm_intrinsic_metrics",
    "gmm_subspace_parts",
    "intrinsic_geometry_parts",
    "memorization_metrics_in_spaces",
    "project_to_subspace",
    "synthetic_distribution_metrics",
    "torus3d_metrics",
    "torus3d_parts",
]
