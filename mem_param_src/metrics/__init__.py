"""Metric functions for the memorization-parameterization study."""

from mem_param_src.metrics.image import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    DINOv2FeatureEncoder,
    LPIPSFeatureEncoder,
    REFERENCE_QUANTILES,
    RHO_SWEEP,
    SSCDFeatureEncoder,
    TorchFidelityFeatureEncoder,
    collect_image32_metric_record,
    compute_f_mem_by_rho,
    compute_f_mem_reference_curve,
    compute_image_fid,
    compute_image_memorization_metrics,
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    encode_image_features,
    make_random_projection_matrix,
)
from mem_param_src.metrics.nearest_neighbors import (
    compute_f_mem,
    compute_memorization_metrics,
    compute_nn_stats,
    resolve_rho,
    summarize_nn_stats,
)
from mem_param_src.metrics.sliced_wasserstein import normalize_directions, sliced_w2

__all__ = [
    "DEFAULT_FID_FEATURE_EXTRACTOR",
    "DINOv2FeatureEncoder",
    "LPIPSFeatureEncoder",
    "REFERENCE_QUANTILES",
    "RHO_SWEEP",
    "SSCDFeatureEncoder",
    "TorchFidelityFeatureEncoder",
    "collect_image32_metric_record",
    "compute_f_mem",
    "compute_f_mem_by_rho",
    "compute_f_mem_reference_curve",
    "compute_image_fid",
    "compute_image_memorization_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "compute_projected_image_features",
    "compute_projected_image_memorization_metrics",
    "encode_image_features",
    "make_random_projection_matrix",
    "normalize_directions",
    "resolve_rho",
    "sliced_w2",
    "summarize_nn_stats",
]
