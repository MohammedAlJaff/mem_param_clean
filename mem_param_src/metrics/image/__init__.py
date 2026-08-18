"""Image-specific metric adapters and record builders."""

from mem_param_src.metrics.image.features import TorchFidelityFeatureEncoder
from mem_param_src.metrics.image.fid import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    compute_image_fid,
)
from mem_param_src.metrics.image.memorization import (
    compute_f_mem_by_rho,
    compute_image_memorization_metrics,
)
from mem_param_src.metrics.image.preprocessing import (
    as_image_batch,
    normalize_images,
    to_uint8_rgb,
)
from mem_param_src.metrics.image.projection import (
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    make_random_projection_matrix,
)
from mem_param_src.metrics.image.records import RHO_SWEEP, collect_image32_metric_record
from mem_param_src.metrics.nearest_neighbors import (
    compute_f_mem,
    compute_memorization_metrics,
    compute_nn_stats,
)

__all__ = [
    "DEFAULT_FID_FEATURE_EXTRACTOR",
    "RHO_SWEEP",
    "TorchFidelityFeatureEncoder",
    "as_image_batch",
    "collect_image32_metric_record",
    "compute_f_mem",
    "compute_f_mem_by_rho",
    "compute_image_fid",
    "compute_image_memorization_metrics",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "compute_projected_image_features",
    "compute_projected_image_memorization_metrics",
    "make_random_projection_matrix",
    "normalize_images",
    "to_uint8_rgb",
]
