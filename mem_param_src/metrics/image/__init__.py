"""Image-specific metric adapters and record builders."""

from mem_param_src.metrics.image.features import (
    DINOv2FeatureEncoder,
    LPIPSFeatureEncoder,
    SSCDFeatureEncoder,
    TorchFidelityFeatureEncoder,
    encode_image_features,
)
from mem_param_src.metrics.image.fid import (
    DEFAULT_FID_FEATURE_EXTRACTOR,
    compute_image_fid,
)
from mem_param_src.metrics.image.memorization import (
    REFERENCE_QUANTILES,
    compute_f_mem_by_rho,
    compute_f_mem_reference_curve,
    compute_image_memorization_metrics,
)
from mem_param_src.metrics.image.mnist_jointae import (
    MNIST_F_MEM_ALPHAS,
    MNIST_JOINTAE_ARCHITECTURE_VERSION,
    MNIST_JOINTAE_CLASS_FD_NAME,
    MNIST_JOINTAE_FD_NAME,
    MNIST_JOINTAE_F_MEM_NAME,
    MNISTJointAE32Metric,
    canonicalize_mnist32,
    compute_feature_statistics,
    compute_frechet_distance,
    compute_mnist_feature_memorization,
    compute_mnist_jointae32_class_fd,
    compute_mnist_jointae32_fd,
    extract_mnist_jointae32_features,
    load_mnist_jointae32,
    load_mnist_jointae32_manifest,
    load_mnist_jointae32_reference_stats,
    normalize_mnist_memorization_features,
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
    "DINOv2FeatureEncoder",
    "LPIPSFeatureEncoder",
    "MNIST_F_MEM_ALPHAS",
    "MNIST_JOINTAE_ARCHITECTURE_VERSION",
    "MNIST_JOINTAE_CLASS_FD_NAME",
    "MNIST_JOINTAE_FD_NAME",
    "MNIST_JOINTAE_F_MEM_NAME",
    "MNISTJointAE32Metric",
    "REFERENCE_QUANTILES",
    "RHO_SWEEP",
    "SSCDFeatureEncoder",
    "TorchFidelityFeatureEncoder",
    "as_image_batch",
    "canonicalize_mnist32",
    "collect_image32_metric_record",
    "compute_f_mem",
    "compute_f_mem_by_rho",
    "compute_f_mem_reference_curve",
    "compute_feature_statistics",
    "compute_frechet_distance",
    "compute_image_fid",
    "compute_image_memorization_metrics",
    "compute_mnist_feature_memorization",
    "compute_mnist_jointae32_class_fd",
    "compute_mnist_jointae32_fd",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "compute_projected_image_features",
    "compute_projected_image_memorization_metrics",
    "encode_image_features",
    "extract_mnist_jointae32_features",
    "load_mnist_jointae32",
    "load_mnist_jointae32_manifest",
    "load_mnist_jointae32_reference_stats",
    "make_random_projection_matrix",
    "normalize_images",
    "normalize_mnist_memorization_features",
    "to_uint8_rgb",
]
