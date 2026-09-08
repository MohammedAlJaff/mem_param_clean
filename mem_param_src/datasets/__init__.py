"""
Reusable dataset-entry helpers for first-party studies.

This package owns the canonical data-bundle loaders used by the active study
surface. Its public entrypoint is `build_data_bundle`, which dispatches to the
appropriate dataset-specific loader and returns tensors plus the metadata
needed for later replay and evaluation.
"""

from mem_param_src.datasets.common import (
    validate_data_bundle,
    validate_finite_data_identity,
)


def build_data_bundle(config):
    """Load one configured dataset and validate the canonical bundle."""
    dataset_name = str(config.data.dataset)
    if dataset_name == "mnist":
        from mem_param_src.datasets.mnist import load_mnist_bundle

        bundle = load_mnist_bundle(config)
    elif dataset_name == "cifar10":
        from mem_param_src.datasets.cifar10 import load_cifar10_bundle

        bundle = load_cifar10_bundle(config)
    elif dataset_name == "celeba32_grey":
        from mem_param_src.datasets.celeba32 import load_celeba32_bundle

        bundle = load_celeba32_bundle(config)
    elif dataset_name in ("flowers102", "afhq_v2", "ffhq128"):
        from mem_param_src.datasets.image_collections import (
            load_afhq_v2_bundle,
            load_ffhq128_bundle,
            load_flowers102_bundle,
        )

        loaders = {
            "flowers102": load_flowers102_bundle,
            "afhq_v2": load_afhq_v2_bundle,
            "ffhq128": load_ffhq128_bundle,
        }
        bundle = loaders[dataset_name](config)
    elif dataset_name == "gmm_subspace":
        from mem_param_src.datasets.gmm import load_gmm_subspace_bundle

        bundle = load_gmm_subspace_bundle(config)
    elif dataset_name == "torus3d":
        from mem_param_src.datasets.torus import load_torus3d_bundle

        bundle = load_torus3d_bundle(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return validate_data_bundle(bundle, config)


__all__ = [
    "build_data_bundle",
    "validate_data_bundle",
    "validate_finite_data_identity",
]
