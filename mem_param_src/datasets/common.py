"""
Shared data-bundle helpers for first-party studies.

This file contains the small helpers used by every dataset loader:
- deterministic subset selection from a larger dataset
- optional flattening when a study wants vectors instead of image tensors
- dataset-name dispatch into the correct loader

Canonical bundle contract:
    train:              the tensor seen by the optimizer
    reference:          the tensor used for evaluation comparisons
    train_labels:       optional labels aligned with `train`
    reference_labels:   optional labels aligned with `reference`
    train_indices:      optional source indices for deterministic replay
    reference_indices:  optional source indices for deterministic replay
    train_source_id:    stable identifier for the training source split/corpus
    reference_source_id: stable identifier for the held-out source split/corpus
    display_shape:      optional original image shape for plotting utilities
    geometry:           optional dataset-specific structure such as GMM basis

The rest of the pipeline assumes this contract. Training reads `train`;
checkpoint evaluation reads `train`, `reference`, and saved metadata; metrics
use optional labels, indices, and geometry when they are present.
"""

import torch


def seeded_indices(size, count, seed):
    """
    Draw `count` unique indices from `range(size)` using a fixed seed.
    """
    if count > size:
        raise ValueError(f"Requested {count} items from a collection of size {size}")

    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(size, generator=generator)[:count]


def maybe_flatten(tensor, config):
    """
    Flatten image-like tensors to `[B, D]` when the study config asks for it.
    """
    if getattr(config.data, "flatten", False):
        return tensor.flatten(1)
    return tensor


def _indices_to_set(indices):
    """
    Convert a tensor/list of source indices into a Python set of integers.
    """
    if indices is None:
        return None
    if torch.is_tensor(indices):
        return {int(item) for item in indices.detach().cpu().flatten().tolist()}
    return {int(item) for item in indices}


def _validate_no_identical_full_tensors(bundle):
    """
    Fail loudly if train and reference tensors are exactly the same tensor batch.
    """
    train = bundle["train"]
    reference = bundle["reference"]
    if train.shape == reference.shape and torch.equal(train, reference):
        raise ValueError(
            "Data leakage guard failed: train and reference tensors are exactly equal."
        )


def validate_data_bundle(bundle, config):
    """
    Validate that train/reference roles and source identities cannot leak.

    The strict default is intentional: a held-out reference set should come from
    a different source split/corpus unless a study explicitly opts into using
    disjoint indices from the same source via
    `config.data.allow_same_source_reference = True`.

    This guard protects memorization metrics. If train and reference overlap,
    a generated sample can look non-memorized only because the baseline is
    contaminated by training examples.
    """
    for key in ("train", "reference", "train_source_id", "reference_source_id"):
        if key not in bundle:
            raise ValueError(f"Data bundle is missing required leakage-guard key: {key}")

    _validate_no_identical_full_tensors(bundle)

    train_source_id = str(bundle["train_source_id"])
    reference_source_id = str(bundle["reference_source_id"])
    same_source = train_source_id == reference_source_id
    overlap_count = None
    allow_same_source = bool(getattr(config.data, "allow_same_source_reference", False))

    if same_source:
        train_indices = _indices_to_set(bundle.get("train_indices"))
        reference_indices = _indices_to_set(bundle.get("reference_indices"))
        if train_indices is None or reference_indices is None:
            raise ValueError(
                "Data leakage guard failed: train/reference share a source but "
                "source indices are missing, so disjointness cannot be verified."
            )
        overlap = train_indices.intersection(reference_indices)
        overlap_count = len(overlap)
        if overlap_count:
            examples = sorted(overlap)[:10]
            raise ValueError(
                "Data leakage guard failed: train/reference source indices overlap "
                f"for source {train_source_id!r}. overlap_count={overlap_count}, "
                f"examples={examples}"
            )
        if not allow_same_source:
            raise ValueError(
                "Data leakage guard failed: train and reference use the same source "
                f"{train_source_id!r}. Use a true held-out split, or set "
                "config.data.allow_same_source_reference=True only for an explicitly "
                "disjoint same-corpus holdout."
            )

    validation = {
        "train_source_id": train_source_id,
        "reference_source_id": reference_source_id,
        "train_role": str(bundle.get("train_role", "train")),
        "reference_role": str(bundle.get("reference_role", "heldout_reference")),
        "same_source": same_source,
        "same_source_allowed": allow_same_source,
        "overlap_checked": same_source,
        "overlap_count": overlap_count,
    }
    bundle["data_validation"] = validation
    return bundle


def build_data_bundle(config):
    """
    Dispatch from `config.data.dataset` to the corresponding canonical loader.

    Every loader returns the same high-level bundle contract documented at the
    top of this module.
    """
    dataset_name = config.data.dataset

    if dataset_name == "mnist":
        from mem_param_src.datasets.mnist import load_mnist_bundle

        bundle = load_mnist_bundle(config)
    elif dataset_name == "gmm_subspace":
        from mem_param_src.datasets.gmm import load_gmm_subspace_bundle

        bundle = load_gmm_subspace_bundle(config)
    elif dataset_name == "torus3d":
        from mem_param_src.datasets.torus import load_torus3d_bundle

        bundle = load_torus3d_bundle(config)
    elif dataset_name == "celeba32_grey":
        from mem_param_src.datasets.celeba32 import load_celeba32_bundle

        bundle = load_celeba32_bundle(config)
    elif dataset_name == "flowers102":
        from mem_param_src.datasets.image_collections import load_flowers102_bundle

        bundle = load_flowers102_bundle(config)
    elif dataset_name == "afhq_v2":
        from mem_param_src.datasets.image_collections import load_afhq_v2_bundle

        bundle = load_afhq_v2_bundle(config)
    elif dataset_name == "ffhq128":
        from mem_param_src.datasets.image_collections import load_ffhq128_bundle

        bundle = load_ffhq128_bundle(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return validate_data_bundle(bundle, config)


if __name__ == "__main__":
    from mem_param_src.config import get_base_config

    torch.manual_seed(0)

    print("mem_param_src.datasets.common example")

    # Show the two small helper contracts first: seeded subset selection and
    # optional flattening of image-shaped tensors.
    indices = seeded_indices(size=10, count=4, seed=3)
    image_batch = torch.randn(2, 1, 4, 4)

    # Build a tiny synthetic-data config so `build_data_bundle(...)` is runnable.
    config = get_base_config()
    config.data.dataset = "gmm_subspace"
    config.data.n_train = 8
    config.data.n_test = 6
    config.data.data_dim = 8
    config.data.intrinsic_dim = 2
    config.data.n_clusters = 3
    config.data.flatten = False

    # Materialize the helper outputs and then the full bundle contract.
    flattened = maybe_flatten(image_batch, config)
    bundle = build_data_bundle(config)

    # The key check here is that every loader returns the same bundle keys.
    print(f"seeded_indices        : {indices.tolist()}")
    print(f"maybe_flatten shape   : {tuple(flattened.shape)}")
    print(f"bundle keys           : {sorted(bundle.keys())}")
    print(f"train shape           : {tuple(bundle['train'].shape)}")
    print(f"reference shape       : {tuple(bundle['reference'].shape)}")
