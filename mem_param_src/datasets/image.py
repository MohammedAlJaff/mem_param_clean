"""Shared deterministic pipeline for finite image datasets."""

import torch

from mem_param_src.datasets.common import (
    maybe_flatten,
    seeded_indices,
    source_catalog_sha256,
)


def _class_count_summary(labels, num_classes, include_empty_classes):
    """Return JSON-compatible counts for every configured class."""
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes)
    return {
        str(class_id): int(counts[class_id].item())
        for class_id in range(num_classes)
        if include_empty_classes or int(counts[class_id].item()) > 0
    }


def _balanced_seeded_indices(targets, count, seed, num_classes, dataset_name, split):
    """Select the same number of seeded examples from every class."""
    if count % num_classes != 0:
        raise ValueError(
            f"Class-balanced {dataset_name} {split} selection requires the sample "
            f"count to be divisible by {num_classes}, got {count}."
        )

    targets = torch.as_tensor(targets, dtype=torch.long).flatten()
    per_class = count // num_classes
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for class_id in range(num_classes):
        class_indices = torch.nonzero(targets == class_id, as_tuple=False).flatten()
        if class_indices.numel() < per_class:
            raise ValueError(
                f"Class-balanced {dataset_name} {split} selection requested "
                f"{per_class} examples for class {class_id}, but only "
                f"{class_indices.numel()} are available."
            )
        order = torch.randperm(class_indices.numel(), generator=generator)[:per_class]
        selected.append(class_indices[order])

    indices = torch.cat(selected)
    order = torch.randperm(indices.numel(), generator=generator)
    return indices[order]


def _select_indices(
    source,
    count,
    seed,
    balanced,
    num_classes,
    dataset_name,
    split,
):
    """Select one fixed subset, with optional exact class balance."""
    if not balanced:
        return seeded_indices(len(source["dataset"]), count, seed)
    return _balanced_seeded_indices(
        source["targets"],
        count,
        seed,
        num_classes,
        dataset_name,
        split,
    )


def materialize_dataset_subset(dataset, targets, indices, config):
    """Stack transformed samples from a standard Torch dataset."""
    del targets, config
    images = []
    labels = []
    for index in indices.tolist():
        image, label = dataset[index]
        images.append(image)
        labels.append(int(torch.as_tensor(label).item()))
    return torch.stack(images), torch.as_tensor(labels, dtype=torch.long)


def _catalog(source):
    """Return the ordered source catalog used for provenance hashes."""
    if "catalog" in source:
        return source["catalog"]
    return range(len(source["dataset"]))


def _selected_catalog_ids(catalog, indices):
    """Select stable source identifiers in exact tensor order."""
    return [catalog[int(index)] for index in indices.tolist()]


def _partition_bundle(source, tensor, labels, indices, role, split, config):
    """Assemble one image partition with its exact source identity."""
    catalog = _catalog(source)
    source_id = str(source["source_id"])
    return {
        role: maybe_flatten(tensor, config).contiguous(),
        f"{role}_labels": labels,
        f"{role}_indices": indices,
        f"{role}_sample_ids": _selected_catalog_ids(catalog, indices),
        f"{role}_sample_id_kind": str(source.get("sample_id_kind", "index")),
        f"{role}_source_size": len(source["dataset"]),
        f"{role}_source_catalog_sha256": source_catalog_sha256(source_id, catalog),
        f"{role}_source_id": source_id,
        f"{role}_split": str(split),
    }


def build_image_bundle(config, source):
    """Build the canonical bundle from one small dataset-source description."""
    dataset_name = str(config.data.dataset)
    balance_supported = bool(source.get("balance_supported", False))
    balance_train = bool(getattr(config.data, "balance_train_classes", False))
    balance_reference = bool(getattr(config.data, "balance_reference_classes", False))
    if (balance_train or balance_reference) and not balance_supported:
        raise ValueError(
            "Class-balanced subsets are supported only for MNIST and CIFAR-10."
        )

    train_source = source["train"]
    reference_source = source["reference"]
    num_classes = int(source.get("num_classes", 0))
    train_count = int(config.data.n_train)
    reference_count = int(config.data.n_test)
    seed = int(config.training.seed)
    if bool(source.get("shared_partition", False)):
        combined = seeded_indices(
            len(train_source["dataset"]),
            train_count + reference_count,
            seed,
        )
        train_indices = combined[:train_count]
        reference_indices = combined[train_count:]
    else:
        train_indices = _select_indices(
            train_source,
            train_count,
            seed,
            balance_train,
            num_classes,
            dataset_name,
            "training",
        )
        reference_indices = _select_indices(
            reference_source,
            reference_count,
            seed + 1,
            balance_reference,
            num_classes,
            dataset_name,
            "reference",
        )

    materialize = source.get("materialize_subset", materialize_dataset_subset)
    train_tensor, train_labels = materialize(
        train_source["dataset"],
        train_source.get("targets"),
        train_indices,
        config,
    )
    reference_tensor, reference_labels = materialize(
        reference_source["dataset"],
        reference_source.get("targets"),
        reference_indices,
        config,
    )

    bundle = _partition_bundle(
        train_source,
        train_tensor,
        train_labels,
        train_indices,
        "train",
        source["train_split"],
        config,
    )
    bundle.update(
        _partition_bundle(
            reference_source,
            reference_tensor,
            reference_labels,
            reference_indices,
            "reference",
            source["reference_split"],
            config,
        )
    )
    bundle["train_role"] = "train"
    bundle["reference_role"] = str(source.get("reference_role", "heldout_reference"))
    bundle["display_shape"] = (
        int(config.data.image_channels),
        int(config.data.image_size),
        int(config.data.image_size),
    )

    if bool(source.get("include_class_counts", True)):
        num_classes = int(source["num_classes"])
        include_empty_classes = bool(source.get("include_empty_classes", True))
        bundle["train_class_counts"] = _class_count_summary(
            train_labels,
            num_classes,
            include_empty_classes,
        )
        bundle["reference_class_counts"] = _class_count_summary(
            reference_labels,
            num_classes,
            include_empty_classes,
        )
    if "class_to_idx" in source:
        bundle["class_to_idx"] = dict(source["class_to_idx"])
    if balance_supported:
        bundle.update(
            {
                "balanced_train_classes": balance_train,
                "balanced_reference_classes": balance_reference,
            }
        )
    return bundle
