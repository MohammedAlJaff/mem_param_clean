"""
Nearest-neighbor memorization metrics for vector data.

Inputs can be vectors or tensors with a batch dimension. Non-vector inputs are
flattened to `[B, D]` before distances are computed.
"""

import torch


def _flatten_batch(tensor):
    """Convert a batch tensor to `[B, D]`."""
    if tensor.dim() == 2:
        return tensor
    return tensor.flatten(1)


def resolve_rho(rho=None, config=None):
    """Return the canonical threshold for the nearest-neighbor ratio ``r``."""
    if rho is not None:
        return float(rho)
    if config is not None and hasattr(config, "evaluation"):
        evaluation = config.evaluation
        if hasattr(evaluation, "f_mem_threshold"):
            return float(evaluation.f_mem_threshold)
        raise ValueError("evaluation.f_mem_threshold is required.")
    return 0.3


def _resolve_batch_size(value, name):
    """Return a positive batch size or `None`."""
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _top2_distances(query, reference, reference_batch_size=None):
    """Return both nearest distances and their reference indices."""
    n_reference = int(reference.shape[0])
    reference_batch_size = _resolve_batch_size(
        reference_batch_size,
        "reference_batch_size",
    )
    if reference_batch_size is None or reference_batch_size >= n_reference:
        distances = torch.cdist(query, reference)
        topk = torch.topk(distances, k=2, largest=False, dim=1)
        return (
            topk.values[:, 0],
            topk.values[:, 1],
            topk.indices[:, 0],
            topk.indices[:, 1],
        )

    best_values = torch.full(
        (query.shape[0], 2),
        float("inf"),
        dtype=query.dtype,
        device=query.device,
    )
    best_indices = torch.full(
        (query.shape[0], 2),
        -1,
        dtype=torch.long,
        device=query.device,
    )

    for ref_start in range(0, n_reference, reference_batch_size):
        ref_end = min(ref_start + reference_batch_size, n_reference)
        ref_chunk = reference[ref_start:ref_end]
        distances = torch.cdist(query, ref_chunk)
        local_k = min(2, int(ref_chunk.shape[0]))
        local_topk = torch.topk(distances, k=local_k, largest=False, dim=1)
        local_indices = local_topk.indices + ref_start

        candidate_values = torch.cat([best_values, local_topk.values], dim=1)
        candidate_indices = torch.cat([best_indices, local_indices], dim=1)
        merged = torch.topk(candidate_values, k=2, largest=False, dim=1)
        best_values = merged.values
        best_indices = torch.gather(candidate_indices, 1, merged.indices)

    return (
        best_values[:, 0],
        best_values[:, 1],
        best_indices[:, 0],
        best_indices[:, 1],
    )


def compute_nn_stats(
    samples,
    reference,
    chunk_size=256,
    query_batch_size=None,
    reference_batch_size=None,
):
    """Compute `d1`, `d2`, `r = d1 / d2`, and both neighbor indices."""
    query_batch_size = _resolve_batch_size(
        chunk_size if query_batch_size is None else query_batch_size,
        "query_batch_size",
    )

    samples = _flatten_batch(samples).float()
    reference = _flatten_batch(reference).float().to(samples.device)
    if reference.shape[0] < 2:
        raise ValueError(
            "Nearest-neighbor metrics require at least two reference samples."
        )

    d1_list = []
    d2_list = []
    nn_idx_list = []
    second_nn_idx_list = []
    for start in range(0, samples.shape[0], query_batch_size):
        chunk = samples[start : start + query_batch_size]
        d1, d2, nn_idx, second_nn_idx = _top2_distances(
            chunk,
            reference,
            reference_batch_size=reference_batch_size,
        )
        d1_list.append(d1)
        d2_list.append(d2)
        nn_idx_list.append(nn_idx)
        second_nn_idx_list.append(second_nn_idx)
    d1 = torch.cat(d1_list)
    d2 = torch.cat(d2_list)
    r = d1 / d2.clamp(min=1.0e-12)
    return {
        "d1": d1,
        "d2": d2,
        "r": r,
        "nn_idx": torch.cat(nn_idx_list),
        "second_nn_idx": torch.cat(second_nn_idx_list),
    }


def compute_f_mem(r, rho=None, config=None):
    """Compute `f_mem = mean(r < rho)`."""
    rho = resolve_rho(rho, config=config)
    r = torch.as_tensor(r, dtype=torch.float32)
    return float((r < rho).float().mean().item())


def summarize_nn_stats(stats, *, rho, n_samples, n_reference):
    """Return scalar summaries from nearest-neighbor tensors."""
    r = stats["r"]
    return {
        "rho": float(rho),
        "n_samples": int(n_samples),
        "N": int(n_reference),
        "f_mem": compute_f_mem(r, rho=rho),
        "mean_r": float(r.mean().item()),
        "std_r": float(r.std(unbiased=False).item()),
        "median_r": float(r.median().item()),
        "mean_d1": float(stats["d1"].mean().item()),
        "mean_d2": float(stats["d2"].mean().item()),
        "median_d1": float(stats["d1"].median().item()),
        "median_d2": float(stats["d2"].median().item()),
    }


def compute_memorization_metrics(
    samples,
    reference,
    rho=None,
    chunk_size=256,
    query_batch_size=None,
    reference_batch_size=None,
    config=None,
):
    """Summarize nearest-neighbor memorization against a reference set."""
    rho = resolve_rho(rho, config=config)
    stats = compute_nn_stats(
        samples,
        reference,
        chunk_size=chunk_size,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    return summarize_nn_stats(
        stats,
        rho=rho,
        n_samples=samples.shape[0],
        n_reference=reference.shape[0],
    )


__all__ = [
    "compute_f_mem",
    "compute_memorization_metrics",
    "compute_nn_stats",
    "resolve_rho",
    "summarize_nn_stats",
]
