"""Checkpoint-level image memorization records."""

from mem_param_src.metrics.nearest_neighbors import (
    compute_f_mem,
    compute_nn_stats,
    resolve_rho,
    summarize_nn_stats,
)

RHO_SWEEP = (0.3, 0.5, 0.7)


def collect_image32_metric_record(
    step,
    samples,
    baseline_samples,
    train_tensor,
    *,
    chunk_size,
    config,
):
    """Compare generated and held-out images with the saved training tensor."""
    rho = resolve_rho(config=config)
    baseline_samples = baseline_samples[: samples.shape[0]]
    generated_stats = compute_nn_stats(
        samples,
        train_tensor,
        chunk_size=int(chunk_size),
    )
    baseline_stats = compute_nn_stats(
        baseline_samples,
        train_tensor,
        chunk_size=int(chunk_size),
    )
    generated = summarize_nn_stats(
        generated_stats,
        rho=rho,
        n_samples=samples.shape[0],
        n_reference=train_tensor.shape[0],
    )
    baseline = summarize_nn_stats(
        baseline_stats,
        rho=rho,
        n_samples=baseline_samples.shape[0],
        n_reference=train_tensor.shape[0],
    )

    record = {
        "step": int(step),
        "rho": generated["rho"],
        "f_mem": generated["f_mem"],
        "baseline_f_mem": baseline["f_mem"],
        "delta_f_mem": generated["f_mem"] - baseline["f_mem"],
        "mean_r": generated["mean_r"],
        "median_r": generated["median_r"],
        "baseline_mean_r": baseline["mean_r"],
        "baseline_median_r": baseline["median_r"],
        "mean_d1": generated["mean_d1"],
        "mean_d2": generated["mean_d2"],
        "baseline_mean_d1": baseline["mean_d1"],
        "baseline_mean_d2": baseline["mean_d2"],
    }
    for rho_value in RHO_SWEEP:
        generated_f_mem = compute_f_mem(generated_stats["r"], rho=rho_value)
        baseline_f_mem = compute_f_mem(baseline_stats["r"], rho=rho_value)
        record[f"f_mem_rho{rho_value}"] = generated_f_mem
        record[f"baseline_f_mem_rho{rho_value}"] = baseline_f_mem
        record[f"delta_f_mem_rho{rho_value}"] = generated_f_mem - baseline_f_mem
    return record


__all__ = ["RHO_SWEEP", "collect_image32_metric_record"]
