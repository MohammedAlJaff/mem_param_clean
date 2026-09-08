"""Fixed-noise image previews for training checkpoints."""

import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from mem_param_src.metrics.image.projection import (
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    make_random_projection_matrix,
)
from mem_param_src.runners.artifacts import save_torch


def prepare_f_mem_monitor(config, train_tensor, run_dir):
    """Cache and save the fixed projection of the actual training images."""
    settings = config.random_projected_f_mem
    if not bool(settings.enabled):
        return None
    projection = make_random_projection_matrix(
        train_tensor.shape[1] * int(settings.resize_size) ** 2,
        settings.projection_dim,
        settings.projection_seed,
    )
    reference = compute_projected_image_features(
        train_tensor,
        resize_size=settings.resize_size,
        projection_dim=settings.projection_dim,
        projection_matrix=projection,
        image_range="-1_1",
        device="cpu",
        batch_size=config.monitoring.batch_size,
    )
    monitor_dir = Path(run_dir) / "monitoring"
    save_torch(monitor_dir / "f_mem_projection.pt", projection)
    save_torch(monitor_dir / "f_mem_train_vectors.pt", reference)
    return {"projection": projection, "reference": reference}


def compute_monitor_f_mem(samples, config, state):
    """Measure r = d1 / d2 against the cached training set, then f_mem."""
    settings = config.random_projected_f_mem
    return compute_projected_image_memorization_metrics(
        samples[: int(settings.n_samples)],
        projected_reference_vectors=state["reference"],
        projection_matrix=state["projection"],
        resize_size=settings.resize_size,
        projection_dim=settings.projection_dim,
        projection_seed=settings.projection_seed,
        rho=config.evaluation.f_mem_threshold,
        query_batch_size=config.monitoring.batch_size,
        reference_batch_size=config.evaluation.chunk_size,
        image_range="-1_1",
        device="cpu",
    )


def generate_monitor_samples(model, z_init, config, device):
    """Generate one fixed-noise sample panel without changing training state."""
    count = int(config.generation_monitor.n_samples)
    if count < 1:
        raise ValueError("generation_monitor.n_samples must be at least 1.")
    if bool(config.random_projected_f_mem.enabled):
        count = max(count, int(config.random_projected_f_mem.n_samples))
    if count > z_init.shape[0]:
        raise ValueError("Monitor sample count exceeds evaluation.sample_bank_size.")
    batch_size = int(config.monitoring.batch_size)
    if batch_size < 1:
        raise ValueError("monitoring.batch_size must be at least 1.")

    # Reuse this exact z_init prefix so checkpoint differences come from training.
    fixed_noise = z_init[:count]
    was_training = model.training
    # Evaluation mode keeps stochastic training layers fixed across checkpoints.
    model.eval()
    samples = []
    try:
        # Sampling is an observation of training and must not create gradients.
        with torch.no_grad():
            for start in range(0, count, batch_size):
                z_batch = fixed_noise[start : start + batch_size].to(device)
                samples.append(
                    model.sample(
                        z_batch,
                        n_steps=int(config.sampling.n_steps),
                        method=str(config.sampling.method),
                        start_time=float(config.sampling.start_time),
                        end_time=float(config.sampling.end_time),
                        use_safe_division=bool(config.sampling.use_safe_division),
                        safe_div_denom_eps=float(config.sampling.safe_div_denom_eps),
                        output=str(config.sampling.output),
                        device=device,
                    )
                    .detach()
                    .cpu()
                )
    finally:
        # Restore the mode that controlled dropout and normalization during training.
        model.train(was_training)

    samples = torch.cat(samples, dim=0)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Generation-monitor samples became non-finite.")
    return samples


def save_fixed_range_grid(images, path):
    """Save one image grid with the study range fixed at ``[-1, 1]``."""
    if images.dim() != 4 or images.shape[0] < 1:
        raise ValueError("Image monitoring expects a non-empty [B, C, H, W] tensor.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(math.ceil(math.sqrt(images.shape[0]))))
    # A fixed range exposes unstable magnitudes instead of rescaling each grid.
    save_image(
        images,
        path,
        nrow=nrow,
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    return path
