"""Fixed-noise image previews for training checkpoints."""

import math
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torchvision.utils import save_image

from mem_param_src.metrics.image.projection import (
    compute_projected_image_features,
    compute_projected_image_memorization_metrics,
    make_random_projection_matrix,
)
from mem_param_src.runners.artifacts import save_torch


_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def resolve_monitor_sampling_profiles(config):
    """Validate named sampling overrides used only by checkpoint monitoring."""
    configured = config.generation_monitor.get("sampling_profiles", ())
    if not configured:
        return ()

    profiles = []
    names = set()
    required_fields = {
        "name",
        "eps_to_v",
        "use_safe_div",
        "safe_div_denominator_floor",
    }
    for index, configured_profile in enumerate(configured):
        if not isinstance(configured_profile, Mapping):
            raise ValueError(
                "generation_monitor.sampling_profiles entries must be mappings."
            )
        missing = sorted(required_fields - set(configured_profile))
        unexpected = sorted(set(configured_profile) - required_fields)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            raise ValueError(
                "Invalid generation-monitor sampling profile: " + "; ".join(details)
            )

        name = str(configured_profile["name"])
        if not _PROFILE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Sampling profile names must contain only letters, numbers, "
                "underscores, and hyphens."
            )
        if name in names:
            raise ValueError(f"Duplicate generation-monitor sampling profile: {name}")
        names.add(name)

        eps_to_v = str(configured_profile["eps_to_v"])
        if eps_to_v not in ("direct", "via_x"):
            raise ValueError("Sampling profile eps_to_v must be 'direct' or 'via_x'.")
        use_safe_div = configured_profile["use_safe_div"]
        if not isinstance(use_safe_div, bool):
            raise ValueError("Sampling profile use_safe_div must be true or false.")
        floor = float(configured_profile["safe_div_denominator_floor"])
        if not math.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError(
                "Sampling profile safe_div_denominator_floor must be between 0 and 1."
            )
        profiles.append(
            {
                "name": name,
                "eps_to_v": eps_to_v,
                "use_safe_div": use_safe_div,
                "safe_div_denominator_floor": floor,
                "required": index == 0,
            }
        )

    primary = profiles[0]
    if primary["eps_to_v"] != "direct" or not primary["use_safe_div"]:
        raise ValueError(
            "The first generation-monitor sampling profile must use direct "
            "epsilon conversion with safe division."
        )
    return tuple(profiles)


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


def generate_monitor_samples(
    model,
    z_init,
    config,
    device,
    *,
    return_raw=False,
    sampling_profile=None,
    allow_nonfinite=False,
):
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
    profile = sampling_profile or {}
    eps_to_v = profile.get("eps_to_v", config.sampling.get("eps_to_v", "via_x"))
    use_safe_div = profile.get("use_safe_div", bool(config.sampling.use_safe_div))
    denominator_floor = profile.get(
        "safe_div_denominator_floor",
        float(config.sampling.safe_div_denominator_floor),
    )
    was_training = model.training
    # Evaluation mode keeps stochastic training layers fixed across checkpoints.
    model.eval()
    final_pixel_clamp = bool(config.sampling.get("final_pixel_clamp", False))
    samples = []
    raw_samples = []
    try:
        # Sampling is an observation of training and must not create gradients.
        with torch.no_grad():
            for start in range(0, count, batch_size):
                z_batch = fixed_noise[start : start + batch_size].to(device)
                generated = model.sample(
                    z_batch,
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    start_time=float(config.sampling.start_time),
                    end_time=float(config.sampling.end_time),
                    use_safe_div=bool(use_safe_div),
                    safe_div_denominator_floor=float(denominator_floor),
                    output=str(config.sampling.output),
                    eps_to_v=str(eps_to_v),
                    heun_final_euler=config.sampling.get("heun_final_euler", False),
                    final_pixel_clamp=final_pixel_clamp,
                    return_raw=return_raw,
                    device=device,
                )
                if return_raw:
                    sample_batch, raw_batch = generated
                    raw_samples.append(raw_batch.detach().cpu())
                else:
                    sample_batch = generated
                samples.append(sample_batch.detach().cpu())
    finally:
        # Restore the mode that controlled dropout and normalization during training.
        model.train(was_training)

    samples = torch.cat(samples, dim=0)
    if not allow_nonfinite and not torch.isfinite(samples).all():
        raise FloatingPointError("Generation-monitor samples became non-finite.")
    if return_raw:
        raw_samples = torch.cat(raw_samples, dim=0)
        if not allow_nonfinite and not torch.isfinite(raw_samples).all():
            raise FloatingPointError("Raw generation-monitor samples became non-finite.")
        return samples, raw_samples
    return samples


def compute_raw_sample_diagnostics(raw_samples, *, step, profile):
    """Summarize raw generated values without writing nonfinite JSON numbers."""
    values = raw_samples.detach().cpu()
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    finite_count = int(finite_values.numel())
    total_values = int(values.numel())
    if finite_count:
        raw_min = float(finite_values.min().item())
        raw_max = float(finite_values.max().item())
        raw_std = float(finite_values.float().std(unbiased=False).item())
        fraction_abs_gt_1 = float((finite_values.abs() > 1.0).float().mean().item())
    else:
        raw_min = None
        raw_max = None
        raw_std = None
        fraction_abs_gt_1 = None
    nonfinite_count = total_values - finite_count
    return {
        "step": int(step),
        "profile": str(profile["name"]),
        "status": "nonfinite" if nonfinite_count else "ok",
        "required": bool(profile["required"]),
        "eps_to_v": str(profile["eps_to_v"]),
        "use_safe_div": bool(profile["use_safe_div"]),
        "safe_div_denominator_floor": float(
            profile["safe_div_denominator_floor"]
        ),
        "n_samples": int(values.shape[0]),
        "total_values": total_values,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "nonfinite_fraction": (
            float(nonfinite_count) / float(total_values) if total_values else 0.0
        ),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_std": raw_std,
        "fraction_abs_gt_1": fraction_abs_gt_1,
    }



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
