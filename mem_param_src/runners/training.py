"""Shared optimizer loop for image and synthetic runs."""

import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mem_param_src.config import config_to_dict
from mem_param_src.datasets import build_data_bundle, validate_finite_data_identity
from mem_param_src.metrics.sliced_wasserstein import normalize_directions
from mem_param_src.metrics.synthetic import intrinsic_geometry_parts
from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime, resolve_device, validate_run_kind
from mem_param_src.runners.artifacts import (
    create_run_record,
    file_sha256,
    load_torch,
    prepare_run_dir,
    read_json,
    save_checkpoint,
    save_data_manifest,
    save_torch,
    update_run_record,
    validate_data_manifest,
    write_json,
)
from mem_param_src.runners.recovery import (
    StatefulBatchSampler,
    StopRequest,
    acquire_training_lock,
    load_latest_recovery,
    preserve_rng_state,
    release_training_lock,
    restore_rng_state,
    save_recovery_state,
)
from mem_param_src.runners.tracking import (
    finish_wandb,
    log_wandb,
    resolve_wandb_identity,
    start_wandb,
)


CHECKPOINT_SCHEDULES = ("interval", "bonnaire-1m")


def _parse_checkpoint_steps(value):
    """Parse configured checkpoint steps."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ()
        parts = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError("training.ckpt_steps must contain integers.")
        value = parts

    steps = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("Checkpoint steps must be integers, not booleans.")
        try:
            steps.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid checkpoint step: {item!r}") from exc
    return tuple(steps)


def _finalize_checkpoint_steps(steps, n_steps):
    """Validate checkpoint steps and add step zero and the final step."""
    resolved = {0, int(n_steps)}
    for step in steps:
        step = int(step)
        if step < 0 or step > int(n_steps):
            raise ValueError(f"Checkpoint step {step} is outside [0, {int(n_steps)}].")
        resolved.add(step)
    return tuple(sorted(resolved))


def _snapped_logspace(start_power, end_power, count, modulus):
    """Return log-spaced steps snapped down to one integer modulus."""
    if count == 1:
        values = [10.0**start_power]
    else:
        values = [
            10.0 ** (start_power + i * (end_power - start_power) / (count - 1))
            for i in range(count)
        ]
    return sorted({int(value - value % modulus) for value in values})


def _bonnaire_checkpoint_steps(n_steps):
    """Return the legacy dense-to-sparse schedule up to one million steps."""
    if n_steps > 1_000_000:
        raise ValueError(
            "The bonnaire-1m schedule requires at most 1000000 training steps."
        )
    phase_one = _snapped_logspace(math.log10(251), 4.0, 10, 250)
    phase_two = _snapped_logspace(4.0, 6.0, 90, 5000)
    return _finalize_checkpoint_steps(
        [step for step in phase_one + phase_two if step <= n_steps],
        n_steps,
    )


def resolve_checkpoint_steps(config):
    """Resolve explicit, interval, or Bonnaire checkpoint steps."""
    n_steps = int(config.training.n_training_steps)
    if n_steps < 1:
        raise ValueError("training.n_training_steps must be at least 1.")

    explicit = _parse_checkpoint_steps(config.training.ckpt_steps)
    if explicit:
        return _finalize_checkpoint_steps(explicit, n_steps)

    schedule = str(config.training.ckpt_schedule)
    if schedule == "bonnaire-1m":
        return _bonnaire_checkpoint_steps(n_steps)
    if schedule != "interval":
        raise ValueError(
            f"Unknown checkpoint schedule {schedule!r}. "
            f"Expected one of: {', '.join(CHECKPOINT_SCHEDULES)}."
        )

    interval = int(config.training.ckpt_every)
    if interval < 1:
        raise ValueError("training.ckpt_every must be at least 1.")
    return _finalize_checkpoint_steps(range(0, n_steps + 1, interval), n_steps)


def _validate_bundle(bundle, run_kind, config):
    """Validate tensor shapes required by the shared training and replay paths."""
    train = bundle["train"]
    reference = bundle["reference"]
    expected_rank = 4 if run_kind == "image" else 2
    if train.dim() != expected_rank or reference.dim() != expected_rank:
        raise ValueError(
            f"The {run_kind} runner expects rank-{expected_rank} train and "
            f"reference tensors, got {train.dim()} and {reference.dim()}."
        )
    if train.shape[1:] != reference.shape[1:]:
        raise ValueError(
            "Train and reference tensors must have matching sample shapes."
        )
    if train.shape[0] < 2:
        raise ValueError("Memorization metrics require at least two training samples.")
    if not train.is_floating_point() or not reference.is_floating_point():
        raise ValueError("Train and reference tensors must use floating-point values.")

    n_samples = int(config.evaluation.n_samples)
    sample_bank_size = int(config.evaluation.sample_bank_size)
    if n_samples < 1:
        raise ValueError("evaluation.n_samples must be at least 1.")
    if sample_bank_size < n_samples:
        raise ValueError(
            "evaluation.sample_bank_size must be at least evaluation.n_samples."
        )
    if reference.shape[0] < n_samples:
        raise ValueError(
            "The saved reference tensor must contain at least evaluation.n_samples."
        )
    if run_kind == "synthetic" and "geometry" not in bundle:
        raise ValueError("Synthetic runs require geometry in the data bundle.")
    if run_kind == "image":
        validate_finite_data_identity(bundle)


def _build_optimizer(model, config):
    """Build the configured Adam or SGD optimizer."""
    name = str(config.training.optimizer).lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config.training.lr),
            momentum=float(config.training.momentum),
            weight_decay=float(config.training.weight_decay),
        )
    raise ValueError("training.optimizer must be 'adam' or 'sgd'.")


def _seed_process_rngs(seed):
    """Initialize every process-global RNG captured by recovery states."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _make_z_init(config, train_tensor):
    """Create the fixed evaluation noise bank without changing training RNG state."""
    generator = torch.Generator().manual_seed(int(config.training.seed) + 2003)
    shape = (int(config.evaluation.sample_bank_size), *train_tensor.shape[1:])
    return torch.randn(shape, generator=generator, dtype=train_tensor.dtype)


def _make_sliced_directions(config, bundle):
    """Create fixed directions in the dataset's intrinsic metric space."""
    intrinsic = intrinsic_geometry_parts(
        bundle["train"], bundle["geometry"], config=config
    )["intrinsic"]
    count = int(config.evaluation.sliced_w2_n_directions)
    if count < 1:
        raise ValueError("evaluation.sliced_w2_n_directions must be at least 1.")
    generator = torch.Generator().manual_seed(int(config.training.seed) + 1009)
    directions = torch.randn(count, intrinsic.shape[1], generator=generator)
    return normalize_directions(directions)


def _record_profiled_image_monitor(
    model,
    z_init,
    config,
    device,
    run_dir,
    wandb_run,
    step,
    profiles,
    f_mem_state,
    train_preview,
):
    """Record named sampler comparisons from one fixed checkpoint and noise bank."""
    from mem_param_src.runners.image_monitoring import (
        compute_monitor_f_mem,
        compute_raw_sample_diagnostics,
        generate_monitor_samples,
        save_fixed_range_grid,
    )
    from mem_param_src.runners.image_plotting import (
        save_monitor_curves,
        save_training_comparison_grid,
    )

    monitor_dir = Path(run_dir) / "monitoring"
    count = int(config.generation_monitor.n_samples)
    if f_mem_state is not None:
        count = max(count, int(config.random_projected_f_mem.n_samples))
    print(
        f"[monitor] Step {step}/{config.training.n_training_steps}: "
        f"generating {count} fixed-noise samples for {len(profiles)} profiles...",
        flush=True,
    )
    values = {
        "optimizer_step": int(step),
        "generation/step": int(step),
    }
    for profile in profiles:
        name = profile["name"]
        print(
            f"[monitor:{name}] eps_to_v={profile['eps_to_v']}, "
            f"safe_div={profile['use_safe_div']}, "
            f"floor={profile['safe_div_denominator_floor']:g}",
            flush=True,
        )
        samples, raw_samples = generate_monitor_samples(
            model,
            z_init,
            config,
            device,
            return_raw=True,
            sampling_profile=profile,
            allow_nonfinite=True,
        )
        if bool(config.evaluation.save_generated_samples):
            sample_path = (
                monitor_dir
                / "generated_samples"
                / name
                / f"step_{int(step):06d}.pt"
            )
            # The study disables final clipping, so this is the exact raw sampler output.
            save_torch(sample_path, raw_samples)
            print(
                f"[saved:{name}] {len(raw_samples)} raw generated samples: "
                f"{sample_path}",
                flush=True,
            )

        diagnostics = compute_raw_sample_diagnostics(
            raw_samples,
            step=step,
            profile=profile,
        )
        diagnostics_path = (
            monitor_dir
            / "sample_diagnostics"
            / name
            / f"step_{int(step):06d}.json"
        )
        write_json(diagnostics_path, diagnostics)
        generation_prefix = f"generation/{name}"
        values.update(
            {
                f"{generation_prefix}/success": int(
                    diagnostics["status"] == "ok"
                ),
                f"{generation_prefix}/nonfinite_count": diagnostics[
                    "nonfinite_count"
                ],
                f"{generation_prefix}/nonfinite_fraction": diagnostics[
                    "nonfinite_fraction"
                ],
            }
        )
        for key in ("raw_min", "raw_max", "raw_std", "fraction_abs_gt_1"):
            if diagnostics[key] is not None:
                values[f"{generation_prefix}/{key}"] = diagnostics[key]

        if diagnostics["status"] != "ok":
            message = (
                f"Generation-monitor profile {name!r} produced "
                f"{diagnostics['nonfinite_count']}/{diagnostics['total_values']} "
                "nonfinite values."
            )
            print(f"warning: {message}", flush=True)
            if profile["required"]:
                log_wandb(wandb_run, values)
                raise FloatingPointError(message)
            continue

        grid_samples = samples[: int(config.generation_monitor.n_samples)]
        if train_preview is not None:
            grid_path = save_training_comparison_grid(
                train_preview,
                grid_samples,
                monitor_dir
                / "comparison_grids"
                / name
                / f"step_{int(step):06d}.png",
                step,
            )
        else:
            grid_path = save_fixed_range_grid(
                grid_samples,
                monitor_dir
                / "sample_grids"
                / name
                / f"step_{int(step):06d}.png",
            )
        values[f"{generation_prefix}/samples"] = grid_path

        if f_mem_state is not None:
            print(
                f"[monitor:{name}] Samples ready; computing projected f_mem...",
                flush=True,
            )
            metric_path = f"monitor/random_projected_pixels/{name}"
            record = {
                "step": int(step),
                "profile": name,
                "metric_path": metric_path,
                **compute_monitor_f_mem(samples, config, f_mem_state),
            }
            record_path = (
                monitor_dir / "f_mem" / name / f"step_{int(step):06d}.json"
            )
            write_json(record_path, record)
            print(
                f"projected f_mem profile={name} step={step} "
                f"n={record['n_samples']} N={record['N']} "
                f"f_mem={record['f_mem']:.6f} mean_r={record['mean_r']:.6f} "
                f"rho={record['rho']:g}",
                flush=True,
            )
            values.update(
                {
                    f"{metric_path}/{key}": record[key]
                    for key in ("f_mem", "mean_r", "rho", "n_samples", "N")
                }
            )

    save_monitor_curves(run_dir)
    print(f"[saved] Profile grids and available curves: {monitor_dir}", flush=True)
    return log_wandb(wandb_run, values)


def _record_image_monitor(
    model,
    z_init,
    config,
    device,
    run_dir,
    wandb_run,
    step,
    f_mem_state=None,
    train_preview=None,
):
    """Save and optionally log one fixed-noise checkpoint panel."""
    from mem_param_src.runners.image_monitoring import (
        compute_monitor_f_mem,
        generate_monitor_samples,
        resolve_monitor_sampling_profiles,
        save_fixed_range_grid,
    )
    from mem_param_src.runners.image_plotting import (
        save_monitor_curves,
        save_training_comparison_grid,
    )

    profiles = resolve_monitor_sampling_profiles(config)
    if profiles:
        return _record_profiled_image_monitor(
            model,
            z_init,
            config,
            device,
            run_dir,
            wandb_run,
            step,
            profiles,
            f_mem_state,
            train_preview,
        )

    monitor_dir = Path(run_dir) / "monitoring"
    # Generate once; the grid and optional metric use prefixes of this batch.
    count = int(config.generation_monitor.n_samples)
    if f_mem_state is not None:
        count = max(count, int(config.random_projected_f_mem.n_samples))
    division = "exact division"
    if config.sampling.use_safe_div:
        division = f"denominator floor {config.sampling.safe_div_denominator_floor:g}"
    solver = str(config.sampling.method)
    if solver == "heun" and config.sampling.get("heun_final_euler", False):
        solver = "heun with final Euler"
    conversion = ""
    if config.framework.prediction_space == "eps":
        conversion = f", eps to v: {config.sampling.get('eps_to_v', 'via_x')}"
    print(
        f"[monitor] Step {step}/{config.training.n_training_steps}: "
        f"generating {count} samples ({solver}, "
        f"{config.sampling.n_steps} steps, "
        f"t=[{config.sampling.start_time:g},{config.sampling.end_time:g}], "
        f"{division}{conversion})...",
        flush=True,
    )
    final_pixel_clamp = bool(config.sampling.get("final_pixel_clamp", False))
    if final_pixel_clamp:
        samples, raw_samples = generate_monitor_samples(
            model, z_init, config, device, return_raw=True
        )
    else:
        samples = generate_monitor_samples(model, z_init, config, device)
        raw_samples = None
    if bool(config.evaluation.save_generated_samples):
        # Retain the full raw batch before display or metric preprocessing.
        sample_path = monitor_dir / "generated_samples" / f"step_{int(step):06d}.pt"
        save_torch(sample_path, samples)
        print(f"[saved] {len(samples)} generated samples: {sample_path}", flush=True)
        if raw_samples is not None:
            raw_path = monitor_dir / "generated_samples_raw" / f"step_{int(step):06d}.pt"
            save_torch(raw_path, raw_samples)
            print(f"[saved] {len(raw_samples)} raw generated samples: {raw_path}", flush=True)

    grid_samples = samples[: int(config.generation_monitor.n_samples)]
    if train_preview is not None:
        sample_grid_path = save_training_comparison_grid(
            train_preview,
            grid_samples,
            monitor_dir / "comparison_grids" / f"step_{int(step):06d}.png",
            step,
        )
    else:
        sample_grid_path = save_fixed_range_grid(
            grid_samples,
            monitor_dir / f"samples_step_{int(step):06d}.png",
        )
    values = {
        "optimizer_step": int(step),
        "generation/step": int(step),
        "generation/samples": sample_grid_path,
    }
    if raw_samples is not None:
        clipped = (raw_samples < -1.0) | (raw_samples > 1.0)
        pixel_clamp = {
            "step": int(step),
            "raw_min": float(raw_samples.min().item()),
            "raw_max": float(raw_samples.max().item()),
            "clipped_pixels": int(clipped.sum().item()),
            "total_pixels": int(raw_samples.numel()),
            "clipped_fraction": float(clipped.float().mean().item()),
        }
        write_json(monitor_dir / f"pixel_clamp_step_{int(step):06d}.json", pixel_clamp)
        values.update(
            {
                "generation/pixel_clamp_raw_min": pixel_clamp["raw_min"],
                "generation/pixel_clamp_raw_max": pixel_clamp["raw_max"],
                "generation/pixel_clamp_fraction": pixel_clamp["clipped_fraction"],
            }
        )

    if f_mem_state is not None:
        print("[monitor] Samples ready; computing projected f_mem...", flush=True)
        record = {
            "step": int(step),
            "metric_path": "monitor/random_projected_pixels",
            **compute_monitor_f_mem(samples, config, f_mem_state),
        }
        write_json(monitor_dir / f"f_mem_step_{int(step):06d}.json", record)
        print(
            f"projected f_mem step={step} n={record['n_samples']} N={record['N']} "
            f"f_mem={record['f_mem']:.6f} mean_r={record['mean_r']:.6f} "
            f"rho={record['rho']:g}",
            flush=True,
        )
        values.update(
            {
                f"monitor/random_projected_pixels/{key}": record[key]
                for key in ("f_mem", "mean_r", "rho", "n_samples", "N")
            }
        )
    save_monitor_curves(run_dir)
    print(f"[saved] Grid and available curves: {monitor_dir}", flush=True)
    return log_wandb(wandb_run, values)


def _attempt_record(run_dir, *, resumed, start_step):
    """Append one process attempt to the run record."""
    record = read_json(Path(run_dir) / "run.json")
    attempts = list(record.get("attempts", []))
    attempt = {
        "index": len(attempts),
        "resumed": bool(resumed),
        "start_step": int(start_step),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
    }
    attempts.append(attempt)
    update_run_record(
        run_dir,
        attempts=attempts,
        attempt_count=len(attempts),
        current_attempt=attempt["index"],
        resumed_from_step=int(start_step) if resumed else None,
        error=None,
    )


def _finish_attempt(run_dir, *, state, step, reason=None):
    """Close the current process attempt in the run record."""
    record = read_json(Path(run_dir) / "run.json")
    attempts = list(record.get("attempts", []))
    if attempts:
        attempts[-1] = {
            **attempts[-1],
            "state": str(state),
            "end_step": int(step),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
    update_run_record(
        run_dir,
        attempts=attempts,
        state=str(state),
        latest_step=int(step),
        stop_reason=reason,
    )


def _resolved_batch_settings(config, n_train):
    """Resolve physical and effective batches for one optimizer update."""
    batch_cap = int(config.training.batch_cap)
    if batch_cap < 1:
        raise ValueError("training.batch_cap must be at least 1.")
    physical = min(int(n_train), batch_cap)
    configured_effective = int(config.training.get("effective_batch_size", 0))
    effective = configured_effective or physical
    if effective < physical or effective % physical:
        raise ValueError(
            "training.effective_batch_size must be zero or a positive multiple "
            "of the resolved physical batch size."
        )
    accumulation_steps = effective // physical
    if accumulation_steps > 1 and int(n_train) % physical:
        raise ValueError(
            "Gradient accumulation requires N to be divisible by the physical "
            "batch size so every microbatch has equal weight."
        )
    config.training.batch_size = physical
    config.training.effective_batch_size = effective
    config.training.accumulation_steps = accumulation_steps
    return physical, effective, accumulation_steps


def _saved_f_mem_state(run_dir):
    """Load the fixed projected-pixel monitor state for a resumed run."""
    monitor_dir = Path(run_dir) / "monitoring"
    return {
        "projection": load_torch(monitor_dir / "f_mem_projection.pt"),
        "reference": load_torch(monitor_dir / "f_mem_train_vectors.pt"),
    }


def _monitor_exists(run_dir, step, train_preview, config):
    """Return whether the fixed-noise panel for one step is already saved."""
    monitor_dir = Path(run_dir) / "monitoring"
    from mem_param_src.runners.image_monitoring import resolve_monitor_sampling_profiles

    profiles = resolve_monitor_sampling_profiles(config)
    if profiles:
        for profile in profiles:
            path = (
                monitor_dir
                / "sample_diagnostics"
                / profile["name"]
                / f"step_{int(step):06d}.json"
            )
            if not path.is_file():
                return False
            record = read_json(path)
            status = record.get("status")
            if status not in ("ok", "nonfinite"):
                return False
            if profile["required"] and status != "ok":
                return False
            if bool(config.evaluation.save_generated_samples):
                sample_path = (
                    monitor_dir
                    / "generated_samples"
                    / profile["name"]
                    / f"step_{int(step):06d}.pt"
                )
                if not sample_path.is_file():
                    return False
            if status == "ok":
                grid_folder = (
                    "comparison_grids" if train_preview is not None else "sample_grids"
                )
                grid_path = (
                    monitor_dir
                    / grid_folder
                    / profile["name"]
                    / f"step_{int(step):06d}.png"
                )
                if not grid_path.is_file():
                    return False
                if bool(config.random_projected_f_mem.enabled):
                    f_mem_path = (
                        monitor_dir
                        / "f_mem"
                        / profile["name"]
                        / f"step_{int(step):06d}.json"
                    )
                    if not f_mem_path.is_file():
                        return False
        return True
    if train_preview is not None:
        path = monitor_dir / "comparison_grids" / f"step_{int(step):06d}.png"
    else:
        path = monitor_dir / f"samples_step_{int(step):06d}.png"
    return path.is_file()


def train_run(
    config,
    run_dir,
    *,
    run_kind,
    device="auto",
    resume=False,
    stop_after_step=None,
):
    """Train or exactly resume one run and save replay artifacts."""
    validate_run_kind(config, run_kind)
    if run_kind != "image" and bool(config.tracking.wandb.enabled):
        raise ValueError("W&B training logs currently support image runs only.")
    if run_kind != "image" and bool(config.generation_monitor.enabled):
        raise ValueError("Generation monitoring currently supports image runs only.")
    n_train_preview = int(config.generation_monitor.n_train_samples)
    if n_train_preview < 0:
        raise ValueError("generation_monitor.n_train_samples must be nonnegative.")
    if n_train_preview and (run_kind != "image" or not config.generation_monitor.enabled):
        raise ValueError("Training comparison grids require image generation monitoring.")
    f_mem_settings = config.random_projected_f_mem
    if bool(config.generation_monitor.enabled):
        from mem_param_src.runners.image_monitoring import (
            resolve_monitor_sampling_profiles,
        )

        resolve_monitor_sampling_profiles(config)
    if bool(f_mem_settings.enabled):
        if run_kind != "image" or not bool(config.generation_monitor.enabled):
            raise ValueError("f_mem monitoring requires image generation monitoring.")
        if int(f_mem_settings.n_samples) < 1:
            raise ValueError("random_projected_f_mem.n_samples must be at least 1.")
        if f_mem_settings.projection_seed is None:
            f_mem_settings.projection_seed = int(config.training.seed) + 3001
        config.evaluation.sample_bank_size = max(
            int(config.evaluation.sample_bank_size),
            int(f_mem_settings.n_samples),
            int(config.generation_monitor.n_samples),
        )
    device = resolve_device(device)
    configure_runtime(config, device)
    _seed_process_rngs(config.training.seed)
    run_dir = Path(run_dir).expanduser().resolve()
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Saved run directory does not exist: {run_dir}")
        record = read_json(run_dir / "run.json")
        if str(record.get("run_kind")) != str(run_kind):
            raise ValueError("Saved run kind does not match the requested runner.")
    else:
        run_dir = prepare_run_dir(run_dir)
        create_run_record(
            run_dir,
            run_kind=run_kind,
            seed=int(config.training.seed),
            device=device,
        )
    show_stages = (
        run_kind == "image"
        and bool(config.tracking.console.enabled)
        and bool(config.tracking.console.stage_banners)
    )

    wandb_run = None
    success = False
    lock_handle = acquire_training_lock(run_dir)
    stop_request = StopRequest().install()
    latest_step = 0
    try:
        recovery = None
        if resume:
            latest_step, recovery_path, recovery = load_latest_recovery(run_dir)
            bundle = load_torch(run_dir / "data_bundle.pt")
            validate_data_manifest(
                run_dir,
                bundle,
                run_kind=run_kind,
                verify_hashes=True,
            )
            if show_stages:
                print(
                    f"[resume] Loaded step {latest_step:,} from {recovery_path}.",
                    flush=True,
                )
        else:
            if show_stages:
                print(
                    f"[data] Loading {config.data.dataset} (training device: {device})...",
                    flush=True,
                )
            bundle = build_data_bundle(config)
        _validate_bundle(bundle, run_kind, config)
        train_tensor = bundle["train"].detach().cpu().contiguous()
        if show_stages:
            print(
                f"[data] Ready: {len(train_tensor)} training, "
                f"{len(bundle['reference'])} held-out; shape {tuple(train_tensor.shape[1:])}.",
                flush=True,
            )
        # Fix the displayed subset in saved tensor order; do not draw fresh examples.
        train_preview = train_tensor[:n_train_preview] if n_train_preview else None
        if bool(f_mem_settings.enabled) and train_tensor.shape[0] < 2:
            raise ValueError("f_mem monitoring requires at least two training images.")

        physical_batch, effective_batch, accumulation_steps = _resolved_batch_settings(
            config,
            len(train_tensor),
        )
        checkpoint_steps = resolve_checkpoint_steps(config)
        config.training.resolved_ckpt_steps = checkpoint_steps
        if run_kind == "image" and bool(config.tracking.wandb.enabled):
            resolve_wandb_identity(config)

        log_every = int(config.training.log_every)
        if log_every < 1:
            raise ValueError("training.log_every must be at least 1.")

        if resume:
            z_init = load_torch(run_dir / "z_init.pt")
            history = list(recovery["history"])
            write_json(run_dir / "train_history.json", history)
            f_mem_state = (
                _saved_f_mem_state(run_dir) if bool(f_mem_settings.enabled) else None
            )
        else:
            if show_stages:
                print("[setup] Saving data and preparing the model...", flush=True)
            write_json(run_dir / "config.json", config_to_dict(config))
            save_torch(run_dir / "data_bundle.pt", bundle)
            manifest_path, manifest = save_data_manifest(
                run_dir,
                config,
                bundle,
                run_kind=run_kind,
            )
            update_run_record(
                run_dir,
                data_bundle_sha256=manifest["bundle"]["sha256"],
                data_manifest_sha256=file_sha256(manifest_path),
            )
            z_init = _make_z_init(config, train_tensor)
            save_torch(run_dir / "z_init.pt", z_init)
            history = []
            write_json(run_dir / "train_history.json", history)
            f_mem_state = None
            if bool(f_mem_settings.enabled):
                from mem_param_src.runners.image_monitoring import prepare_f_mem_monitor

                f_mem_state = prepare_f_mem_monitor(config, train_tensor, run_dir)
            if run_kind == "synthetic":
                directions = _make_sliced_directions(config, bundle)
                save_torch(run_dir / "sliced_directions.pt", directions)

        model = GDFM(config).to(device)
        optimizer = _build_optimizer(model, config)
        if resume:
            model.load_state_dict(recovery["model"])
            optimizer.load_state_dict(recovery["optimizer"])
        if show_stages:
            print(
                f"[model] {config.network.arch}, {model.n_params():,} parameters; "
                f"{config.framework.prediction_space}/{config.framework.loss_space}, "
                f"physical batch {physical_batch}, effective batch {effective_batch}.",
                flush=True,
            )
        if not resume:
            save_checkpoint(run_dir, 0, model)
        update_run_record(
            run_dir,
            state="training",
            parameter_count=int(model.n_params()),
            n_train=int(train_tensor.shape[0]),
            n_reference=int(bundle["reference"].shape[0]),
        )
        if run_kind == "image":
            resume_run_id = recovery.get("wandb_run_id") if resume else None
            with preserve_rng_state():
                wandb_run = start_wandb(
                    config,
                    run_dir,
                    resume_run_id=resume_run_id,
                )
            if not resume and bool(config.generation_monitor.enabled):
                with preserve_rng_state():
                    wandb_run = _record_image_monitor(
                        model,
                        z_init,
                        config,
                        device,
                        run_dir,
                        wandb_run,
                        0,
                        f_mem_state,
                        train_preview,
                    )

        sampler_state = recovery["batch_sampler"] if resume else None
        batch_sampler = StatefulBatchSampler(
            len(train_tensor),
            physical_batch,
            int(config.training.seed),
            state=sampler_state,
        )
        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_sampler=batch_sampler,
            generator=torch.Generator().manual_seed(int(config.training.seed) + 5001),
            num_workers=0,
            pin_memory=(bool(config.runtime.pin_memory) and device.type == "cuda"),
        )
        loader_iterator = iter(loader)
        if resume:
            restore_rng_state(recovery["rng"])
        non_blocking = bool(config.runtime.non_blocking_h2d) and device.type == "cuda"
        checkpoint_set = set(checkpoint_steps)
        n_steps = int(config.training.n_training_steps)
        stop_step = n_steps if stop_after_step is None else int(stop_after_step)
        if stop_step < 1 or stop_step > n_steps:
            raise ValueError("stop_after_step must be inside the configured training range.")
        if latest_step >= stop_step:
            raise ValueError(
                f"Recovery step {latest_step} has already reached stop step {stop_step}."
            )
        recovery_every = int(config.training.get("recovery_every", 0))
        if recovery_every < 0:
            raise ValueError("training.recovery_every must be nonnegative.")
        elapsed_before = float(recovery.get("elapsed_seconds", 0.0)) if resume else 0.0
        training_start = time.perf_counter()
        _attempt_record(run_dir, resumed=resume, start_step=latest_step)

        latest_checkpoint = run_dir / "checkpoints" / f"step_{int(latest_step):06d}.pt"
        if resume and (latest_step in checkpoint_set or latest_checkpoint.is_file()):
            if bool(config.generation_monitor.enabled) and not _monitor_exists(
                run_dir,
                latest_step,
                train_preview,
                config,
            ):
                with preserve_rng_state():
                    wandb_run = _record_image_monitor(
                        model,
                        z_init,
                        config,
                        device,
                        run_dir,
                        wandb_run,
                        latest_step,
                        f_mem_state,
                        train_preview,
                    )

        if not resume:
            wandb_id = wandb_run["metadata"]["run_id"] if wandb_run else None
            save_recovery_state(
                run_dir,
                step=0,
                model=model,
                optimizer=optimizer,
                batch_sampler=batch_sampler,
                history=history,
                elapsed_seconds=0.0,
                wandb_run_id=wandb_id,
            )

        if show_stages:
            print(
                f"[train] Updates {latest_step + 1:,} through {stop_step:,} "
                f"of {n_steps:,}...",
                flush=True,
            )
        model.train()
        for step in range(latest_step + 1, stop_step + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(accumulation_steps):
                (batch,) = next(loader_iterator)
                batch = batch.to(device, non_blocking=non_blocking)
                loss = model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Training loss became non-finite at optimizer step {step}."
                    )
                (loss / accumulation_steps).backward()
                loss_value += float(loss.detach().cpu().item()) / accumulation_steps
            optimizer.step()

            should_record = step == 1 or step % log_every == 0 or step == stop_step
            if should_record:
                history.append({"step": int(step), "loss": loss_value})
                if bool(config.training.show_progress):
                    print(f"step {step}/{n_steps} loss={loss_value:.6f}", flush=True)
            should_checkpoint = step in checkpoint_set or step == stop_step
            if should_checkpoint:
                save_checkpoint(run_dir, step, model)
            should_recover = (
                step == 1
                or step in checkpoint_set
                or step == stop_step
                or stop_request.requested
                or (recovery_every and step % recovery_every == 0)
            )
            if should_recover:
                write_json(run_dir / "train_history.json", history)
                elapsed = elapsed_before + time.perf_counter() - training_start
                wandb_id = wandb_run["metadata"]["run_id"] if wandb_run else None
                save_recovery_state(
                    run_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    batch_sampler=batch_sampler,
                    history=history,
                    elapsed_seconds=elapsed,
                    wandb_run_id=wandb_id,
                )
                if run_kind == "image":
                    with preserve_rng_state():
                        wandb_run = log_wandb(
                            wandb_run,
                            {
                                "optimizer_step": int(step),
                                "train/step": int(step),
                                "train/loss": loss_value,
                                "train/learning_rate": float(
                                    optimizer.param_groups[0]["lr"]
                                ),
                                "train/elapsed_seconds": elapsed,
                                "train/progress": float(step) / float(n_steps),
                                "train/attempt": int(
                                    read_json(run_dir / "run.json")[
                                        "current_attempt"
                                    ]
                                ),
                            },
                        )
            if stop_request.requested:
                reason = f"received {stop_request.signal_name}"
                _finish_attempt(run_dir, state="interrupted", step=step, reason=reason)
                if show_stages:
                    print(f"[train] Interrupted after step {step}: {reason}.", flush=True)
                return Path(run_dir)
            if should_checkpoint:
                if run_kind == "image" and bool(config.generation_monitor.enabled):
                    with preserve_rng_state():
                        wandb_run = _record_image_monitor(
                            model,
                            z_init,
                            config,
                            device,
                            run_dir,
                            wandb_run,
                            step,
                            f_mem_state,
                            train_preview,
                        )
                    if show_stages and step < stop_step:
                        print(
                            f"[train] Continuing from step {step}/{stop_step}...",
                            flush=True,
                        )

        write_json(run_dir / "train_history.json", history)
        if stop_step < n_steps:
            _finish_attempt(
                run_dir,
                state="paused",
                step=stop_step,
                reason="planned stop_after_step",
            )
            if show_stages:
                print(f"[train] Paused at planned step {stop_step:,}.", flush=True)
        else:
            _finish_attempt(run_dir, state="trained", step=n_steps)
            update_run_record(run_dir, final_step=n_steps)
            if show_stages:
                print(f"[train] Finished {n_steps:,} updates.", flush=True)
        success = True
        return Path(run_dir)
    except Exception as exc:
        update_run_record(run_dir, state="failed", error=str(exc))
        raise
    finally:
        stop_request.restore()
        finish_wandb(wandb_run, success)
        release_training_lock(lock_handle)


__all__ = ["CHECKPOINT_SCHEDULES", "resolve_checkpoint_steps", "train_run"]
