"""Shared optimizer loop for image and synthetic runs."""

import math
import time
from pathlib import Path

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
    prepare_run_dir,
    save_checkpoint,
    save_data_manifest,
    save_torch,
    update_run_record,
    write_json,
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
        generate_monitor_samples,
        compute_monitor_f_mem,
        save_fixed_range_grid,
    )
    from mem_param_src.runners.image_plotting import (
        save_monitor_curves,
        save_training_comparison_grid,
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
                f"generation/random_projected_pixels/{key}": record[key]
                for key in ("f_mem", "mean_r", "rho", "n_samples", "N")
            }
        )
    save_monitor_curves(run_dir)
    print(f"[saved] Grid and available curves: {monitor_dir}", flush=True)
    return log_wandb(wandb_run, values)


def train_run(config, run_dir, *, run_kind, device="auto"):
    """Train one image or synthetic run and save its replay artifacts."""
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
    try:
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

        batch_cap = int(config.training.batch_cap)
        if batch_cap < 1:
            raise ValueError("training.batch_cap must be at least 1.")
        config.training.batch_size = min(int(train_tensor.shape[0]), batch_cap)
        checkpoint_steps = resolve_checkpoint_steps(config)
        config.training.resolved_ckpt_steps = checkpoint_steps
        if run_kind == "image" and bool(config.tracking.wandb.enabled):
            resolve_wandb_identity(config)

        log_every = int(config.training.log_every)
        if log_every < 1:
            raise ValueError("training.log_every must be at least 1.")

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

        torch.manual_seed(int(config.training.seed))
        model = GDFM(config).to(device)
        optimizer = _build_optimizer(model, config)
        if show_stages:
            print(
                f"[model] {config.network.arch}, {model.n_params():,} parameters; "
                f"{config.framework.prediction_space}/{config.framework.loss_space}, "
                f"batch size {config.training.batch_size}.",
                flush=True,
            )
        save_checkpoint(run_dir, 0, model)
        update_run_record(
            run_dir,
            state="training",
            parameter_count=int(model.n_params()),
            n_train=int(train_tensor.shape[0]),
            n_reference=int(bundle["reference"].shape[0]),
        )
        if run_kind == "image":
            wandb_run = start_wandb(config, run_dir)
            if bool(config.generation_monitor.enabled):
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

        loader_generator = torch.Generator().manual_seed(int(config.training.seed))
        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=int(config.training.batch_size),
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            pin_memory=(bool(config.runtime.pin_memory) and device.type == "cuda"),
        )
        loader_iterator = iter(loader)
        non_blocking = bool(config.runtime.non_blocking_h2d) and device.type == "cuda"
        checkpoint_set = set(checkpoint_steps)
        n_steps = int(config.training.n_training_steps)
        training_start = time.perf_counter()

        if show_stages:
            print(f"[train] Starting {n_steps:,} updates...", flush=True)
        model.train()
        for step in range(1, n_steps + 1):
            try:
                (batch,) = next(loader_iterator)
            except StopIteration:
                loader_iterator = iter(loader)
                (batch,) = next(loader_iterator)

            batch = batch.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Training loss became non-finite at optimizer step {step}."
                )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            should_record = step == 1 or step % log_every == 0 or step == n_steps
            if should_record:
                history.append({"step": int(step), "loss": loss_value})
                if bool(config.training.show_progress):
                    print(f"step {step}/{n_steps} loss={loss_value:.6f}", flush=True)
                if run_kind == "image":
                    # W&B observes completed updates and does not control optimization.
                    wandb_run = log_wandb(
                        wandb_run,
                        {
                            "optimizer_step": int(step),
                            "train/step": int(step),
                            "train/loss": loss_value,
                            "train/learning_rate": float(
                                optimizer.param_groups[0]["lr"]
                            ),
                            "train/elapsed_seconds": (
                                time.perf_counter() - training_start
                            ),
                            "train/progress": float(step) / float(n_steps),
                        },
                    )
            if step in checkpoint_set:
                save_checkpoint(run_dir, step, model)
                write_json(run_dir / "train_history.json", history)
                if run_kind == "image" and bool(config.generation_monitor.enabled):
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
                    if show_stages and step < n_steps:
                        print(
                            f"[train] Continuing from step {step}/{n_steps}...",
                            flush=True,
                        )

        write_json(run_dir / "train_history.json", history)
        update_run_record(run_dir, state="trained", final_step=n_steps)
        if show_stages:
            print(f"[train] Finished {n_steps:,} updates.", flush=True)
        success = True
        return Path(run_dir)
    except Exception as exc:
        update_run_record(run_dir, state="failed", error=str(exc))
        raise
    finally:
        finish_wandb(wandb_run, success)


__all__ = ["CHECKPOINT_SCHEDULES", "resolve_checkpoint_steps", "train_run"]
