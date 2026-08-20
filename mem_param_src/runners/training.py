"""Shared optimizer loop for image and synthetic runs."""

import math
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


def train_run(config, run_dir, *, run_kind, device="auto"):
    """Train one image or synthetic run and save its replay artifacts."""
    validate_run_kind(config, run_kind)
    device = resolve_device(device)
    configure_runtime(config, device)
    run_dir = prepare_run_dir(run_dir)
    create_run_record(
        run_dir,
        run_kind=run_kind,
        seed=int(config.training.seed),
        device=device,
    )

    try:
        bundle = build_data_bundle(config)
        _validate_bundle(bundle, run_kind, config)
        train_tensor = bundle["train"].detach().cpu().contiguous()

        batch_cap = int(config.training.batch_cap)
        if batch_cap < 1:
            raise ValueError("training.batch_cap must be at least 1.")
        config.training.batch_size = min(int(train_tensor.shape[0]), batch_cap)
        checkpoint_steps = resolve_checkpoint_steps(config)
        config.training.resolved_ckpt_steps = checkpoint_steps

        log_every = int(config.training.log_every)
        if log_every < 1:
            raise ValueError("training.log_every must be at least 1.")

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
        if run_kind == "synthetic":
            directions = _make_sliced_directions(config, bundle)
            save_torch(run_dir / "sliced_directions.pt", directions)

        torch.manual_seed(int(config.training.seed))
        model = GDFM(config).to(device)
        optimizer = _build_optimizer(model, config)
        save_checkpoint(run_dir, 0, model)
        update_run_record(
            run_dir,
            state="training",
            parameter_count=int(model.n_params()),
            n_train=int(train_tensor.shape[0]),
            n_reference=int(bundle["reference"].shape[0]),
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
        history = []
        checkpoint_set = set(checkpoint_steps)
        n_steps = int(config.training.n_training_steps)

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
            if step in checkpoint_set:
                save_checkpoint(run_dir, step, model)

        write_json(run_dir / "train_history.json", history)
        update_run_record(run_dir, state="trained", final_step=n_steps)
        return Path(run_dir)
    except Exception as exc:
        update_run_record(run_dir, state="failed", error=str(exc))
        raise


__all__ = ["CHECKPOINT_SCHEDULES", "resolve_checkpoint_steps", "train_run"]
