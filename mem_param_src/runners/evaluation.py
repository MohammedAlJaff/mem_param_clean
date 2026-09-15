"""Standalone checkpoint replay for saved image and synthetic runs."""

from pathlib import Path

import torch
from ml_collections import ConfigDict

from mem_param_src.config import (
    apply_config_overlay,
    apply_overrides,
    config_to_dict,
    load_python_config,
    parse_cli_overrides,
)
from mem_param_src.metrics.image import collect_image32_metric_record
from mem_param_src.metrics.synthetic import (
    collect_controlled_synthetic_metric_record,
    intrinsic_geometry_parts,
)
from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime, resolve_device, validate_run_kind
from mem_param_src.runners.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    list_checkpoints,
    load_torch,
    prepare_evaluation_dir,
    read_json,
    save_torch,
    update_run_record,
    validate_data_manifest,
    write_json,
)


EVALUATION_CONFIG_SECTIONS = ("sampling", "evaluation")


def resolve_evaluation_config(run_dir, *, eval_config=None, override_items=()):
    """Overlay optional evaluation settings onto one saved run config."""
    run_dir = Path(run_dir).expanduser().resolve()
    config = ConfigDict(read_json(run_dir / "config.json"))
    # Older runs predate optional storage of every generated checkpoint batch.
    if "save_generated_samples" not in config.evaluation:
        config.evaluation.save_generated_samples = False
    # Old saved runs used chained epsilon conversion and Heun on every interval.
    # Materialize those choices so replay overrides and settings.json are explicit.
    if "eps_to_v" not in config.sampling:
        config.sampling.eps_to_v = "via_x"
    if "heun_final_euler" not in config.sampling:
        config.sampling.heun_final_euler = False
    saved_settings = {
        section: config_to_dict(config[section])
        for section in EVALUATION_CONFIG_SECTIONS
    }

    if eval_config is not None:
        overlay = load_python_config(
            eval_config,
            allowed_sections=EVALUATION_CONFIG_SECTIONS,
        )
        apply_config_overlay(
            config,
            overlay,
            allowed_sections=EVALUATION_CONFIG_SECTIONS,
        )

    overrides = parse_cli_overrides(
        override_items,
        allowed_sections=EVALUATION_CONFIG_SECTIONS,
    )
    apply_overrides(config, overrides)
    effective_settings = {
        section: config_to_dict(config[section])
        for section in EVALUATION_CONFIG_SECTIONS
    }
    return config, effective_settings != saved_settings


def _validate_evaluation_inputs(config, bundle, z_init, run_kind):
    """Check saved tensors and requested evaluation counts before replay."""
    train = bundle.get("train")
    reference = bundle.get("reference")
    if not torch.is_tensor(train) or not torch.is_tensor(reference):
        raise ValueError(
            "Saved data_bundle.pt must contain train and reference tensors."
        )

    expected_rank = 4 if run_kind == "image" else 2
    if train.dim() != expected_rank or reference.dim() != expected_rank:
        raise ValueError(
            f"The {run_kind} evaluator expects rank-{expected_rank} saved tensors."
        )
    if not torch.is_tensor(z_init) or z_init.dim() != expected_rank:
        raise ValueError("Saved z_init.pt does not match the run tensor rank.")
    if z_init.shape[1:] != train.shape[1:]:
        raise ValueError("Saved z_init and training samples have different shapes.")

    n_samples = int(config.evaluation.n_samples)
    if n_samples < 1:
        raise ValueError("evaluation.n_samples must be at least 1.")
    if z_init.shape[0] < n_samples:
        raise ValueError("Saved z_init is smaller than evaluation.n_samples.")
    if reference.shape[0] < n_samples:
        raise ValueError("Saved reference data is smaller than evaluation.n_samples.")
    if int(config.evaluation.batch_size) < 1:
        raise ValueError("evaluation.batch_size must be at least 1.")
    if int(config.evaluation.chunk_size) < 1:
        raise ValueError("evaluation.chunk_size must be at least 1.")
    if str(config.sampling.output) not in ("x_t_max", "x_1"):
        raise ValueError("sampling.output must be 'x_t_max' or 'x_1'.")
    if run_kind == "synthetic" and "geometry" not in bundle:
        raise ValueError("Synthetic evaluation requires saved geometry.")


def _sample_checkpoint(model, z_init, config, device):
    """Generate one checkpoint sample bank in fixed batches."""
    batch_size = int(config.evaluation.batch_size)
    non_blocking = bool(config.runtime.non_blocking_h2d) and device.type == "cuda"
    samples = []
    for start in range(0, z_init.shape[0], batch_size):
        z_batch = z_init[start : start + batch_size].to(
            device,
            non_blocking=non_blocking,
        )
        sample_batch = model.sample(
            z_batch,
            n_steps=int(config.sampling.n_steps),
            method=str(config.sampling.method),
            start_time=float(config.sampling.start_time),
            end_time=float(config.sampling.end_time),
            use_safe_div=bool(config.sampling.use_safe_div),
            safe_div_denominator_floor=float(config.sampling.safe_div_denominator_floor),
            output=str(config.sampling.output),
            eps_to_v=config.sampling.eps_to_v,
            heun_final_euler=config.sampling.heun_final_euler,
            final_pixel_clamp=bool(config.sampling.get("final_pixel_clamp", False)),
            device=device,
        )
        samples.append(sample_batch.detach().cpu())
    return torch.cat(samples, dim=0)


def _image_metric_record(step, samples, bundle, config):
    """Compute the current validated image memorization record."""
    return collect_image32_metric_record(
        step,
        samples,
        bundle["reference"],
        bundle["train"],
        chunk_size=int(config.evaluation.chunk_size),
        config=config,
    )


def _synthetic_metric_inputs(bundle, config, directions):
    """Build saved intrinsic inputs shared by every synthetic checkpoint."""
    geometry = bundle["geometry"]
    train_intrinsic = intrinsic_geometry_parts(
        bundle["train"], geometry, config=config
    )["intrinsic"]
    reference_intrinsic = intrinsic_geometry_parts(
        bundle["reference"], geometry, config=config
    )["intrinsic"]
    requested = int(config.evaluation.sliced_w2_n_directions)
    if requested < 1:
        raise ValueError("evaluation.sliced_w2_n_directions must be at least 1.")
    if directions.shape[0] < requested:
        raise ValueError(
            "Saved sliced directions are fewer than the requested evaluation count."
        )
    return train_intrinsic, reference_intrinsic, directions[:requested]


def _synthetic_metric_record(
    step,
    samples,
    bundle,
    config,
    train_intrinsic,
    reference_intrinsic,
    directions,
):
    """Compute the current controlled synthetic metric record."""
    return collect_controlled_synthetic_metric_record(
        step=step,
        samples=samples,
        baseline_samples=bundle["reference"][: samples.shape[0]],
        train_tensor=bundle["train"],
        train_intrinsic=train_intrinsic,
        geometry=bundle["geometry"],
        reference_intrinsic=reference_intrinsic,
        sliced_directions=directions,
        chunk_size=int(config.evaluation.chunk_size),
        config=config,
    )


def evaluate_run(
    run_dir,
    *,
    run_kind,
    eval_name=None,
    eval_config=None,
    override_items=(),
    device="auto",
):
    """Replay all checkpoints from one saved image or synthetic run."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_record = read_json(run_dir / "run.json")
    if int(run_record.get("artifact_format_version", -1)) != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Saved run uses an unsupported artifact format version.")
    if str(run_record.get("run_kind")) != run_kind:
        raise ValueError(
            f"Saved run kind is {run_record.get('run_kind')!r}, not {run_kind!r}."
        )

    config, settings_changed = resolve_evaluation_config(
        run_dir,
        eval_config=eval_config,
        override_items=override_items,
    )
    validate_run_kind(config, run_kind)
    if settings_changed and not eval_name:
        raise ValueError(
            "Evaluation settings differ from the saved run. Provide --eval-name."
        )
    eval_name = eval_name or "default"

    device = resolve_device(device)
    configure_runtime(config, device)
    bundle = load_torch(run_dir / "data_bundle.pt")
    validate_data_manifest(run_dir, bundle, run_kind=run_kind)
    z_init = load_torch(run_dir / "z_init.pt")
    _validate_evaluation_inputs(config, bundle, z_init, run_kind)
    z_eval = z_init[: int(config.evaluation.n_samples)]
    checkpoints = list_checkpoints(run_dir)

    synthetic_inputs = None
    if run_kind == "synthetic":
        directions = load_torch(run_dir / "sliced_directions.pt")
        synthetic_inputs = _synthetic_metric_inputs(
            bundle,
            config,
            directions,
        )

    eval_dir = prepare_evaluation_dir(run_dir, eval_name)
    write_json(
        eval_dir / "settings.json",
        {
            "eval_name": eval_name,
            "sampling": config_to_dict(config.sampling),
            "evaluation": config_to_dict(config.evaluation),
        },
    )

    model = GDFM(config).to(device)
    records = []
    final_samples = None
    for step, checkpoint_path in checkpoints:
        checkpoint = load_torch(checkpoint_path)
        if int(checkpoint.get("step", -1)) != step:
            raise ValueError(
                f"Checkpoint step does not match its path: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        samples = _sample_checkpoint(model, z_eval, config, device)
        if not torch.isfinite(samples).all():
            raise FloatingPointError(
                f"Generated samples became non-finite at checkpoint step {step}."
            )
        if run_kind == "image" and bool(config.evaluation.save_generated_samples):
            sample_path = eval_dir / "generated_samples" / f"step_{step:06d}.pt"
            save_torch(sample_path, samples)
            print(f"[saved] {len(samples)} generated samples: {sample_path}", flush=True)

        if run_kind == "image":
            record = _image_metric_record(step, samples, bundle, config)
        else:
            record = _synthetic_metric_record(
                step,
                samples,
                bundle,
                config,
                *synthetic_inputs,
            )
        record["checkpoint"] = checkpoint_path.name
        record["sampling_output"] = str(config.sampling.output)
        records.append(record)
        final_samples = samples

    write_json(eval_dir / "checkpoint_metrics.json", records)
    sample_name = f"final_samples_{config.sampling.output}.pt"
    save_torch(eval_dir / sample_name, final_samples)
    update_run_record(
        run_dir,
        state="evaluated",
        last_evaluation=eval_name,
    )
    return eval_dir


def evaluate_image_run(run_dir, **kwargs):
    """Replay one saved image run."""
    return evaluate_run(run_dir, run_kind="image", **kwargs)


def evaluate_synthetic_run(run_dir, **kwargs):
    """Replay one saved synthetic run."""
    return evaluate_run(run_dir, run_kind="synthetic", **kwargs)


__all__ = [
    "EVALUATION_CONFIG_SECTIONS",
    "evaluate_image_run",
    "evaluate_run",
    "evaluate_synthetic_run",
    "resolve_evaluation_config",
]
