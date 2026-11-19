"""Measure saved-checkpoint sampling throughput without computing metrics."""

import argparse
import csv
import json
import platform
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mem_param_src.config import config_to_dict
from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime, resolve_device
from mem_param_src.runners.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    file_sha256,
    list_checkpoints,
    load_torch,
    read_json,
    validate_data_manifest,
    write_json,
)
from mem_param_src.runners.evaluation import resolve_evaluation_config

SAMPLING_BENCHMARK_SCHEMA_VERSION = 1
_STOP_REQUESTED = False


def _utc_now():
    """Return one UTC timestamp for benchmark records."""
    return datetime.now(timezone.utc).isoformat()


def _request_stop(signum, frame):
    """Request a clean stop after the current complete sampling batch."""
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _synchronize(device):
    """Wait for queued CUDA work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_batch_csv(path, rows):
    """Replace the batch timing table through a temporary sibling file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = (
        "batch_index",
        "sample_start",
        "sample_stop",
        "sample_count",
        "seconds",
        "samples_per_second",
        "cumulative_samples",
        "cumulative_seconds",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_evaluation_noise(saved_noise, sample_count, seed):
    """Extend saved noise from its recorded seeded CPU random stream."""
    if not torch.is_tensor(saved_noise) or saved_noise.dim() < 2:
        raise ValueError("Saved z_init.pt must be a batched tensor.")
    sample_count = int(sample_count)
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    saved_noise = saved_noise.detach().cpu().contiguous()
    if sample_count <= len(saved_noise):
        return saved_noise[:sample_count].clone()

    generator = torch.Generator().manual_seed(int(seed))
    reproduced = torch.randn(
        saved_noise.shape,
        generator=generator,
        dtype=saved_noise.dtype,
    )
    if not torch.equal(reproduced, saved_noise):
        raise ValueError(
            "The recorded z_init seed does not reproduce the saved noise bank."
        )
    tail_shape = (sample_count - len(saved_noise), *saved_noise.shape[1:])
    tail = torch.randn(tail_shape, generator=generator, dtype=saved_noise.dtype)
    return torch.cat((saved_noise, tail), dim=0)


def summarize_sampling_runtime(measured_samples, sampling_seconds, target_count, checkpoint_count):
    """Project one-checkpoint and full-run sampling time from completed batches."""
    measured_samples = int(measured_samples)
    target_count = int(target_count)
    checkpoint_count = int(checkpoint_count)
    sampling_seconds = float(sampling_seconds)
    if measured_samples < 1 or sampling_seconds <= 0.0:
        raise ValueError("At least one timed sample and a positive duration are required.")
    if target_count < 1 or checkpoint_count < 1:
        raise ValueError("target_count and checkpoint_count must be positive.")

    seconds_per_sample = sampling_seconds / measured_samples
    one_checkpoint_seconds = seconds_per_sample * target_count
    all_checkpoints_seconds = one_checkpoint_seconds * checkpoint_count
    timing_kind = "measured" if measured_samples == target_count else "projected"
    return {
        "measured_samples": measured_samples,
        "sampling_seconds": sampling_seconds,
        "samples_per_second": measured_samples / sampling_seconds,
        "target_samples_per_checkpoint": target_count,
        "one_checkpoint_sampling_seconds": one_checkpoint_seconds,
        "one_checkpoint_sampling_minutes": one_checkpoint_seconds / 60.0,
        "one_checkpoint_timing_kind": timing_kind,
        "checkpoint_count": checkpoint_count,
        "all_checkpoints_sampling_seconds": all_checkpoints_seconds,
        "all_checkpoints_sampling_hours": all_checkpoints_seconds / 3600.0,
        "all_checkpoints_timing_kind": (
            "measured projection" if timing_kind == "measured" else "throughput projection"
        ),
    }


def _sampling_options(config, device):
    """Return the exact saved sampler settings used by the model."""
    return {
        "n_steps": int(config.sampling.n_steps),
        "method": str(config.sampling.method),
        "start_time": float(config.sampling.start_time),
        "end_time": float(config.sampling.end_time),
        "use_safe_div": bool(config.sampling.use_safe_div),
        "safe_div_denominator_floor": float(
            config.sampling.safe_div_denominator_floor
        ),
        "output": str(config.sampling.output),
        "eps_to_v": str(config.sampling.eps_to_v),
        "heun_final_euler": bool(config.sampling.heun_final_euler),
        "final_pixel_clamp": bool(
            config.sampling.get("final_pixel_clamp", False)
        ),
        "device": device,
    }


def _update_diagnostics(state, samples):
    """Accumulate finite-value and raw-range checks without saving samples."""
    detached = samples.detach()
    finite = torch.isfinite(detached)
    values = detached[finite]
    state["value_count"] += int(detached.numel())
    state["finite_count"] += int(finite.sum().item())
    state["outside_count"] += int(
        (((detached < -1.0) | (detached > 1.0)) & finite).sum().item()
    )
    if values.numel():
        state["finite_square_sum"] += float(values.double().square().sum().item())
        value_min = float(values.min().item())
        value_max = float(values.max().item())
        state["raw_min"] = (
            value_min if state["raw_min"] is None else min(state["raw_min"], value_min)
        )
        state["raw_max"] = (
            value_max if state["raw_max"] is None else max(state["raw_max"], value_max)
        )


def _finalize_diagnostics(state):
    """Convert accumulated diagnostic totals into public scalar values."""
    value_count = state["value_count"]
    finite_count = state["finite_count"]
    return {
        "finite_fraction": finite_count / value_count if value_count else None,
        "raw_min": state["raw_min"],
        "raw_max": state["raw_max"],
        "raw_rms": (
            (state["finite_square_sum"] / finite_count) ** 0.5
            if finite_count
            else None
        ),
        "outside_minus1_1_fraction": (
            state["outside_count"] / value_count if value_count else None
        ),
    }


def _hardware_record(device):
    """Describe the device that produced the timing result."""
    record = {
        "device": str(device),
        "host": platform.node(),
        "pytorch": torch.__version__,
    }
    if device.type == "cuda":
        record.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return record


def _checkpoint_selection(run_dir, checkpoint_step):
    """Select one checkpoint and retain the full saved checkpoint count."""
    checkpoints = list_checkpoints(run_dir)
    available = {step: path for step, path in checkpoints}
    step = max(available) if checkpoint_step is None else int(checkpoint_step)
    if step not in available:
        raise ValueError(f"Checkpoint step {step} does not exist in the saved run.")
    return step, available[step], checkpoints


def benchmark_checkpoint_sampling(
    run_dir,
    *,
    checkpoint_step=None,
    measure_count=10000,
    target_count=10000,
    batch_size=None,
    warmup_batches=1,
    device="auto",
    output_dir=None,
    time_budget_seconds=None,
    preview_count=16,
    retain_samples=False,
    stop_requested=None,
):
    """Benchmark one saved checkpoint and project the run's sampling workload."""
    started_at = _utc_now()
    total_started = time.perf_counter()
    run_dir = Path(run_dir).expanduser().resolve()
    run_record = read_json(run_dir / "run.json")
    if int(run_record.get("artifact_format_version", -1)) != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Saved run uses an unsupported artifact format version.")
    if str(run_record.get("run_kind")) != "image":
        raise ValueError("The sampling benchmark requires one saved image run.")

    measure_count = int(measure_count)
    target_count = int(target_count)
    warmup_batches = int(warmup_batches)
    preview_count = int(preview_count)
    if measure_count < 1 or target_count < 1:
        raise ValueError("measure_count and target_count must be at least 1.")
    if measure_count > target_count:
        raise ValueError("measure_count cannot exceed target_count.")
    if warmup_batches < 0 or preview_count < 0:
        raise ValueError("warmup_batches and preview_count cannot be negative.")
    if time_budget_seconds is not None and float(time_budget_seconds) <= 0.0:
        raise ValueError("time_budget_seconds must be positive when provided.")

    output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Loading saved run: {run_dir}", flush=True)

    setup_started = time.perf_counter()
    config, settings_changed = resolve_evaluation_config(run_dir)
    if settings_changed:
        raise RuntimeError("Saved sampling defaults changed during config resolution.")
    resolved_device = resolve_device(device)
    configure_runtime(config, resolved_device)
    bundle = load_torch(run_dir / "data_bundle.pt")
    validate_data_manifest(run_dir, bundle, run_kind="image", verify_hashes=True)
    saved_noise = load_torch(run_dir / "z_init.pt")
    noise_seed = int(run_record.get("z_init_seed", int(config.training.seed) + 2003))
    noise = build_evaluation_noise(saved_noise, measure_count, noise_seed)
    step, checkpoint_path, checkpoints = _checkpoint_selection(
        run_dir, checkpoint_step
    )
    checkpoint = load_torch(checkpoint_path)
    if int(checkpoint.get("step", -1)) != step:
        raise ValueError("Checkpoint step does not match its filename.")
    setup_seconds = time.perf_counter() - setup_started

    print(
        f"[setup] Building model and loading checkpoint step {step:,}...",
        flush=True,
    )
    model_started = time.perf_counter()
    model = GDFM(config).to(resolved_device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    _synchronize(resolved_device)
    model_load_seconds = time.perf_counter() - model_started

    resolved_batch_size = (
        int(config.evaluation.batch_size)
        if batch_size is None or int(batch_size) == 0
        else int(batch_size)
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    non_blocking = (
        bool(config.runtime.non_blocking_h2d) and resolved_device.type == "cuda"
    )
    sampler_options = _sampling_options(config, resolved_device)
    print(
        "[setup] "
        f"dataset={config.data.dataset} N={len(bundle['train'])} "
        f"shape={tuple(bundle['train'].shape[1:])} device={resolved_device}",
        flush=True,
    )
    print(
        "[setup] "
        f"measure={measure_count:,} target={target_count:,} "
        f"batch={resolved_batch_size} checkpoints={len(checkpoints)}",
        flush=True,
    )
    print(
        "[setup] "
        f"sampler={config.sampling.method} steps={config.sampling.n_steps} "
        f"t=[{config.sampling.start_time},{config.sampling.end_time}]",
        flush=True,
    )
    if output_dir is not None:
        print(f"[setup] Timing artifacts: {output_dir}", flush=True)

    print(f"[warmup] Running {warmup_batches} batch(es)...", flush=True)
    warmup_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(warmup_batches):
            warmup_noise = noise[: min(resolved_batch_size, len(noise))].to(
                resolved_device,
                non_blocking=non_blocking,
            )
            model.sample(warmup_noise, **sampler_options)
            _synchronize(resolved_device)
    warmup_seconds = time.perf_counter() - warmup_started
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    print(f"[warmup] Complete in {warmup_seconds:.2f}s.", flush=True)
    print("[sample] Starting timed sampling...", flush=True)

    rows = []
    previews = []
    retained_samples = []
    diagnostics = {
        "value_count": 0,
        "finite_count": 0,
        "outside_count": 0,
        "finite_square_sum": 0.0,
        "raw_min": None,
        "raw_max": None,
    }
    sampling_started = time.perf_counter()
    completed = 0
    stop_reason = None
    stop_requested = stop_requested or (lambda: False)
    with torch.inference_mode():
        while completed < measure_count:
            if completed and (_STOP_REQUESTED or stop_requested()):
                stop_reason = "signal requested a stop"
                break
            sample_stop = min(completed + resolved_batch_size, measure_count)
            z_batch = noise[completed:sample_stop].to(
                resolved_device,
                non_blocking=non_blocking,
            )
            _synchronize(resolved_device)
            batch_started = time.perf_counter()
            samples = model.sample(z_batch, **sampler_options)
            _synchronize(resolved_device)
            batch_seconds = time.perf_counter() - batch_started
            _update_diagnostics(diagnostics, samples)
            remaining_preview = preview_count - sum(len(item) for item in previews)
            if remaining_preview > 0:
                previews.append(samples[:remaining_preview].detach().cpu())
            if retain_samples:
                retained_samples.append(samples.detach().cpu())
            batch_count = sample_stop - completed
            completed = sample_stop
            cumulative_seconds = sum(item["seconds"] for item in rows) + batch_seconds
            rows.append(
                {
                    "batch_index": len(rows),
                    "sample_start": completed - batch_count,
                    "sample_stop": completed,
                    "sample_count": batch_count,
                    "seconds": batch_seconds,
                    "samples_per_second": batch_count / batch_seconds,
                    "cumulative_samples": completed,
                    "cumulative_seconds": cumulative_seconds,
                }
            )
            batch_number = len(rows)
            if batch_number == 1 or batch_number % 5 == 0 or completed == measure_count:
                throughput = completed / cumulative_seconds
                eta_seconds = (measure_count - completed) / throughput
                print(
                    "[sample] "
                    f"{completed:,}/{measure_count:,} "
                    f"({100.0 * completed / measure_count:.1f}%) | "
                    f"{throughput:.2f} images/s | "
                    f"elapsed {cumulative_seconds / 60.0:.2f}m | "
                    f"ETA {eta_seconds / 60.0:.2f}m",
                    flush=True,
                )
            if output_dir is not None:
                _write_batch_csv(output_dir / "batches.csv", rows)
                write_json(
                    output_dir / "progress.json",
                    {
                        "checkpoint_step": step,
                        "completed_samples": completed,
                        "measure_count": measure_count,
                        "sampling_seconds": cumulative_seconds,
                        "status": "running",
                        "updated_at": _utc_now(),
                    },
                )
            if completed < measure_count and time_budget_seconds is not None:
                elapsed = time.perf_counter() - sampling_started
                next_batch_allowance = max(batch_seconds * 1.25, batch_seconds + 5.0)
                if elapsed + next_batch_allowance >= float(time_budget_seconds):
                    stop_reason = "sampling time budget reached"
                    break

    sampling_seconds = sum(item["seconds"] for item in rows)
    performance = summarize_sampling_runtime(
        completed,
        sampling_seconds,
        target_count,
        len(checkpoints),
    )
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    status = "complete" if completed == measure_count else "partial"
    summary = {
        "schema_version": SAMPLING_BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "scope": "sampling_only",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "run": {
            "run_dir": str(run_dir),
            "dataset": str(config.data.dataset),
            "n_train": int(bundle["train"].shape[0]),
            "sample_shape": list(bundle["train"].shape[1:]),
            "seed": int(config.training.seed),
            "prediction_space": str(config.framework.prediction_space),
            "loss_space": str(config.framework.loss_space),
        },
        "checkpoint": {
            "step": step,
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "saved_checkpoint_count": len(checkpoints),
        },
        "noise": {
            "seed": noise_seed,
            "saved_prefix_rows": min(len(saved_noise), measure_count),
            "measured_rows": measure_count,
            "shape": list(noise.shape),
        },
        "sampler": config_to_dict(config.sampling),
        "benchmark": {
            "batch_size": resolved_batch_size,
            "measure_count": measure_count,
            "target_count": target_count,
            "warmup_batches": warmup_batches,
            "time_budget_seconds": (
                None if time_budget_seconds is None else float(time_budget_seconds)
            ),
        },
        "timing": {
            "artifact_and_config_setup_seconds": setup_seconds,
            "model_and_checkpoint_load_seconds": model_load_seconds,
            "warmup_seconds": warmup_seconds,
            "total_wall_seconds": time.perf_counter() - total_started,
            **performance,
        },
        "diagnostics": _finalize_diagnostics(diagnostics),
        "hardware": _hardware_record(resolved_device),
        "peak_cuda_memory_bytes": peak_cuda_bytes,
        "peak_cuda_memory_gib": (
            peak_cuda_bytes / (1024**3) if peak_cuda_bytes is not None else None
        ),
        "projection_note": (
            "The full-run estimate multiplies sampling time only. It assumes the "
            "same model, sampler, batch size, and device for every saved checkpoint."
        ),
    }
    preview = (
        torch.cat(previews, dim=0)
        if previews
        else torch.empty((0, *noise.shape[1:]), dtype=noise.dtype)
    )
    if output_dir is not None:
        _write_batch_csv(output_dir / "batches.csv", rows)
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "progress.json",
            {
                "checkpoint_step": step,
                "completed_samples": completed,
                "measure_count": measure_count,
                "sampling_seconds": sampling_seconds,
                "status": status,
                "stop_reason": stop_reason,
                "updated_at": _utc_now(),
            },
        )
    print(
        "[result] "
        f"one checkpoint: {performance['one_checkpoint_sampling_minutes']:.2f}m "
        f"({performance['one_checkpoint_timing_kind']})",
        flush=True,
    )
    print(
        "[result] "
        f"all {len(checkpoints)} checkpoints: "
        f"{performance['all_checkpoints_sampling_hours']:.2f} GPU-hours "
        "(sampling only)",
        flush=True,
    )
    if output_dir is not None:
        print(f"[result] Summary: {output_dir / 'summary.json'}", flush=True)
    sample_bank = torch.cat(retained_samples, dim=0) if retained_samples else None
    return {
        "summary": summary,
        "batches": rows,
        "preview": preview,
        "samples": sample_bank,
    }


def build_parser():
    """Build the sampling-only benchmark command parser."""
    parser = argparse.ArgumentParser(
        description="Time sampling for one saved image checkpoint."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--measure-count", type=int, default=10000)
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--time-budget-seconds", type=float)
    parser.add_argument("--preview-count", type=int, default=0)
    return parser


def main(argv=None):
    """Run one sampling benchmark and print its JSON summary."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGUSR1, _request_stop)
    args = build_parser().parse_args(argv)
    result = benchmark_checkpoint_sampling(
        args.run_dir,
        checkpoint_step=args.checkpoint_step,
        measure_count=args.measure_count,
        target_count=args.target_count,
        batch_size=args.batch_size,
        warmup_batches=args.warmup_batches,
        device=args.device,
        output_dir=args.output_dir,
        time_budget_seconds=args.time_budget_seconds,
        preview_count=args.preview_count,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
