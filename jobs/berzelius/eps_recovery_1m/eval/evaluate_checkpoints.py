#!/usr/bin/env python3
"""Evaluate three checkpoints from the six existing CelebA32 eps/eps runs.

Manifest creation uses only the Python standard library. Evaluation imports the
project and numerical dependencies lazily so ``manifest`` and ``--help`` remain
usable on a login node without activating the training environment.
"""
import argparse
import copy
import csv
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


FID_IMAGE_RANGE = "-1_1"
RHO_VALUES = (0.3, 0.5, 0.7)
EXPECTED_TORCH_FIDELITY_VERSION = "0.4.0-beta"
PROFILE_FIELDS = (
    "eps_to_v",
    "use_safe_div",
    "safe_div_denominator_floor",
    "heun_final_euler",
)
EPS_PROFILE_BY_SOURCE = {
    0: "old_full_heun",
    1: "direct_full_heun",
    2: "direct_final_euler",
    3: "exact_full_heun",
    4: "exact_final_euler",
    5: "direct_final_euler",
}
METRIC_FIELDS = [
    "source_index",
    "source_label",
    "source_run",
    "prediction_space",
    "loss_space",
    "training_lr",
    "step",
    "sampling_profile",
    "sampling_signature",
    "sampling_method",
    "sampling_steps",
    "sampling_start_time",
    "sampling_end_time",
    "sampling_output",
    "eps_to_v",
    "use_safe_div",
    "safe_div_denominator_floor",
    "heun_final_euler",
    "final_pixel_clamp",
    "status",
    "fid",
    "fid_10k",
    "frechet_inception_distance",
    "f_mem",
    "baseline_f_mem",
    "delta_f_mem",
    "mean_r",
    "median_r",
    "mean_d1",
    "mean_d2",
    "baseline_mean_r",
    "baseline_median_r",
    "baseline_mean_d1",
    "baseline_mean_d2",
    "f_mem_rho0.3",
    "baseline_f_mem_rho0.3",
    "delta_f_mem_rho0.3",
    "f_mem_rho0.5",
    "baseline_f_mem_rho0.5",
    "delta_f_mem_rho0.5",
    "f_mem_rho0.7",
    "baseline_f_mem_rho0.7",
    "delta_f_mem_rho0.7",
    "sample_count",
    "fid_reference_count",
    "train_count",
    "heldout_count",
    "finite_fraction",
    "raw_min",
    "raw_max",
    "raw_rms",
    "outside_minus1_1_fraction",
    "test_random_t_loss",
    "test_fixed_t_loss_mean",
    "test_loss_t0_5",
    "fid_backend_version",
    "fid_backend_path",
    "fid_image_range",
    "test_loss_t0_75",
    "test_loss_count",
    "fid_feature_extractor",
    "fid_feature_layer",
    "fid_reference_artifact",
    "fid_reference_artifact_sha256",
    "fid_reference_cache",
    "heldout_source",
    "z_init_source",
    "z_init_path",
    "checkpoint_path",
    "sample_runtime_sec",
    "f_mem_runtime_sec",
    "fid_runtime_sec",
    "test_loss_runtime_sec",
    "checkpoint_profile_runtime_sec",
    "error",
]
LOSS_FIELDS = [
    "source_index",
    "source_label",
    "source_run",
    "step",
    "test_random_t_loss",
    "test_fixed_t_loss_mean",
    "test_loss_t0_5",
    "test_loss_t0_75",
    "test_loss_count",
    "test_loss_runtime_sec",
    "checkpoint_path",
]


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}.")


def parse_step_list(value):
    """Parse one comma-separated checkpoint list."""
    if value is None or str(value).strip() == "":
        return None
    steps = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(steps) != len(set(steps)) or any(step < 0 for step in steps):
        raise ValueError("Checkpoint steps must be unique nonnegative integers.")
    return sorted(steps)


def checkpoint_step(path):
    stem = Path(path).stem
    if not stem.startswith("step_"):
        raise ValueError(f"Unexpected checkpoint filename: {path}")
    return int(stem[len("step_") :])


def checkpoint_steps(run_dir):
    return sorted(
        checkpoint_step(path)
        for path in (Path(run_dir) / "checkpoints").glob("step_*.pt")
    )


def select_common_steps(run_dirs, *, max_step, requested_steps=None):
    """Return a frozen checkpoint intersection for all source runs."""
    step_sets = [set(checkpoint_steps(run_dir)) for run_dir in run_dirs]
    if not step_sets or any(not steps for steps in step_sets):
        raise FileNotFoundError("Every source run must contain checkpoints.")
    common = sorted(set.intersection(*step_sets))
    eligible = [step for step in common if step <= int(max_step)]
    if requested_steps is not None:
        missing = sorted(set(requested_steps).difference(eligible))
        if missing:
            raise ValueError(f"Requested checkpoints are not common under the limit: {missing}")
        eligible = list(requested_steps)
    if not eligible:
        raise ValueError("No common checkpoints satisfy the requested selection.")
    return eligible


def load_profiles(path):
    """Load and validate the explicit epsilon sampling profiles."""
    profiles = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            profile = {
                "profile": str(row["profile"]),
                "eps_to_v": str(row["eps_to_v"]),
                "use_safe_div": _parse_bool(row["use_safe_div"]),
                "safe_div_denominator_floor": float(row["safe_div_denominator_floor"]),
                "heun_final_euler": _parse_bool(row["heun_final_euler"]),
                "description": str(row.get("description", "")),
            }
            if profile["eps_to_v"] not in {"direct", "via_x"}:
                raise ValueError(f"Invalid eps_to_v in profile {profile['profile']!r}.")
            profiles.append(profile)
    names = [profile["profile"] for profile in profiles]
    if not profiles or len(names) != len(set(names)):
        raise ValueError("Sampling profiles must have unique nonempty names.")
    return profiles


def _saved_sampling(config):
    sampling = config["sampling"]
    keys = (
        "method",
        "n_steps",
        "start_time",
        "end_time",
        "output",
        "eps_to_v",
        "use_safe_div",
        "safe_div_denominator_floor",
        "heun_final_euler",
        "final_pixel_clamp",
    )
    return {key: sampling[key] for key in keys}


def build_manifest(run_root, variants_path, profiles_path, *, max_step, requested_steps=None):
    """Describe the six epsilon runs and freeze their three checkpoints."""
    run_root = Path(run_root).expanduser().resolve()
    profiles = load_profiles(profiles_path)
    profiles_by_name = {profile["profile"]: profile for profile in profiles}
    variants = []
    with Path(variants_path).open("r", newline="", encoding="utf-8") as handle:
        variants.extend(csv.DictReader(handle, delimiter="\t"))
    if len(variants) != 8:
        raise ValueError(f"Expected eight training variants, found {len(variants)}.")

    sources = []
    run_dirs = []
    for row in variants:
        source_index = int(row["id"])
        label = str(row["label"])
        matches = sorted(run_root.glob(f"{source_index}_{label}_s*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one run for task {source_index} {label}, found {len(matches)}."
            )
        run_dir = matches[0].resolve()
        config = _read_json(run_dir / "config.json")
        prediction = str(config["framework"]["prediction_space"])
        loss = str(config["framework"]["loss_space"])
        if prediction != str(row["prediction"]) or loss != str(row["loss"]):
            raise ValueError(f"Variant metadata disagrees with {run_dir / 'config.json'}.")
        if not math.isclose(float(config["training"]["lr"]), float(row["lr"])):
            raise ValueError(f"Learning rate metadata disagrees for {run_dir.name}.")

        run_record = _read_json(run_dir / "run.json")
        saved = _saved_sampling(config)
        if source_index in EPS_PROFILE_BY_SOURCE:
            if prediction != "eps" or loss != "eps":
                raise ValueError(f"Expected eps/eps for source {source_index}.")
            profile_name = EPS_PROFILE_BY_SOURCE[source_index]
            evaluation_profiles = [profiles_by_name[profile_name]]
            mismatched = [
                field for field in PROFILE_FIELDS
                if saved[field] != evaluation_profiles[0][field]
            ]
            if mismatched:
                raise ValueError(f"Saved sampler disagrees on fields: {mismatched}.")
        else:
            evaluation_profiles = []
        sources.append(
            {
                "source_index": source_index,
                "source_label": label,
                "source_run": run_dir.name,
                "run_dir": str(run_dir),
                "prediction_space": prediction,
                "loss_space": loss,
                "training_lr": float(config["training"]["lr"]),
                "training_seed": int(config["training"]["seed"]),
                "data_bundle_sha256": run_record.get("data_bundle_sha256"),
                "saved_sampling": saved,
                "evaluation_profiles": evaluation_profiles,
                "available_checkpoint_count": len(checkpoint_steps(run_dir)),
                "available_checkpoint_max": max(checkpoint_steps(run_dir)),
            }
        )
        if evaluation_profiles:
            run_dirs.append(run_dir)

    source_indices = [source["source_index"] for source in sources]
    if source_indices != list(range(8)):
        raise ValueError(f"Expected source indices 0 through 7, got {source_indices}.")
    data_hashes = {source["data_bundle_sha256"] for source in sources}
    if None in data_hashes or len(data_hashes) != 1:
        raise ValueError("All source runs must use the same recorded data bundle.")
    selected = select_common_steps(
        run_dirs,
        max_step=max_step,
        requested_steps=requested_steps,
    )
    tasks = []
    for source in sources:
        if not source["evaluation_profiles"]:
            continue
        task = dict(source)
        task["task_index"] = len(tasks)
        tasks.append(task)
    if len(tasks) != 6:
        raise ValueError(f"Expected six epsilon evaluation tasks, found {len(tasks)}.")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "max_step": int(max_step),
        "checkpoint_steps": selected,
        "checkpoint_count": len(selected),
        "sources": sources,
        "tasks": tasks,
    }


@contextmanager
def file_lock(path, *, timeout_seconds=1800.0, stale_seconds=7200.0):
    """Coordinate shared cache creation across Slurm array tasks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    acquired = False
    while not acquired:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
            os.close(descriptor)
            acquired = True
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_seconds:
                path.unlink(missing_ok=True)
                continue
            if time.monotonic() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for cache lock: {path}")
            time.sleep(2.0)
    try:
        yield
    finally:
        if acquired:
            path.unlink(missing_ok=True)


def _atomic_torch_save(payload, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_eval_noise(config, train_tensor, count):
    import torch

    generator = torch.Generator().manual_seed(int(config.training.seed) + 2003)
    return torch.randn(
        int(count),
        *train_tensor.shape[1:],
        generator=generator,
        dtype=train_tensor.dtype,
    )


def load_or_build_eval_noise(cache_root, config, train_tensor, saved_z, count):
    """Persist the exact replay noise and verify the saved training prefix."""
    import torch

    seed = int(config.training.seed) + 2003
    shape = "x".join(str(value) for value in train_tensor.shape[1:])
    cache_path = Path(cache_root) / "noise" / f"seed{seed}_n{int(count)}_{shape}.pt"
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    if not cache_path.exists():
        with file_lock(lock_path):
            if not cache_path.exists():
                generator = torch.Generator().manual_seed(seed)
                replayed_saved = torch.randn(
                    int(saved_z.shape[0]),
                    *train_tensor.shape[1:],
                    generator=generator,
                    dtype=train_tensor.dtype,
                )
                _verify_prefix("saved evaluation noise", replayed_saved, saved_z)
                if int(count) <= int(saved_z.shape[0]):
                    tensor = saved_z[: int(count)].clone().contiguous()
                else:
                    tail = torch.randn(
                        int(count) - int(saved_z.shape[0]),
                        *train_tensor.shape[1:],
                        generator=generator,
                        dtype=train_tensor.dtype,
                    )
                    tensor = torch.cat((saved_z, tail)).contiguous()
                _atomic_torch_save(
                    {
                        "tensor": tensor,
                        "seed": seed,
                        "count": int(count),
                        "saved_prefix_count": int(min(count, saved_z.shape[0])),
                        "shape": list(tensor.shape),
                    },
                    cache_path,
                )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    tensor = payload["tensor"].contiguous()
    if int(tensor.shape[0]) != int(count):
        raise RuntimeError(
            f"Replay-noise cache has {len(tensor)} samples, expected {count}."
        )
    _verify_prefix("evaluation noise", tensor, saved_z)
    return tensor, cache_path


def _verify_prefix(name, candidate, saved):
    import torch

    count = min(int(candidate.shape[0]), int(saved.shape[0]))
    if candidate.shape[1:] != saved.shape[1:] or not torch.equal(
        candidate[:count], saved[:count]
    ):
        raise RuntimeError(f"Rebuilt {name} does not match the saved prefix.")


def load_saved_heldout(saved_bundle, count):
    """Select held-out images only from the persisted training artifact."""
    reference = saved_bundle["reference"].contiguous()
    if int(count) > int(reference.shape[0]):
        raise ValueError(
            f"Requested {count} held-out images, but the saved artifact contains "
            f"{int(reference.shape[0])}."
        )
    return reference[: int(count)]


def load_fid_reference_artifact(path, count, saved_bundle):
    """Load the frozen extended reference and verify its saved 1,024 prefix."""
    import torch
    from mem_param_src.runners.artifacts import file_sha256, tensor_sha256

    path = Path(path).expanduser().resolve()
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen FID reference artifact: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = _read_json(manifest_path)
    if int(payload.get("artifact_format_version", -1)) != 1:
        raise RuntimeError("Unsupported FID reference artifact format.")
    artifact_sha256 = file_sha256(path)
    if artifact_sha256 != str(manifest.get("artifact_sha256")):
        raise RuntimeError("FID reference artifact hash does not match its manifest.")
    reference = payload["reference"].contiguous()
    if int(reference.shape[0]) != int(count):
        raise RuntimeError(
            f"Frozen FID reference has {len(reference)} images, expected {count}."
        )
    if tensor_sha256(reference) != str(manifest.get("reference_tensor_sha256")):
        raise RuntimeError("FID reference tensor hash does not match its manifest.")

    saved_reference = saved_bundle["reference"].contiguous()
    saved_count = int(saved_reference.shape[0])
    if int(payload.get("saved_prefix_count", -1)) != saved_count:
        raise RuntimeError("FID reference saved-prefix count does not match the run.")
    _verify_prefix("FID reference", reference, saved_reference)
    for suffix in ("labels", "indices", "sample_ids"):
        key = f"reference_{suffix}"
        if key not in saved_bundle:
            continue
        candidate = payload[key]
        saved = saved_bundle[key]
        if torch.is_tensor(candidate) and torch.is_tensor(saved):
            _verify_prefix(key, candidate, saved)
        elif list(candidate[:saved_count]) != list(saved):
            raise RuntimeError(f"FID reference disagrees with the run on {key}.")
    for suffix in (
        "sample_id_kind",
        "source_size",
        "source_catalog_sha256",
        "source_id",
        "split",
    ):
        key = f"reference_{suffix}"
        if payload.get(key) != saved_bundle.get(key):
            raise RuntimeError(f"FID reference disagrees with the run on {key}.")
    return reference, manifest, path


def covariance_stats(features):
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected two-dimensional features, got {values.shape}.")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False) if len(values) > 1 else np.zeros(
        (values.shape[1], values.shape[1]), dtype=np.float64
    )
    return mean, covariance


def fid_from_stats(sample_mean, sample_covariance, reference_mean, reference_covariance):
    import numpy as np
    from scipy import linalg

    sample_mean = np.asarray(sample_mean, dtype=np.float64)
    sample_covariance = np.asarray(sample_covariance, dtype=np.float64)
    reference_mean = np.asarray(reference_mean, dtype=np.float64)
    reference_covariance = np.asarray(reference_covariance, dtype=np.float64)
    difference = sample_mean - reference_mean
    covariance_mean, _ = linalg.sqrtm(
        sample_covariance @ reference_covariance,
        disp=False,
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sample_covariance.shape[0], dtype=np.float64) * 1.0e-6
        covariance_mean = linalg.sqrtm(
            (sample_covariance + offset) @ (reference_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = difference.dot(difference) + np.trace(
        sample_covariance + reference_covariance - 2.0 * covariance_mean
    )
    return float(max(value, 0.0))


def encode_fid_features(images, encoder, batch_size):
    import torch
    from mem_param_src.metrics.image.preprocessing import normalize_images

    normalized = normalize_images(images, image_range=FID_IMAGE_RANGE)
    features = []
    for start in range(0, len(normalized), int(batch_size)):
        features.append(encoder.encode(normalized[start : start + int(batch_size)]))
    return torch.cat(features).numpy()


def load_or_build_fid_reference(
    cache_root,
    heldout,
    *,
    reference_count,
    encoder,
    feature_extractor,
    batch_size,
    seed,
    reference_artifact_sha256,
):
    import torch

    layer = str(encoder.feature_layer)
    safe_extractor = str(feature_extractor).replace("/", "_").replace(":", "_")
    safe_layer = layer.replace("/", "_").replace(":", "_")
    reference_id = str(reference_artifact_sha256)[:12]
    cache_path = (
        Path(cache_root)
        / "fid"
        / f"valid_seed{int(seed)}_n{int(reference_count)}_{reference_id}_{safe_extractor}_{safe_layer}.pt"
    )
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    if not cache_path.exists():
        with file_lock(lock_path):
            if not cache_path.exists():
                features = encode_fid_features(
                    heldout[: int(reference_count)], encoder, batch_size
                )
                mean, covariance = covariance_stats(features)
                _atomic_torch_save(
                    {
                        "mean": mean,
                        "covariance": covariance,
                        "count": int(reference_count),
                        "feature_extractor": str(feature_extractor),
                        "feature_layer": layer,
                        "image_range": FID_IMAGE_RANGE,
                        "reference_artifact_sha256": str(
                            reference_artifact_sha256
                        ),
                    },
                    cache_path,
                )
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if int(payload["count"]) != int(reference_count):
        raise RuntimeError("FID reference cache count does not match the request.")
    if str(payload.get("reference_artifact_sha256")) != str(
        reference_artifact_sha256
    ):
        raise RuntimeError("FID reference cache points to another reference artifact.")
    return payload, cache_path


def raw_sample_stats(samples):
    import torch

    finite = torch.isfinite(samples)
    finite_fraction = float(finite.float().mean().item())
    finite_values = samples[finite]
    if finite_values.numel() == 0:
        minimum = maximum = rms = math.nan
    else:
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        rms = float(finite_values.square().mean().sqrt().item())
    outside = ((samples < -1.0) | (samples > 1.0)) & finite
    return {
        "finite_fraction": finite_fraction,
        "raw_min": minimum,
        "raw_max": maximum,
        "raw_rms": rms,
        "outside_minus1_1_fraction": float(outside.float().mean().item()),
    }


def _memorization_baseline(heldout, train_tensor, *, count, device, chunk_size, rho):
    from mem_param_src.metrics.nearest_neighbors import compute_nn_stats, summarize_nn_stats

    stats = compute_nn_stats(
        heldout[: int(count)].to(device),
        train_tensor.to(device),
        chunk_size=int(chunk_size),
    )
    return stats, summarize_nn_stats(
        stats,
        rho=float(rho),
        n_samples=int(count),
        n_reference=int(train_tensor.shape[0]),
    )


def memorization_record(samples, train_tensor, baseline_stats, baseline, *, device, chunk_size):
    from mem_param_src.metrics.nearest_neighbors import (
        compute_f_mem,
        compute_nn_stats,
        summarize_nn_stats,
    )

    generated_stats = compute_nn_stats(
        samples.to(device),
        train_tensor.to(device),
        chunk_size=int(chunk_size),
    )
    generated = summarize_nn_stats(
        generated_stats,
        rho=float(baseline["rho"]),
        n_samples=int(samples.shape[0]),
        n_reference=int(train_tensor.shape[0]),
    )
    record = {
        "f_mem": generated["f_mem"],
        "baseline_f_mem": baseline["f_mem"],
        "delta_f_mem": generated["f_mem"] - baseline["f_mem"],
        "mean_r": generated["mean_r"],
        "median_r": generated["median_r"],
        "mean_d1": generated["mean_d1"],
        "mean_d2": generated["mean_d2"],
        "baseline_mean_r": baseline["mean_r"],
        "baseline_median_r": baseline["median_r"],
        "baseline_mean_d1": baseline["mean_d1"],
        "baseline_mean_d2": baseline["mean_d2"],
    }
    for rho in RHO_VALUES:
        generated_f_mem = compute_f_mem(generated_stats["r"], rho=rho)
        baseline_f_mem = compute_f_mem(baseline_stats["r"], rho=rho)
        record[f"f_mem_rho{rho}"] = generated_f_mem
        record[f"baseline_f_mem_rho{rho}"] = baseline_f_mem
        record[f"delta_f_mem_rho{rho}"] = generated_f_mem - baseline_f_mem
    return record


def _weighted_mean(items):
    count = sum(item_count for _, item_count in items)
    return float(sum(value * item_count for value, item_count in items) / count)


def evaluate_test_loss(model, heldout, config, *, step, device, batch_size):
    import torch

    fixed_times = [float(value) for value in config.evaluation.loss_eval_times]
    if not fixed_times:
        raise ValueError("evaluation.loss_eval_times must contain at least one time.")
    random_losses = []
    fixed_losses = {value: [] for value in fixed_times}
    devices = []
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [cuda_index]
    started = time.perf_counter()
    with torch.no_grad(), torch.random.fork_rng(devices=devices, enabled=True):
        for batch_index, start in enumerate(range(0, len(heldout), int(batch_size))):
            batch = heldout[start : start + int(batch_size)].to(device)
            count = int(batch.shape[0])
            torch.manual_seed(int(config.training.seed) + int(step) + 1009 + batch_index)
            random_losses.append((float(model.compute_loss(batch).item()), count))
            for fixed_time in fixed_times:
                torch.manual_seed(
                    int(config.training.seed)
                    + int(step)
                    + int(round(fixed_time * 10000.0))
                    + 2027
                    + batch_index
                )
                eps = torch.randn_like(batch)
                t = torch.full((count,), fixed_time, device=device, dtype=torch.float32)
                fixed_losses[fixed_time].append(
                    (float(model.compute_loss(batch, eps=eps, t=t).item()), count)
                )
    fixed_means = {value: _weighted_mean(items) for value, items in fixed_losses.items()}
    result = {
        "test_random_t_loss": _weighted_mean(random_losses),
        "test_fixed_t_loss_mean": float(sum(fixed_means.values()) / len(fixed_means)),
        "test_loss_count": int(len(heldout)),
        "test_loss_runtime_sec": time.perf_counter() - started,
    }
    for value, mean in fixed_means.items():
        result[f"test_loss_t{str(value).replace('.', '_')}"] = mean
    return result


def apply_sampling_profile(config, profile):
    resolved = copy.deepcopy(config)
    if profile["profile"] != "saved":
        for field in PROFILE_FIELDS:
            resolved.sampling[field] = profile[field]
    return resolved


def sampling_signature(config):
    fields = {
        key: config.sampling[key]
        for key in (
            "method",
            "n_steps",
            "start_time",
            "end_time",
            "output",
            "eps_to_v",
            "use_safe_div",
            "safe_div_denominator_floor",
            "heun_final_euler",
            "final_pixel_clamp",
        )
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def sample_in_batches(model, z_eval, config, *, device, batch_size):
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z_eval), int(batch_size)):
            z_batch = z_eval[start : start + int(batch_size)].to(device)
            outputs.append(
                model.sample(
                    z_batch,
                    n_steps=int(config.sampling.n_steps),
                    method=str(config.sampling.method),
                    start_time=float(config.sampling.start_time),
                    end_time=float(config.sampling.end_time),
                    use_safe_div=bool(config.sampling.use_safe_div),
                    safe_div_denominator_floor=float(
                        config.sampling.safe_div_denominator_floor
                    ),
                    output=str(config.sampling.output),
                    eps_to_v=str(config.sampling.eps_to_v),
                    heun_final_euler=bool(config.sampling.heun_final_euler),
                    final_pixel_clamp=bool(config.sampling.final_pixel_clamp),
                    device=device,
                ).detach().cpu().float()
            )
    return torch.cat(outputs)


def _append_row(csv_path, jsonl_path, row, fieldnames):
    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _completed_metric_keys(path):
    return {
        (
            int(row["source_index"]),
            str(row["sampling_profile"]),
            int(row["step"]),
            int(row["sample_count"]),
            int(row["fid_reference_count"]),
            str(row["sampling_signature"]),
        )
        for row in _read_jsonl(path)
    }


def _existing_losses(path):
    return {int(row["step"]): row for row in _read_jsonl(path)}


def run_task(args):
    """Evaluate one source trajectory selected by its Slurm array index."""
    import torch
    import torch_fidelity
    from ml_collections import ConfigDict

    from mem_param_src.metrics.image.features import TorchFidelityFeatureEncoder
    from mem_param_src.model import GDFM
    from mem_param_src.runners import configure_runtime, resolve_device
    from mem_param_src.runners.artifacts import load_torch
    from mem_param_src.runners.image_plotting import save_training_comparison_grid

    manifest = _read_json(args.manifest)
    by_index = {int(task["task_index"]): task for task in manifest["tasks"]}
    if int(args.task_index) not in by_index:
        raise ValueError(f"Task index {args.task_index} is not in the manifest.")
    task = by_index[int(args.task_index)]
    run_dir = Path(task["run_dir"])
    output_root = Path(args.output_root).expanduser().resolve()
    profile_name = task["evaluation_profiles"][0]["profile"]
    task_dir = (
        output_root
        / "tasks"
        / f"{int(task['task_index']):02d}_{task['source_label']}_{profile_name}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "cache"

    config = ConfigDict(_read_json(run_dir / "config.json"))
    device = resolve_device(args.device)
    configure_runtime(config, device)
    saved_bundle = load_torch(run_dir / "data_bundle.pt")
    train_tensor = saved_bundle["train"].contiguous()
    saved_z = load_torch(run_dir / "z_init.pt")
    heldout_count = int(args.loss_count)
    heldout = load_saved_heldout(saved_bundle, heldout_count)
    heldout_source = run_dir / "data_bundle.pt"
    fid_reference_tensor, fid_reference_manifest, fid_reference_artifact = (
        load_fid_reference_artifact(
            args.fid_reference_artifact,
            int(args.fid_reference_count),
            saved_bundle,
        )
    )
    z_eval, z_eval_path = load_or_build_eval_noise(
        cache_root,
        config,
        train_tensor,
        saved_z,
        int(args.sample_count),
    )

    baseline_stats, baseline = _memorization_baseline(
        heldout,
        train_tensor,
        count=int(heldout_count),
        device=device,
        chunk_size=int(args.nn_chunk_size),
        rho=float(config.evaluation.f_mem_threshold),
    )
    fid_backend_version = str(torch_fidelity.__version__)
    if fid_backend_version != EXPECTED_TORCH_FIDELITY_VERSION:
        raise RuntimeError(
            f"Expected torch-fidelity {EXPECTED_TORCH_FIDELITY_VERSION}, "
            f"found {fid_backend_version}."
        )
    fid_backend_path = str(Path(torch_fidelity.__file__).resolve())
    feature_extractor = str(args.fid_feature_extractor)
    encoder = TorchFidelityFeatureEncoder(
        f"torch_fidelity:{feature_extractor}", device=device
    )
    fid_reference, fid_cache = load_or_build_fid_reference(
        cache_root,
        fid_reference_tensor,
        reference_count=int(args.fid_reference_count),
        encoder=encoder,
        feature_extractor=feature_extractor,
        batch_size=int(args.fid_batch_size),
        seed=int(config.training.seed),
        reference_artifact_sha256=fid_reference_manifest["artifact_sha256"],
    )

    profiles = task["evaluation_profiles"]
    profile_configs = {
        profile["profile"]: apply_sampling_profile(config, profile)
        for profile in profiles
    }
    metric_jsonl = task_dir / "checkpoint_metrics.jsonl"
    metric_csv = task_dir / "checkpoint_metrics.csv"
    loss_jsonl = task_dir / "loss_metrics.jsonl"
    loss_csv = task_dir / "loss_metrics.csv"
    completed = _completed_metric_keys(metric_jsonl)
    losses = _existing_losses(loss_jsonl)

    settings = {
        "manifest": str(Path(args.manifest).resolve()),
        "loss_batch_size": int(args.loss_batch_size),
        "nn_chunk_size": int(args.nn_chunk_size),
        "fid_backend_version": fid_backend_version,
        "fid_backend_path": fid_backend_path,
        "fid_image_range": FID_IMAGE_RANGE,
        "task": task,
        "checkpoint_steps": manifest["checkpoint_steps"],
        "sample_count": int(args.sample_count),
        "fid_reference_count": int(args.fid_reference_count),
        "loss_count": int(args.loss_count),
        "sample_batch_size": int(args.sample_batch_size),
        "fid_batch_size": int(args.fid_batch_size),
        "fid_feature_extractor": feature_extractor,
        "fid_reference_artifact": str(fid_reference_artifact),
        "fid_reference_artifact_sha256": fid_reference_manifest[
            "artifact_sha256"
        ],
        "heldout_source": str(heldout_source),
        "z_init_path": str(z_eval_path),
        "fid_reference_cache": str(fid_cache),
        "device": str(device),
    }
    settings_path = task_dir / "settings.json"
    if settings_path.exists() and _read_json(settings_path) != settings:
        raise RuntimeError(f"Existing task settings differ: {settings_path}")
    _atomic_write_json(settings_path, settings)

    model = GDFM(config).to(device)
    for step in manifest["checkpoint_steps"]:
        step = int(step)
        checkpoint_path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
        pending_profiles = []
        for profile in profiles:
            profile_name = profile["profile"]
            signature = sampling_signature(profile_configs[profile_name])
            key = (
                int(task["source_index"]),
                profile_name,
                step,
                int(args.sample_count),
                int(args.fid_reference_count),
                signature,
            )
            if key not in completed:
                pending_profiles.append(profile)
        if not pending_profiles and step in losses:
            continue

        checkpoint = load_torch(checkpoint_path)
        if int(checkpoint.get("step", -1)) != step:
            raise RuntimeError(f"Checkpoint metadata disagrees with {checkpoint_path}.")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()

        if step not in losses:
            loss_record = evaluate_test_loss(
                model,
                heldout[: int(args.loss_count)],
                config,
                step=step,
                device=device,
                batch_size=int(args.loss_batch_size),
            )
            loss_record.update(
                {
                    "source_index": int(task["source_index"]),
                    "source_label": task["source_label"],
                    "source_run": task["source_run"],
                    "step": step,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            _append_row(loss_csv, loss_jsonl, loss_record, LOSS_FIELDS)
            losses[step] = loss_record

        for profile in pending_profiles:
            profile_name = profile["profile"]
            resolved = profile_configs[profile_name]
            signature = sampling_signature(resolved)
            started = time.perf_counter()
            sample_started = time.perf_counter()
            samples = sample_in_batches(
                model,
                z_eval,
                resolved,
                device=device,
                batch_size=int(args.sample_batch_size),
            )
            sample_runtime = time.perf_counter() - sample_started
            row = {
                "source_index": int(task["source_index"]),
                "source_label": task["source_label"],
                "source_run": task["source_run"],
                "prediction_space": task["prediction_space"],
                "loss_space": task["loss_space"],
                "training_lr": float(task["training_lr"]),
                "step": step,
                "sampling_profile": profile_name,
                "sampling_signature": signature,
                "sampling_method": str(resolved.sampling.method),
                "sampling_steps": int(resolved.sampling.n_steps),
                "sampling_start_time": float(resolved.sampling.start_time),
                "sampling_end_time": float(resolved.sampling.end_time),
                "sampling_output": str(resolved.sampling.output),
                "eps_to_v": str(resolved.sampling.eps_to_v),
                "use_safe_div": bool(resolved.sampling.use_safe_div),
                "safe_div_denominator_floor": float(
                    resolved.sampling.safe_div_denominator_floor
                ),
                "heun_final_euler": bool(resolved.sampling.heun_final_euler),
                "final_pixel_clamp": bool(resolved.sampling.final_pixel_clamp),
                "fid_backend_version": fid_backend_version,
                "fid_backend_path": fid_backend_path,
                "fid_image_range": FID_IMAGE_RANGE,
                "sample_count": int(args.sample_count),
                "fid_reference_count": int(args.fid_reference_count),
                "train_count": int(train_tensor.shape[0]),
                "heldout_count": int(heldout_count),
                "fid_feature_extractor": feature_extractor,
                "fid_feature_layer": str(encoder.feature_layer),
                "fid_reference_artifact": str(fid_reference_artifact),
                "fid_reference_artifact_sha256": fid_reference_manifest[
                    "artifact_sha256"
                ],
                "fid_reference_cache": str(fid_cache),
                "heldout_source": str(heldout_source),
                "z_init_source": f"deterministic_seed_{int(config.training.seed) + 2003}",
                "z_init_path": str(z_eval_path),
                "checkpoint_path": str(checkpoint_path),
                "sample_runtime_sec": sample_runtime,
                **raw_sample_stats(samples),
                **losses[step],
                "error": "",
            }
            if row["finite_fraction"] == 1.0:
                f_mem_started = time.perf_counter()
                row.update(
                    memorization_record(
                        samples,
                        train_tensor,
                        baseline_stats,
                        baseline,
                        device=device,
                        chunk_size=int(args.nn_chunk_size),
                    )
                )
                row["f_mem_runtime_sec"] = time.perf_counter() - f_mem_started
                fid_started = time.perf_counter()
                sample_features = encode_fid_features(
                    samples,
                    encoder,
                    int(args.fid_batch_size),
                )
                sample_mean, sample_covariance = covariance_stats(sample_features)
                fid_value = fid_from_stats(
                    sample_mean,
                    sample_covariance,
                    fid_reference["mean"],
                    fid_reference["covariance"],
                )
                row["fid"] = fid_value
                row["fid_10k"] = (
                    fid_value
                    if int(args.sample_count) == 10_000
                    and int(args.fid_reference_count) == 10_000
                    else ""
                )
                row["frechet_inception_distance"] = fid_value
                row["fid_runtime_sec"] = time.perf_counter() - fid_started
                row["status"] = "ok"
                save_training_comparison_grid(
                    train_tensor[:32],
                    samples[:64],
                    task_dir
                    / "panels"
                    / profile_name
                    / f"step_{step:06d}.png",
                    step,
                )
            else:
                row["status"] = "nonfinite"
                row["error"] = "Generated samples contain non-finite values."
            row["checkpoint_profile_runtime_sec"] = time.perf_counter() - started
            _append_row(metric_csv, metric_jsonl, row, METRIC_FIELDS)
            completed.add(
                (
                    int(task["source_index"]),
                    profile_name,
                    step,
                    int(args.sample_count),
                    int(args.fid_reference_count),
                    signature,
                )
            )
            print(
                f"task={task['task_index']} source={task['source_index']} "
                f"step={step} profile={profile_name} "
                f"status={row['status']} fid={row.get('fid')} "
                f"f_mem={row.get('f_mem')}",
                flush=True,
            )
            del samples
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _atomic_write_json(
        task_dir / "summary.json",
        {
            "state": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "task_index": int(task["task_index"]),
            "source_index": int(task["source_index"]),
            "source_run": task["source_run"],
            "metric_rows": len(_read_jsonl(metric_jsonl)),
            "loss_rows": len(_read_jsonl(loss_jsonl)),
        },
    )
    return task_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    manifest = commands.add_parser("manifest", help="Freeze source runs and checkpoints.")
    manifest.add_argument("--run-root", required=True)
    manifest.add_argument("--variants", required=True)
    manifest.add_argument("--profiles", required=True)
    manifest.add_argument("--max-step", type=int, default=700_000)
    manifest.add_argument("--checkpoint-steps", default="100000,390000,695000")
    manifest.add_argument("--output")

    run = commands.add_parser("run", help="Evaluate one manifest task.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--sample-count", type=int, default=10_000)
    run.add_argument("--fid-reference-count", type=int, default=10_000)
    run.add_argument("--fid-reference-artifact", required=True)
    run.add_argument("--loss-count", type=int, default=1_024)
    run.add_argument("--sample-batch-size", type=int, default=256)
    run.add_argument("--fid-batch-size", type=int, default=128)
    run.add_argument("--loss-batch-size", type=int, default=256)
    run.add_argument("--nn-chunk-size", type=int, default=256)
    run.add_argument("--fid-feature-extractor", default="inception-v3-compat")
    run.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().error("one command is required")
    if args.command == "manifest":
        payload = build_manifest(
            args.run_root,
            args.variants,
            args.profiles,
            max_step=args.max_step,
            requested_steps=parse_step_list(args.checkpoint_steps),
        )
        if args.output:
            _atomic_write_json(args.output, payload)
        print(
            f"tasks={len(payload['tasks'])} checkpoints={payload['checkpoint_count']} "
            f"first={payload['checkpoint_steps'][0]} last={payload['checkpoint_steps'][-1]}",
            flush=True,
        )
        return 0
    output = run_task(args)
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
