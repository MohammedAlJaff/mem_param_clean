"""Interactive sampling, FID, and memorization benchmark for one checkpoint."""

import argparse
import time
from pathlib import Path

import torch

from mem_param_src.config import config_to_dict
from mem_param_src.metrics.image import (
    REFERENCE_QUANTILES,
    SSCDFeatureEncoder,
    TorchFidelityFeatureEncoder,
    compute_f_mem_by_rho,
    compute_f_mem_reference_curve,
    compute_feature_statistics,
    compute_frechet_distance,
    compute_nn_stats,
    encode_image_features,
)
from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    write_json,
)
from mem_param_src.runners.sampling_benchmark import (
    benchmark_checkpoint_sampling,
)

METRICS_BENCHMARK_SCHEMA_VERSION = 2
PIXEL_RHOS = (0.3, 0.5, 0.7)
CALIBRATION_COUNT = 1024
EXPECTED_REFERENCE_COUNT = 10000


def _stage(label, operation):
    """Run one visible stage and return its result and duration."""
    print(f"[{label}] Starting...", flush=True)
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    print(f"[{label}] Complete in {elapsed / 60.0:.2f}m.", flush=True)
    return result, elapsed


def _per_image_tensor_hashes(tensor):
    """Hash every image independently in saved tensor order."""
    return [tensor_sha256(image) for image in tensor]


def _validate_reference_payload(reference, payload, manifest, expected_count=10000):
    """Verify the clean test-bank membership and all aligned identities."""
    preprocessing = manifest.get("preprocessing", {})
    try:
        image_channels = int(preprocessing["image_channels"])
        image_size = int(preprocessing["image_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "CelebA reference manifest has no valid image shape."
        ) from error
    expected_shape = (int(expected_count), image_channels, image_size, image_size)
    if tuple(reference.shape) != expected_shape:
        raise ValueError(
            f"CelebA test reference must have shape {expected_shape}, "
            f"got {tuple(reference.shape)}."
        )
    if payload.get("reference_split") != "test":
        raise ValueError("CelebA reference artifact must use reference_split='test'.")
    if manifest.get("reference_split") != "test":
        raise ValueError("CelebA reference manifest must use reference_split='test'.")
    if payload.get("reference_source_id") != "torchvision_celeba:test":
        raise ValueError("CelebA reference artifact has an unexpected test source.")

    fields = {
        "filenames": list(payload.get("reference_filenames", ())),
        "JPEG hashes": list(payload.get("reference_jpeg_sha256s", ())),
        "transformed tensor hashes": list(payload.get("reference_tensor_sha256s", ())),
    }
    for label, values in fields.items():
        if len(values) != int(expected_count):
            raise ValueError(
                f"CelebA test reference must contain {expected_count} {label}."
            )
        if len(set(values)) != int(expected_count):
            raise ValueError(f"CelebA test reference contains duplicate {label}.")

    indices = payload.get("reference_indices")
    labels = payload.get("reference_labels")
    if not torch.is_tensor(indices) or indices.numel() != int(expected_count):
        raise ValueError("CelebA test reference indices are missing or misaligned.")
    if not torch.is_tensor(labels) or labels.numel() != int(expected_count):
        raise ValueError(
            "CelebA test reference identity labels are missing or misaligned."
        )
    if len({int(value) for value in indices.flatten().tolist()}) != int(expected_count):
        raise ValueError(
            "CelebA test reference contains duplicate split-local indices."
        )

    actual_hashes = _per_image_tensor_hashes(reference)
    if fields["transformed tensor hashes"] != actual_hashes:
        raise ValueError("CelebA test reference per-image tensor hashes do not match.")
    if payload.get("reference_tensor_sha256") != tensor_sha256(reference):
        raise ValueError("CelebA test reference payload tensor hash does not match.")
    if payload.get("training_union_manifest_sha256") != manifest.get(
        "training_union_manifest_sha256"
    ):
        raise ValueError("CelebA test reference training-union hashes differ.")
    if payload.get("partition_file_sha256") != manifest.get("partition_file_sha256"):
        raise ValueError("CelebA test reference partition-file hashes differ.")
    return set(actual_hashes)


def _load_reference(path):
    """Load and verify the frozen 10,000-image clean CelebA test bank."""
    path = Path(path).expanduser().resolve()
    manifest_path = path.with_suffix(".json")
    payload = load_torch(path)
    manifest = read_json(manifest_path)
    if file_sha256(path) != manifest["artifact_sha256"]:
        raise ValueError("Frozen test artifact SHA-256 does not match.")
    reference = payload.get("reference")
    if not torch.is_tensor(reference):
        raise ValueError("Frozen test artifact has no reference tensor.")
    reference = reference.contiguous()
    if len(reference) != EXPECTED_REFERENCE_COUNT:
        raise ValueError("FID reference must contain exactly 10,000 images.")
    if tensor_sha256(reference) != manifest["reference_tensor_sha256"]:
        raise ValueError("Frozen test tensor SHA-256 does not match.")
    _validate_reference_payload(
        reference,
        payload,
        manifest,
        expected_count=EXPECTED_REFERENCE_COUNT,
    )
    return reference, payload, manifest, path


def _validate_reference_compatibility(
    bundle, data_manifest, reference, payload, manifest
):
    """Verify preprocessing and zero tensor overlap with this saved run."""
    del reference
    if (
        payload.get("reference_split") != "test"
        or manifest.get("reference_split") != "test"
    ):
        raise ValueError("CelebA reference must use the official test split.")

    preprocessing = data_manifest["preprocessing"]
    expected = manifest["preprocessing"]
    checks = {
        "image_channels": int(preprocessing["image_channels"]),
        "image_size": int(preprocessing["image_size"]),
        "resize_size": int(preprocessing["resize_size"]),
        "training_augmentation": bool(preprocessing["training_augmentation"]),
        "value_range": list(preprocessing["value_range"]),
    }
    for key, value in checks.items():
        if expected[key] != value:
            raise ValueError(f"Frozen test preprocessing differs for {key}.")

    training = bundle.get("train")
    if not torch.is_tensor(training):
        raise ValueError("Saved run has no training tensor.")
    reference_hashes = set(payload.get("reference_tensor_sha256s", ()))
    training_hashes = set(_per_image_tensor_hashes(training.contiguous()))
    overlap = reference_hashes.intersection(training_hashes)
    if overlap:
        raise ValueError(
            "CelebA test reference is contaminated by the evaluated run's "
            f"training tensor ({len(overlap)} collision(s))."
        )


def _load_assets(path):
    """Load and verify the pinned SSCD and Inception assets."""
    path = Path(path).expanduser().resolve()
    manifest = read_json(path)
    if int(manifest.get("schema_version", -1)) != 3:
        raise ValueError("Encoder asset manifest must use schema version 3.")
    root = Path(manifest["root"]).expanduser().resolve()
    assets = {}
    for key in ("sscd", "fid_inception_v3"):
        record = manifest["encoders"][key]
        asset = root / record["path"]
        if not asset.is_file() or file_sha256(asset) != record["sha256"]:
            raise ValueError(f"Pinned encoder asset failed verification: {key}.")
        assets[key] = asset
    return root, manifest, assets, path


def _encode_with_progress(images, encoder, batch_size, device, label):
    """Encode images in visible batches and return one CPU feature matrix."""
    blocks = []
    total = len(images)
    for batch_index, start in enumerate(range(0, total, int(batch_size))):
        stop = min(start + int(batch_size), total)
        block = (
            encode_image_features(
                images[start:stop],
                encoder=encoder,
                batch_size=int(batch_size),
                image_range="-1_1",
                device=device,
            )
            .float()
            .cpu()
        )
        blocks.append(block)
        if batch_index == 0 or (batch_index + 1) % 10 == 0 or stop == total:
            print(
                f"[{label}] {stop:,}/{total:,} images encoded "
                f"({100.0 * stop / total:.1f}%).",
                flush=True,
            )
    features = torch.cat(blocks).contiguous()
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError(f"{label} produced non-finite features.")
    return features


def _neighbors_with_progress(query, training, device, chunk_size, label):
    """Compute exact top-two neighbors in visible query chunks."""
    query = query.flatten(1).float().contiguous()
    training = training.flatten(1).float().to(device)
    blocks = {key: [] for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx")}
    total = len(query)
    for batch_index, start in enumerate(range(0, total, int(chunk_size))):
        stop = min(start + int(chunk_size), total)
        stats = compute_nn_stats(
            query[start:stop].to(device),
            training,
            query_batch_size=stop - start,
        )
        for key, values in blocks.items():
            values.append(stats[key].detach().cpu())
        if batch_index == 0 or (batch_index + 1) % 5 == 0 or stop == total:
            print(
                f"[{label}] {stop:,}/{total:,} queries ({100.0 * stop / total:.1f}%).",
                flush=True,
            )
    return {key: torch.cat(values) for key, values in blocks.items()}


def _save_neighbors(path, stats, metadata):
    """Save compact top-two neighbor evidence for later thresholds."""
    payload = {"metadata": metadata}
    payload.update(stats)
    save_torch(path, payload)
    return path


def _curve_record(generated, calibration, validation):
    """Return held-out-calibrated memorization values and the alpha-0.01 result."""
    curve = compute_f_mem_reference_curve(
        generated["r"],
        calibration["r"],
        validation["r"],
        REFERENCE_QUANTILES,
    )
    headline = next(row for row in curve if float(row["heldout_quantile"]) == 0.01)
    return curve, headline


def _expected_generated_bank_identity(run_dir, checkpoint_step, sample_count):
    """Return the exact checkpoint, noise, and sampler identity for one bank."""
    checkpoint_path = run_dir / "checkpoints" / f"step_{checkpoint_step:06d}.pt"
    config = read_json(run_dir / "config.json")
    return {
        "run_dir": str(run_dir),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sample_count": int(sample_count),
        "noise_seed": int(read_json(run_dir / "run.json")["z_init_seed"]),
        "sampling": config["sampling"],
    }


def _load_generated_bank(path, identity):
    """Load one generated bank only when every saved identity field matches."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated bank: {path}")
    payload = load_torch(path)
    if not isinstance(payload, dict):
        raise TypeError("Generated bank must contain a dictionary payload.")
    if payload.get("metadata") != identity:
        raise ValueError(
            "Generated bank checkpoint, sampler, noise seed, or sample count differs."
        )
    samples = payload.get("samples")
    if not torch.is_tensor(samples) or len(samples) != identity["sample_count"]:
        raise ValueError("Generated bank tensor is missing or has the wrong count.")
    samples = samples.contiguous()
    if payload.get("sample_tensor_sha256") != tensor_sha256(samples):
        raise ValueError("Generated bank tensor SHA-256 does not match.")

    summary = payload.get("sampling_summary")
    checkpoint = summary.get("checkpoint", {}) if isinstance(summary, dict) else {}
    timing = summary.get("timing", {}) if isinstance(summary, dict) else {}
    if (
        int(checkpoint.get("step", -1)) != identity["checkpoint_step"]
        or checkpoint.get("sha256") != identity["checkpoint_sha256"]
        or int(timing.get("measured_samples", -1)) != identity["sample_count"]
    ):
        raise ValueError("Generated bank sampling summary does not match its metadata.")
    return samples, summary, path


def _load_or_sample(
    run_dir,
    checkpoint_step,
    sample_count,
    sample_batch_size,
    device,
    output_dir,
    generated_bank=None,
):
    """Reuse a verified generated bank or run the visible sampling benchmark."""
    identity = _expected_generated_bank_identity(run_dir, checkpoint_step, sample_count)
    if generated_bank is not None:
        samples, summary, bank_path = _load_generated_bank(generated_bank, identity)
        print(f"[sample] Reusing explicit bank: {bank_path}", flush=True)
        return samples, summary, True, bank_path

    bank_path = output_dir / "generated_samples.pt"
    if bank_path.is_file():
        samples, summary, bank_path = _load_generated_bank(bank_path, identity)
        print(f"[sample] Reusing {bank_path}", flush=True)
        return samples, summary, True, bank_path

    result = benchmark_checkpoint_sampling(
        run_dir,
        checkpoint_step=checkpoint_step,
        measure_count=sample_count,
        target_count=sample_count,
        batch_size=sample_batch_size,
        warmup_batches=1,
        device=device,
        output_dir=output_dir / "sampling",
        preview_count=16,
        retain_samples=True,
    )
    samples = result["samples"]
    if samples is None or len(samples) != sample_count:
        raise RuntimeError("Sampling benchmark did not retain the complete bank.")
    save_torch(
        bank_path,
        {
            "metadata": identity,
            "samples": samples,
            "sample_tensor_sha256": tensor_sha256(samples),
            "sampling_summary": result["summary"],
        },
    )
    print(f"[sample] Saved reusable bank: {bank_path}", flush=True)
    return samples, result["summary"], False, bank_path


def project_full_run_runtime(
    checkpoint_count,
    sampling_seconds,
    shared_metric_seconds,
    checkpoint_metric_seconds,
):
    """Project one run while charging shared reference work only once."""
    checkpoint_count = int(checkpoint_count)
    per_checkpoint = float(sampling_seconds) + float(checkpoint_metric_seconds)
    all_seconds = float(shared_metric_seconds) + checkpoint_count * per_checkpoint
    return {
        "per_checkpoint_sampling_and_metrics_seconds": per_checkpoint,
        "per_checkpoint_sampling_and_metrics_minutes": per_checkpoint / 60.0,
        "shared_metric_preparation_seconds": float(shared_metric_seconds),
        "all_checkpoints_projected_seconds": all_seconds,
        "all_checkpoints_projected_hours": all_seconds / 3600.0,
    }


def evaluate_checkpoint_metrics(
    run_dir,
    *,
    checkpoint_step,
    reference_artifact,
    assets_manifest,
    output_dir,
    sample_count=10000,
    sample_batch_size=64,
    feature_batch_size=32,
    fid_batch_size=64,
    nn_chunk_size=256,
    device="cuda",
    generated_bank=None,
):
    """Sample once, then compute raw f_mem, SSCD f_mem, and FID-10k."""
    total_started = time.perf_counter()
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(sample_count)
    if sample_count != EXPECTED_REFERENCE_COUNT:
        raise ValueError("This paper-scale benchmark requires exactly 10,000 samples.")

    print("[input] Verifying saved run, test bank, and encoder assets...", flush=True)
    bundle = load_torch(run_dir / "data_bundle.pt")
    data_manifest = read_json(run_dir / "data_manifest.json")
    reference, reference_payload, reference_manifest, reference_path = _load_reference(
        reference_artifact
    )
    _validate_reference_compatibility(
        bundle,
        data_manifest,
        reference,
        reference_payload,
        reference_manifest,
    )
    assets_root, assets, asset_paths, assets_path = _load_assets(assets_manifest)
    print(
        "[input] Frozen CelebA test bank: 10,000 unique images; "
        "zero transformed-tensor overlap with this training bank.",
        flush=True,
    )
    print("[input] Feature space: SSCD 512D at 320 pixels.", flush=True)
    print(
        "[input] Quality metric: Inception FID, 10k generated vs 10k test.", flush=True
    )

    training = bundle["train"].contiguous()
    train_hash = tensor_sha256(training)
    reference_hash = tensor_sha256(reference)
    input_setup_seconds = time.perf_counter() - total_started
    sampling_stage_started = time.perf_counter()
    samples, sampling_summary, samples_reused, generated_bank_path = _load_or_sample(
        run_dir,
        int(checkpoint_step),
        sample_count,
        int(sample_batch_size),
        device,
        output_dir,
        generated_bank=generated_bank,
    )
    if not bool(torch.isfinite(samples).all()):
        raise FloatingPointError("Generated samples contain non-finite values.")
    sample_hash = tensor_sha256(samples)
    sampling_stage_seconds = time.perf_counter() - sampling_stage_started
    common = {
        "train_tensor_sha256": train_hash,
        "reference_tensor_sha256": reference_hash,
        "generated_tensor_sha256": sample_hash,
    }

    metric_root = output_dir / "metrics"
    shared_seconds = 0.0
    checkpoint_seconds = 0.0

    def raw_shared():
        calibration = _neighbors_with_progress(
            reference[:CALIBRATION_COUNT],
            training,
            device,
            nn_chunk_size,
            "raw:calibration",
        )
        validation = _neighbors_with_progress(
            reference[CALIBRATION_COUNT:],
            training,
            device,
            nn_chunk_size,
            "raw:validation",
        )
        return calibration, validation

    (raw_calibration, raw_validation), elapsed = _stage("raw:heldout", raw_shared)
    shared_seconds += elapsed
    raw_generated, elapsed = _stage(
        "raw:generated",
        lambda: _neighbors_with_progress(
            samples, training, device, nn_chunk_size, "raw:generated"
        ),
    )
    checkpoint_seconds += elapsed
    raw_curve, raw_headline = _curve_record(
        raw_generated, raw_calibration, raw_validation
    )
    raw_fixed = compute_f_mem_by_rho(raw_generated["r"], PIXEL_RHOS)
    _save_neighbors(
        metric_root / "raw_pixels" / "generated_neighbors.pt",
        raw_generated,
        {**common, "partition": "generated", "space": "raw_pixels"},
    )
    _save_neighbors(
        metric_root / "raw_pixels" / "calibration_neighbors.pt",
        raw_calibration,
        {**common, "partition": "heldout_calibration", "space": "raw_pixels"},
    )
    _save_neighbors(
        metric_root / "raw_pixels" / "validation_neighbors.pt",
        raw_validation,
        {**common, "partition": "heldout_validation", "space": "raw_pixels"},
    )
    raw_record = {
        "metric": "f_mem_raw_pixels",
        "preprocessing": "raw_native_1x32x32_no_clamp_no_resize",
        "sample_count": sample_count,
        "train_count": len(training),
        "calibration_count": CALIBRATION_COUNT,
        "validation_count": len(reference) - CALIBRATION_COUNT,
        "fixed_rho": raw_fixed,
        "heldout_curve": raw_curve,
        "headline_alpha_0.01": raw_headline,
    }
    write_json(metric_root / "raw_pixels" / "metrics.json", raw_record)

    sscd_encoder = SSCDFeatureEncoder(asset_paths["sscd"], input_size=320)

    def sscd_shared():
        train_features = _encode_with_progress(
            training, sscd_encoder, feature_batch_size, device, "sscd:train"
        )
        reference_features = _encode_with_progress(
            reference, sscd_encoder, feature_batch_size, device, "sscd:reference"
        )
        save_torch(metric_root / "sscd" / "training_features.pt", train_features)
        save_torch(metric_root / "sscd" / "reference_features.pt", reference_features)
        calibration = _neighbors_with_progress(
            reference_features[:CALIBRATION_COUNT],
            train_features,
            device,
            nn_chunk_size,
            "sscd:calibration",
        )
        validation = _neighbors_with_progress(
            reference_features[CALIBRATION_COUNT:],
            train_features,
            device,
            nn_chunk_size,
            "sscd:validation",
        )
        return train_features, calibration, validation

    (train_sscd, sscd_calibration, sscd_validation), elapsed = _stage(
        "sscd:shared", sscd_shared
    )
    shared_seconds += elapsed

    def sscd_generated_stage():
        generated_features = _encode_with_progress(
            samples, sscd_encoder, feature_batch_size, device, "sscd:generated"
        )
        save_torch(metric_root / "sscd" / "generated_features.pt", generated_features)
        neighbors = _neighbors_with_progress(
            generated_features,
            train_sscd,
            device,
            nn_chunk_size,
            "sscd:generated-nn",
        )
        return generated_features, neighbors

    (generated_sscd, sscd_generated), elapsed = _stage(
        "sscd:generated", sscd_generated_stage
    )
    checkpoint_seconds += elapsed
    sscd_curve, sscd_headline = _curve_record(
        sscd_generated, sscd_calibration, sscd_validation
    )
    _save_neighbors(
        metric_root / "sscd" / "generated_neighbors.pt",
        sscd_generated,
        {**common, "partition": "generated", "space": "sscd"},
    )
    _save_neighbors(
        metric_root / "sscd" / "calibration_neighbors.pt",
        sscd_calibration,
        {**common, "partition": "heldout_calibration", "space": "sscd"},
    )
    _save_neighbors(
        metric_root / "sscd" / "validation_neighbors.pt",
        sscd_validation,
        {**common, "partition": "heldout_validation", "space": "sscd"},
    )
    sscd_record = {
        "metric": "f_mem_sscd",
        "preprocessing": "gray_to_rgb_bilinear320_imagenet_norm_final_l2",
        "feature_dim": int(generated_sscd.shape[1]),
        "encoder_sha256": assets["encoders"]["sscd"]["sha256"],
        "sample_count": sample_count,
        "train_count": len(training),
        "calibration_count": CALIBRATION_COUNT,
        "validation_count": len(reference) - CALIBRATION_COUNT,
        "heldout_curve": sscd_curve,
        "headline_alpha_0.01": sscd_headline,
    }
    write_json(metric_root / "sscd" / "metrics.json", sscd_record)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fid_encoder = TorchFidelityFeatureEncoder(
        "torch_fidelity:inception-v3-compat:2048",
        device,
        weights_path=asset_paths["fid_inception_v3"],
    )

    def fid_shared():
        features = _encode_with_progress(
            reference, fid_encoder, fid_batch_size, device, "fid:reference"
        )
        stats = compute_feature_statistics(features)
        save_torch(metric_root / "fid" / "reference_features.pt", features)
        save_torch(metric_root / "fid" / "reference_statistics.pt", stats)
        return stats

    fid_reference_stats, elapsed = _stage("fid:shared", fid_shared)
    shared_seconds += elapsed

    def fid_generated_stage():
        features = _encode_with_progress(
            samples, fid_encoder, fid_batch_size, device, "fid:generated"
        )
        stats = compute_feature_statistics(features)
        print("[fid:distance] Computing the 2048D Fréchet distance...", flush=True)
        value = compute_frechet_distance(stats, fid_reference_stats)
        save_torch(metric_root / "fid" / "generated_features.pt", features)
        save_torch(metric_root / "fid" / "generated_statistics.pt", stats)
        return value

    fid_value, elapsed = _stage("fid:generated", fid_generated_stage)
    checkpoint_seconds += elapsed
    fid_record = {
        "metric": "inception_fid",
        "fid_10k": fid_value,
        "sample_count": sample_count,
        "reference_count": len(reference),
        "reference_split": reference_manifest["reference_split"],
        "feature_extractor": "inception-v3-compat",
        "feature_layer": "2048",
        "encoder_sha256": assets["encoders"]["fid_inception_v3"]["sha256"],
        "reference_artifact_sha256": reference_manifest["artifact_sha256"],
    }
    write_json(metric_root / "fid" / "metrics.json", fid_record)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_count = sampling_summary["checkpoint"]["saved_checkpoint_count"]
    projection_sampling_seconds = sampling_stage_seconds
    projection_kind = "fresh_full_sampling_stage"
    if samples_reused:
        projection_sampling_seconds = sampling_summary["timing"]["total_wall_seconds"]
        projection_kind = "saved_sampling_summary_without_bank_write"
    projection = project_full_run_runtime(
        checkpoint_count,
        projection_sampling_seconds,
        input_setup_seconds + shared_seconds,
        checkpoint_seconds,
    )
    total_wall_seconds = time.perf_counter() - total_started
    summary = {
        "schema_version": METRICS_BENCHMARK_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "checkpoint_step": int(checkpoint_step),
        "sample_count": sample_count,
        "training_count": len(training),
        "reference_count": len(reference),
        "reference_role": "frozen_celeba_test_heldout",
        "reference_artifact": str(reference_path),
        "reference_artifact_sha256": reference_manifest["artifact_sha256"],
        "generated_bank": str(generated_bank_path),
        "generated_tensor_sha256": sample_hash,
        "assets_manifest": str(assets_path),
        "assets_root": str(assets_root),
        "sampling": {
            "reused": samples_reused,
            "settings": config_to_dict(read_json(run_dir / "config.json")["sampling"]),
            "summary": sampling_summary,
        },
        "raw_pixel_f_mem": raw_record,
        "sscd_f_mem": sscd_record,
        "inception_fid": fid_record,
        "timing": {
            "input_verification_seconds": input_setup_seconds,
            "checkpoint_sampling_stage_seconds": sampling_stage_seconds,
            "sampling_kernel_seconds": sampling_summary["timing"][
                "one_checkpoint_sampling_seconds"
            ],
            "shared_metric_preparation_seconds": shared_seconds,
            "checkpoint_metric_seconds": checkpoint_seconds,
            "total_wall_seconds": total_wall_seconds,
            "total_wall_minutes": total_wall_seconds / 60.0,
            "projection_kind": projection_kind,
            **projection,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"[result] FID-10k: {fid_value:.4f} (10k generated vs 10k CelebA test)",
        flush=True,
    )
    print(
        f"[result] raw-pixel f_mem rho=0.3: {raw_fixed[0.3]:.4f}",
        flush=True,
    )
    print(
        "[result] raw-pixel calibrated f_mem alpha=0.01: "
        f"{raw_headline['generated_f_mem']:.4f}",
        flush=True,
    )
    print(
        "[result] SSCD calibrated f_mem alpha=0.01: "
        f"{sscd_headline['generated_f_mem']:.4f}",
        flush=True,
    )
    print(
        f"[result] one checkpoint full wall time: {total_wall_seconds / 60.0:.2f}m",
        flush=True,
    )
    print(
        f"[result] all {checkpoint_count} checkpoints: "
        f"{projection['all_checkpoints_projected_hours']:.2f} GPU-hours",
        flush=True,
    )
    print(f"[result] Complete summary: {output_dir / 'summary.json'}", flush=True)
    return summary


def build_parser():
    """Build the interactive checkpoint-metric command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=1000000)
    parser.add_argument("--reference-artifact", required=True)
    parser.add_argument("--assets-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--generated-bank",
        help="Reuse this verified generated_samples.pt without sampling.",
    )
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--nn-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    """Run the exact interactive sampling and metric benchmark."""
    args = build_parser().parse_args(argv)
    evaluate_checkpoint_metrics(
        args.run_dir,
        checkpoint_step=args.checkpoint_step,
        reference_artifact=args.reference_artifact,
        assets_manifest=args.assets_manifest,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        sample_batch_size=args.sample_batch_size,
        feature_batch_size=args.feature_batch_size,
        fid_batch_size=args.fid_batch_size,
        nn_chunk_size=args.nn_chunk_size,
        device=args.device,
        generated_bank=args.generated_bank,
    )
    return 0


if __name__ == "__main__":
    main()
