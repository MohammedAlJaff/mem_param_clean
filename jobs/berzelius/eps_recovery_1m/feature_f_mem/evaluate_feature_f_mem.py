#!/usr/bin/env python3
"""Replay checkpoints and retain raw pixel/SSCD/LPIPS/DINOv2 f_mem data."""

import argparse
import json
import math
import os
import sys
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_STEPS = (100_000, 390_000, 695_000)
PRIMARY_SPACES = ("full_pixels", "sscd", "lpips", "dinov2")
SENSITIVITY_SPACES = ("sscd_288", "lpips_64", "dinov2_518")
QUANTILES = (0.001, 0.005, 0.01, 0.02, 0.05)
PIXEL_RHOS = (0.3, 0.5, 0.7)
SAMPLING_SETTINGS = {
    "method": "heun",
    "n_steps": 50,
    "start_time": 0.001,
    "end_time": 0.999,
    "use_safe_div": True,
    "safe_div_denominator_floor": 0.05,
    "eps_to_v": "direct",
    "heun_final_euler": False,
    "output": "x_t_max",
    "final_pixel_clamp": False,
}
SPACE_SPECS = {
    "full_pixels": {
        "family": "pixels",
        "resolution": 32,
        "dimension": 1024,
        "preprocessing": "raw_native_1x32x32_no_clamp_no_resize",
    },
    "sscd": {
        "family": "sscd",
        "resolution": 320,
        "dimension": 512,
        "preprocessing": "gray_to_rgb_bilinear320_imagenet_norm_final_l2",
    },
    "sscd_288": {
        "family": "sscd",
        "resolution": 288,
        "dimension": 512,
        "preprocessing": "gray_to_rgb_bilinear288_imagenet_norm_final_l2",
    },
    "lpips": {
        "family": "lpips",
        "resolution": 32,
        "dimension": 5760,
        "preprocessing": "gray_to_rgb_native32_lpips_v0.1_alex_weighted",
    },
    "lpips_64": {
        "family": "lpips",
        "resolution": 64,
        "dimension": 31872,
        "preprocessing": "gray_to_rgb_bilinear64_lpips_v0.1_alex_weighted",
    },
    "dinov2": {
        "family": "dinov2",
        "resolution": 224,
        "dimension": 384,
        "preprocessing": "gray_to_rgb_bicubic224_imagenet_norm_cls_l2",
    },
    "dinov2_518": {
        "family": "dinov2",
        "resolution": 518,
        "dimension": 384,
        "preprocessing": "gray_to_rgb_bicubic518_imagenet_norm_cls_l2",
    },
}
PARAMETERIZATIONS = {
    "eps": ("eps", "eps", "eps_direct_full_heun"),
    "v": ("v", "v", "vv_reference"),
    "x": ("x", "x", "xx_reference"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_specs(seed0_root, multiseed_root=None, scope="seed0"):
    seed0_root = Path(seed0_root).expanduser().resolve()
    specs = [
        (0, "eps", seed0_root / "1_eps_direct_full_heun_s0"),
        (0, "v", seed0_root / "6_vv_reference_s0"),
        (0, "x", seed0_root / "7_xx_reference_s0"),
    ]
    if scope == "all":
        if multiseed_root is None:
            raise ValueError("--multiseed-root is required for scope=all.")
        root = Path(multiseed_root).expanduser().resolve()
        specs.extend(
            [
                (1, "eps", root / "0_eps_direct_full_heun_s1"),
                (1, "v", root / "1_vv_reference_s1"),
                (1, "x", root / "2_xx_reference_s1"),
                (2, "eps", root / "3_eps_direct_full_heun_s2"),
                (2, "v", root / "4_vv_reference_s2"),
                (2, "x", root / "5_xx_reference_s2"),
            ]
        )
    return specs


def _architecture_signature(config):
    return {
        "dataset": config["data"]["dataset"],
        "n_train": int(config["data"]["n_train"]),
        "image_size": int(config["data"]["image_size"]),
        "network": config["network"],
        "optimizer": config["training"]["optimizer"],
        "lr": float(config["training"]["lr"]),
        "batch_cap": int(config["training"]["batch_cap"]),
        "loss_weighting": config["framework"]["loss_weighting"],
    }


def build_manifest(seed0_root, multiseed_root=None, scope="seed0"):
    tasks = []
    signatures = []
    for seed, parameterization, run_dir in source_specs(
        seed0_root, multiseed_root, scope
    ):
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing source run: {run_dir}")
        config_path = run_dir / "config.json"
        config = read_json(config_path)
        prediction, loss, _ = PARAMETERIZATIONS[parameterization]
        if int(config["training"]["seed"]) != seed:
            raise ValueError(f"Training seed disagrees for {run_dir}.")
        if config["framework"]["prediction_space"] != prediction:
            raise ValueError(f"Prediction space disagrees for {run_dir}.")
        if config["framework"]["loss_space"] != loss:
            raise ValueError(f"Loss space disagrees for {run_dir}.")
        signature = _architecture_signature(config)
        signatures.append(signature)
        if signature["n_train"] != 256 or signature["batch_cap"] != 256:
            raise ValueError("Feature validation requires N=batch=256 training runs.")
        if not math.isclose(signature["lr"], 1.0e-4):
            raise ValueError("Feature validation requires Adam at 1e-4.")
        checkpoints = []
        for step in CHECKPOINT_STEPS:
            path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected checkpoint: {path}")
            checkpoints.append(str(path.resolve()))
        run_record = read_json(run_dir / "run.json")
        bundle_path = run_dir / "data_bundle.pt"
        bundle_hash = run_record.get("data_bundle_sha256") or file_sha256(bundle_path)
        tasks.append(
            {
                "task_index": len(tasks),
                "seed": seed,
                "parameterization": parameterization,
                "prediction_space": prediction,
                "loss_space": loss,
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "config_path": str(config_path.resolve()),
                "config_sha256": file_sha256(config_path),
                "data_bundle_path": str(bundle_path.resolve()),
                "data_bundle_sha256": bundle_hash,
                "checkpoints": checkpoints,
            }
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Architecture, data, optimizer, or weighting differs across runs.")
    for seed in sorted({task["seed"] for task in tasks}):
        hashes = {
            task["data_bundle_sha256"] for task in tasks if task["seed"] == seed
        }
        if len(hashes) != 1:
            raise ValueError(f"Parameterizations for seed {seed} use different data.")
    expected = 3 if scope == "seed0" else 9
    if len(tasks) != expected:
        raise ValueError(f"Expected {expected} evaluation tasks, got {len(tasks)}.")
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "sampling": SAMPLING_SETTINGS,
        "tasks": tasks,
    }


def load_assets(manifest_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stage_encoder_assets import verify

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = verify(manifest_path)
    return manifest_path.parent, manifest


def asset_path(root, manifest, key):
    return root / manifest["encoders"][key]["path"]


def encoder_weight_hashes(family, manifest):
    keys = {
        "pixels": (),
        "sscd": ("sscd",),
        "lpips": ("lpips_calibration", "lpips_alexnet"),
        "dinov2": ("dinov2_weights",),
    }[family]
    return {key: manifest["encoders"][key]["sha256"] for key in keys}


def make_encoder(space, assets_root, assets, device):
    from mem_param_src.metrics.image import (
        DINOv2FeatureEncoder,
        LPIPSFeatureEncoder,
        SSCDFeatureEncoder,
    )

    spec = SPACE_SPECS[space]
    family = spec["family"]
    if family == "pixels":
        return None
    if family == "sscd":
        return SSCDFeatureEncoder(
            asset_path(assets_root, assets, "sscd"), spec["resolution"]
        )
    if family == "lpips":
        os.environ["TORCH_HOME"] = str(assets_root / assets["torch_home"])
        return LPIPSFeatureEncoder(
            input_size=spec["resolution"],
            weights_path=asset_path(assets_root, assets, "lpips_calibration"),
        )
    return DINOv2FeatureEncoder(
        assets_root / assets["dinov2_source"]["path"],
        asset_path(assets_root, assets, "dinov2_weights"),
        spec["resolution"],
    )


def feature_metadata(space, image_hash, source_hash, assets):
    spec = SPACE_SPECS[space]
    return {
        "schema_version": 1,
        "space": space,
        "family": spec["family"],
        "resolution": spec["resolution"],
        "preprocessing": spec["preprocessing"],
        "image_tensor_sha256": image_hash,
        "source_artifact_sha256": source_hash,
        "encoder_weight_sha256": encoder_weight_hashes(spec["family"], assets),
        "encoder_package_versions": assets["package_versions"],
        "layer": {
            "pixels": "raw_flattened",
            "sscd": "final_512_l2",
            "lpips": "relu1_relu5_weighted_normalized",
            "dinov2": "final_cls_l2",
        }[spec["family"]],
    }


def load_or_encode(path, images, *, space, source_hash, assets_root, assets, device, batch_size):
    import torch

    from mem_param_src.metrics.image import encode_image_features
    from mem_param_src.runners.artifacts import load_torch, save_torch, tensor_sha256

    path = Path(path)
    image_hash = tensor_sha256(images)
    metadata = feature_metadata(space, image_hash, source_hash, assets)
    if path.exists():
        payload = load_torch(path)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"Feature cache metadata mismatch: {path}")
        features = payload["features"].contiguous()
        if tensor_sha256(features) != payload.get("feature_tensor_sha256"):
            raise RuntimeError(f"Feature cache tensor hash mismatch: {path}")
        return features

    encoder = make_encoder(space, assets_root, assets, device)
    if encoder is None:
        features = images.detach().cpu().float().flatten(1).contiguous()
    else:
        safe_batch_size = min(
            int(batch_size),
            {
                "sscd": 32,
                "sscd_288": 32,
                "lpips_64": 32,
                "dinov2": 32,
                "dinov2_518": 4,
            }.get(space, int(batch_size)),
        )
        features = encode_image_features(
            images,
            encoder=encoder,
            batch_size=safe_batch_size,
            image_range="-1_1",
            device=device,
        ).contiguous()
    if len(features) != len(images) or not bool(torch.isfinite(features).all()):
        raise RuntimeError(f"Nonfinite or misaligned features in {space}.")
    expected = SPACE_SPECS[space]["dimension"]
    if expected is not None and int(features.shape[1]) != expected:
        raise RuntimeError(
            f"{space} produced {features.shape[1]} features, expected {expected}."
        )
    save_torch(
        path,
        {
            "metadata": metadata,
            "features": features,
            "feature_tensor_sha256": tensor_sha256(features),
            "shape": list(features.shape),
            "dtype": str(features.dtype),
        },
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features


def load_reference(reference_path, expected_count=10_000):
    from mem_param_src.runners.artifacts import file_sha256, load_torch, tensor_sha256

    reference_path = Path(reference_path).expanduser().resolve()
    manifest_path = reference_path.with_suffix(".json")
    payload = load_torch(reference_path)
    manifest = read_json(manifest_path)
    if file_sha256(reference_path) != manifest["artifact_sha256"]:
        raise RuntimeError("Frozen validation artifact hash mismatch.")
    reference = payload["reference"].contiguous()
    if len(reference) != expected_count:
        raise RuntimeError(f"Frozen validation count is {len(reference)}, not {expected_count}.")
    if tensor_sha256(reference) != manifest["reference_tensor_sha256"]:
        raise RuntimeError("Frozen validation tensor hash mismatch.")
    return reference, payload, manifest, reference_path


def fid_features(images, encoder, batch_size):
    from mem_param_src.metrics.image import encode_image_features

    return encode_image_features(
        images,
        encoder=encoder,
        batch_size=batch_size,
        image_range="-1_1",
        device=encoder.device,
    ).numpy()


def covariance_stats(features):
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    return values.mean(0), np.cov(values, rowvar=False)


def fid_from_stats(sample_mean, sample_covariance, reference_mean, reference_covariance):
    import numpy as np
    from scipy import linalg

    difference = sample_mean - reference_mean
    covariance_mean, _ = linalg.sqrtm(
        sample_covariance @ reference_covariance, disp=False
    )
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sample_covariance.shape[0]) * 1.0e-6
        covariance_mean = linalg.sqrtm(
            (sample_covariance + offset) @ (reference_covariance + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = difference.dot(difference) + np.trace(
        sample_covariance + reference_covariance - 2.0 * covariance_mean
    )
    return float(max(value, 0.0))


def prepare_reference(args):
    from mem_param_src.metrics.image import TorchFidelityFeatureEncoder
    from mem_param_src.runners.artifacts import save_torch

    assets_root, assets = load_assets(args.assets_manifest)
    reference, _, reference_manifest, reference_path = load_reference(
        args.reference_artifact
    )
    output_root = Path(args.output_root).expanduser().resolve()
    feature_root = output_root / "features" / "reference"
    reference_hash = reference_manifest["reference_tensor_sha256"]
    source_hash = reference_manifest["artifact_sha256"]
    for space in PRIMARY_SPACES + SENSITIVITY_SPACES:
        load_or_encode(
            feature_root / f"{space}.pt",
            reference,
            space=space,
            source_hash=source_hash,
            assets_root=assets_root,
            assets=assets,
            device=args.device,
            batch_size=args.feature_batch_size,
        )

    inception_weights = asset_path(assets_root, assets, "fid_inception_v3")
    encoder = TorchFidelityFeatureEncoder(
        "torch_fidelity:inception-v3-compat:2048",
        args.device,
        weights_path=inception_weights,
    )
    features = fid_features(reference, encoder, args.fid_batch_size)
    mean, covariance = covariance_stats(features)
    save_torch(
        output_root / "features" / "reference" / "fid_inception_v3_2048.pt",
        {
            "mean": mean,
            "covariance": covariance,
            "count": len(reference),
            "reference_artifact_sha256": source_hash,
            "reference_tensor_sha256": reference_hash,
            "encoder_weight_sha256": assets["encoders"]["fid_inception_v3"]["sha256"],
            "feature_layer": "2048",
        },
    )
    write_json(
        output_root / "manifests" / "reference.json",
        {
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": source_hash,
            "reference_tensor_sha256": reference_hash,
            "reference_count": len(reference),
            "calibration_count": 1024,
            "validation_count": len(reference) - 1024,
            "spaces": list(PRIMARY_SPACES + SENSITIVITY_SPACES),
            "assets_manifest": str(Path(args.assets_manifest).resolve()),
        },
    )
    print(output_root / "manifests" / "reference.json", flush=True)


def raw_sample_stats(samples):
    import torch

    finite = torch.isfinite(samples)
    values = samples[finite]
    return {
        "finite_fraction": float(finite.float().mean()),
        "raw_min": float(values.min()) if values.numel() else math.nan,
        "raw_max": float(values.max()) if values.numel() else math.nan,
        "raw_rms": float(values.square().mean().sqrt()) if values.numel() else math.nan,
        "outside_minus1_1_fraction": float(
            (((samples < -1.0) | (samples > 1.0)) & finite).float().mean()
        ),
    }


def sample_in_batches(model, noise, config, device, batch_size):
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(noise), batch_size):
            outputs.append(
                model.sample(
                    noise[start : start + batch_size].to(device),
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
    return torch.cat(outputs).contiguous()


def compute_and_save_stats(path, query, train, metadata, device, chunk_size):
    from mem_param_src.metrics import compute_nn_stats
    from mem_param_src.runners.artifacts import load_torch, save_torch, tensor_sha256

    path = Path(path)
    if path.exists():
        payload = load_torch(path)
        if payload.get("metadata") != metadata:
            raise RuntimeError(f"Nearest-neighbor cache metadata mismatch: {path}")
        for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
            if tensor_sha256(payload[key]) != payload.get(f"{key}_sha256"):
                raise RuntimeError(
                    f"Nearest-neighbor cache tensor hash mismatch: {path} ({key})"
                )
        return payload
    stats = compute_nn_stats(
        query.to(device),
        train.to(device),
        query_batch_size=chunk_size,
        reference_batch_size=chunk_size,
    )
    payload = {"metadata": metadata}
    for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
        payload[key] = stats[key].detach().cpu().contiguous()
        payload[f"{key}_sha256"] = tensor_sha256(payload[key])
    save_torch(path, payload)
    return payload

def identity_set(value):
    if value is None:
        return set()
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    return {str(item) for item in value}



def json_sequence(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    return list(value)


def validate_noise(path, count=10_000):
    import torch

    from mem_param_src.runners.artifacts import file_sha256, load_torch, tensor_sha256

    path = Path(path).expanduser().resolve()
    payload = load_torch(path)
    noise = payload["tensor"].contiguous()
    if int(payload.get("seed", -1)) != 2003 or len(noise) != count:
        raise RuntimeError("Common evaluation noise must be seed 2003 with 10,000 samples.")
    if tuple(noise.shape[1:]) != (1, 32, 32) or noise.dtype != torch.float32:
        raise RuntimeError("Common evaluation noise must be float32 [10000,1,32,32].")
    return noise, file_sha256(path), tensor_sha256(noise), path


def resolve_task(manifest, task_index):
    tasks = {int(task["task_index"]): task for task in manifest["tasks"]}
    if task_index not in tasks:
        raise ValueError(f"Task {task_index} is not in the evaluation manifest.")
    return tasks[task_index]


def run_task(args):
    import torch
    from ml_collections import ConfigDict

    from mem_param_src.metrics.image import (
        TorchFidelityFeatureEncoder,
        compute_f_mem_by_rho,
        compute_f_mem_reference_curve,
    )
    from mem_param_src.model import GDFM
    from mem_param_src.runners import configure_runtime, resolve_device
    from mem_param_src.runners.artifacts import (
        file_sha256,
        load_torch,
        save_torch,
        tensor_sha256,
    )

    if args.sample_count != 10_000:
        raise ValueError("Full feature validation requires exactly 10,000 samples.")
    manifest = read_json(args.manifest)
    task = resolve_task(manifest, args.task_index)
    assets_root, assets = load_assets(args.assets_manifest)
    reference, reference_payload, reference_manifest, reference_path = load_reference(
        args.reference_artifact
    )
    noise, noise_file_hash, noise_hash, noise_path = validate_noise(
        args.noise_artifact, args.sample_count
    )
    run_dir = Path(task["run_dir"])
    if file_sha256(run_dir / "data_bundle.pt") != task["data_bundle_sha256"]:
        raise RuntimeError("Training data bundle no longer matches the evaluation manifest.")
    saved_bundle = load_torch(run_dir / "data_bundle.pt")
    train = saved_bundle["train"].contiguous()
    if len(train) != 256:
        raise RuntimeError("Expected exactly 256 training images.")
    if identity_set(saved_bundle.get("train_sample_ids")).intersection(
        identity_set(reference_payload.get("reference_sample_ids"))
    ):
        raise RuntimeError("Frozen validation artifact overlaps training identities.")

    output_root = Path(args.output_root).expanduser().resolve()
    task_dir = output_root / "tasks" / (
        f"{task['task_index']:02d}_s{task['seed']}_{task['parameterization']}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    config = ConfigDict(read_json(run_dir / "config.json"))
    for key, value in SAMPLING_SETTINGS.items():
        config.sampling[key] = value
    configure_runtime(config, device)
    model = GDFM(config).to(device)
    train_hash = tensor_sha256(train)
    reference_hash = reference_manifest["reference_tensor_sha256"]
    reference_source_hash = reference_manifest["artifact_sha256"]
    records = []

    for step in CHECKPOINT_STEPS:
        checkpoint_path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint = load_torch(checkpoint_path)
        if int(checkpoint.get("step", -1)) != step:
            raise RuntimeError(f"Checkpoint metadata disagrees: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        step_dir = task_dir / f"step_{step:06d}"
        bank_path = step_dir / "generated" / "samples.pt"
        bank_metadata = {
            "checkpoint_sha256": checkpoint_hash,
            "noise_file_sha256": noise_file_hash,
            "noise_tensor_sha256": noise_hash,
            "sampling": SAMPLING_SETTINGS,
            "sample_count": args.sample_count,
        }
        if bank_path.exists():
            bank = load_torch(bank_path)
            if bank.get("metadata") != bank_metadata:
                raise RuntimeError(f"Generated bank metadata mismatch: {bank_path}")
            samples = bank["samples"].contiguous()
            if tensor_sha256(samples) != bank["sample_tensor_sha256"]:
                raise RuntimeError(f"Generated bank hash mismatch: {bank_path}")
        else:
            started = time.perf_counter()
            samples = sample_in_batches(
                model, noise, config, device, args.sample_batch_size
            )
            save_torch(
                bank_path,
                {
                    "metadata": bank_metadata,
                    "samples": samples,
                    "sample_tensor_sha256": tensor_sha256(samples),
                    "runtime_sec": time.perf_counter() - started,
                    "raw_stats": raw_sample_stats(samples),
                },
            )
        if tuple(samples.shape) != (10_000, 1, 32, 32):
            raise RuntimeError(f"Unexpected generated bank shape: {tuple(samples.shape)}")
        sample_hash = tensor_sha256(samples)
        sample_stats = raw_sample_stats(samples)

        fid_reference = load_torch(
            output_root / "features" / "reference" / "fid_inception_v3_2048.pt"
        )
        if fid_reference["reference_artifact_sha256"] != reference_source_hash:
            raise RuntimeError("FID reference cache uses another validation artifact.")
        if fid_reference["reference_tensor_sha256"] != reference_hash:
            raise RuntimeError("FID reference cache tensor hash is incorrect.")
        if fid_reference["encoder_weight_sha256"] != assets["encoders"][
            "fid_inception_v3"
        ]["sha256"]:
            raise RuntimeError(
                "FID reference cache uses another Inception weight file."
            )
        fid_encoder = TorchFidelityFeatureEncoder(
            "torch_fidelity:inception-v3-compat:2048",
            device,
            weights_path=asset_path(assets_root, assets, "fid_inception_v3"),
        )
        fid_mean, fid_covariance = covariance_stats(
            fid_features(samples, fid_encoder, args.fid_batch_size)
        )
        fid_value = fid_from_stats(
            fid_mean,
            fid_covariance,
            fid_reference["mean"],
            fid_reference["covariance"],
        )
        write_json(
            step_dir / "manifests" / "fid.json",
            {
                "fid_10k": fid_value,
                "sample_count": 10_000,
                "reference_count": 10_000,
                "reference_artifact_sha256": reference_source_hash,
                "feature_extractor": "inception-v3-compat",
                "feature_layer": "2048",
                "weight_sha256": assets["encoders"]["fid_inception_v3"]["sha256"],
            },
        )
        del fid_encoder
        torch.cuda.empty_cache()

        spaces = list(PRIMARY_SPACES)
        if task["seed"] == 0 and step == 390_000:
            spaces.extend(SENSITIVITY_SPACES)
        for space in spaces:
            reference_features = load_or_encode(
                output_root / "features" / "reference" / f"{space}.pt",
                reference,
                space=space,
                source_hash=reference_source_hash,
                assets_root=assets_root,
                assets=assets,
                device=device,
                batch_size=args.feature_batch_size,
            )
            train_features = load_or_encode(
                task_dir / "features" / "train" / f"{space}.pt",
                train,
                space=space,
                source_hash=task["data_bundle_sha256"],
                assets_root=assets_root,
                assets=assets,
                device=device,
                batch_size=args.feature_batch_size,
            )
            generated_features = load_or_encode(
                step_dir / "features" / f"{space}.pt",
                samples,
                space=space,
                source_hash=file_sha256(bank_path),
                assets_root=assets_root,
                assets=assets,
                device=device,
                batch_size=args.feature_batch_size,
            )
            common_metadata = {
                "space": space,
                "train_tensor_sha256": train_hash,
                "reference_tensor_sha256": reference_hash,
                "train_feature_sha256": tensor_sha256(train_features),
            }
            calibration_stats = compute_and_save_stats(
                task_dir / "nn_stats" / "reference" / space / "calibration.pt",
                reference_features[:1024],
                train_features,
                {**common_metadata, "partition": "calibration", "count": 1024},
                device,
                args.nn_chunk_size,
            )
            validation_stats = compute_and_save_stats(
                task_dir / "nn_stats" / "reference" / space / "validation.pt",
                reference_features[1024:],
                train_features,
                {
                    **common_metadata,
                    "partition": "validation",
                    "count": len(reference_features) - 1024,
                },
                device,
                args.nn_chunk_size,
            )
            generated_stats = compute_and_save_stats(
                step_dir / "nn_stats" / f"{space}.pt",
                generated_features,
                train_features,
                {
                    **common_metadata,
                    "partition": "generated",
                    "count": len(generated_features),
                    "generated_tensor_sha256": sample_hash,
                    "generated_feature_sha256": tensor_sha256(generated_features),
                },
                device,
                args.nn_chunk_size,
            )
            curve = compute_f_mem_reference_curve(
                generated_stats["r"],
                calibration_stats["r"],
                validation_stats["r"],
                QUANTILES,
            )
            row = {
                "seed": task["seed"],
                "parameterization": task["parameterization"],
                "prediction_space": task["prediction_space"],
                "loss_space": task["loss_space"],
                "step": step,
                "space": space,
                "metric_path": (
                    "eval/full_pixels"
                    if space == "full_pixels"
                    else f"eval/features/{SPACE_SPECS[space]['family']}"
                ),
                "resolution": SPACE_SPECS[space]["resolution"],
                "primary_space": space in PRIMARY_SPACES,
                "sample_count": len(samples),
                "train_count": len(train),
                "calibration_count": 1024,
                "validation_count": len(reference) - 1024,
                "generated_tensor_sha256": sample_hash,
                "reference_artifact_sha256": reference_source_hash,
                "fid_10k": fid_value,
                "raw_sample_stats": sample_stats,
                "heldout_curve": curve,
                "pixel_f_mem": (
                    compute_f_mem_by_rho(generated_stats["r"], PIXEL_RHOS)
                    if space == "full_pixels"
                    else None
                ),
                "generated_nn_stats": str(
                    step_dir / "nn_stats" / f"{space}.pt"
                ),
                "calibration_nn_stats": str(
                    task_dir / "nn_stats" / "reference" / space / "calibration.pt"
                ),
                "validation_nn_stats": str(
                    task_dir / "nn_stats" / "reference" / space / "validation.pt"
                ),
                "generated_features": str(step_dir / "features" / f"{space}.pt"),
                "train_features": str(task_dir / "features" / "train" / f"{space}.pt"),
            }
            write_json(step_dir / "manifests" / f"{space}.json", row)
            records.append(row)

    jsonl = task_dir / "checkpoint_metrics.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = jsonl.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(jsonl)
    write_json(
        task_dir / "settings.json",
        {
            "task": task,
            "assets_manifest": str(Path(args.assets_manifest).resolve()),
            "reference_artifact": str(reference_path),
            "sample_identities": {
                "train_kind": saved_bundle.get("train_sample_id_kind"),
                "train_ids": json_sequence(saved_bundle.get("train_sample_ids")),
                "train_source_indices": json_sequence(
                    saved_bundle.get("train_indices")
                ),
                "validation_kind": reference_payload.get(
                    "reference_sample_id_kind"
                ),
                "validation_ids": json_sequence(
                    reference_payload.get("reference_sample_ids")
                ),
                "validation_source_indices": json_sequence(
                    reference_payload.get("reference_indices")
                ),
                "generated_kind": "seed2003_noise_bank_row",
                "generated_rows": [0, args.sample_count - 1],
                "generated_row_order_is_inclusive_range": True,
            },
            "noise_artifact": str(noise_path),
            "sampling": SAMPLING_SETTINGS,
            "spaces": list(PRIMARY_SPACES),
            "sensitivity_spaces": (
                list(SENSITIVITY_SPACES) if task["seed"] == 0 else []
            ),
        },
    )
    print(jsonl, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--seed0-root", required=True)
    manifest_parser.add_argument("--multiseed-root")
    manifest_parser.add_argument("--scope", choices=("seed0", "all"), default="seed0")
    manifest_parser.add_argument("--output")

    reference_parser = subparsers.add_parser("prepare-reference")
    reference_parser.add_argument("--assets-manifest", required=True)
    reference_parser.add_argument("--reference-artifact", required=True)
    reference_parser.add_argument("--output-root", required=True)
    reference_parser.add_argument("--device", default="cuda")
    reference_parser.add_argument("--feature-batch-size", type=int, default=64)
    reference_parser.add_argument("--fid-batch-size", type=int, default=128)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--task-index", type=int, required=True)
    run_parser.add_argument("--assets-manifest", required=True)
    run_parser.add_argument("--reference-artifact", required=True)
    run_parser.add_argument("--noise-artifact", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--sample-count", type=int, default=10_000)
    run_parser.add_argument("--sample-batch-size", type=int, default=256)
    run_parser.add_argument("--feature-batch-size", type=int, default=64)
    run_parser.add_argument("--fid-batch-size", type=int, default=128)
    run_parser.add_argument("--nn-chunk-size", type=int, default=512)
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose manifest, prepare-reference, or run")

    if args.command == "manifest":
        manifest = build_manifest(
            args.seed0_root, args.multiseed_root, scope=args.scope
        )
        if args.output:
            write_json(args.output, manifest)
            print(Path(args.output).resolve())
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "prepare-reference":
        prepare_reference(args)
    else:
        run_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
