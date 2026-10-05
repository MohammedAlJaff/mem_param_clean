"""Dataset-aware, restartable metrics for selected saved image checkpoints."""

import csv
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mem_param_src import __version__
from mem_param_src.config import config_to_dict
from mem_param_src.metrics.image import (
    MNIST_F_MEM_ALPHAS,
    MNIST_JOINTAE_CLASS_FD_NAME,
    MNIST_JOINTAE_F_MEM_NAME,
    MNIST_JOINTAE_FD_NAME,
    DINOv2FeatureEncoder,
    LPIPSFeatureEncoder,
    SSCDFeatureEncoder,
    TorchFidelityFeatureEncoder,
    compute_f_mem_by_rho,
    compute_feature_statistics,
    compute_frechet_distance,
    compute_mnist_jointae32_class_fd,
    encode_image_features,
    extract_mnist_jointae32_features,
    load_mnist_jointae32,
    load_mnist_jointae32_manifest,
    load_mnist_jointae32_reference_stats,
    normalize_mnist_memorization_features,
)
from mem_param_src.metrics.nearest_neighbors import compute_nn_stats
from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime, resolve_device
from mem_param_src.runners.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    file_sha256,
    list_checkpoints,
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    validate_data_manifest,
    write_json,
)
from mem_param_src.runners.evaluation import (
    _sample_checkpoint,
    resolve_evaluation_config,
)

IMAGE_EVALUATION_SCHEMA_VERSION = 1
_PROTOCOL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NATURAL_IMAGE_DATASETS = {
    "afhq_v2",
    "celeba",
    "celeba32",
    "celeba32_grey",
    "celeba64",
    "celeba64_rgb",
    "cifar10",
    "ffhq128",
    "flowers102",
}
MNIST_FEATURE_SPACES = {"dataset_default", "mnist_jointae32"}
NATURAL_FEATURE_SPACES = {
    "dataset_default",
    "inception_v3",
    "sscd",
    "dinov2",
    "lpips",
}
NATURAL_SPACE_SPECS = {
    "inception_v3": {
        "dimension": 2048,
        "layer": "2048",
        "preprocessing": "rgb_uint8_inception_v3_compat",
    },
    "sscd": {
        "dimension": 512,
        "layer": "final_512_l2",
        "preprocessing": "rgb_bilinear320_imagenet_norm_final_l2",
    },
    "dinov2": {
        "dimension": 384,
        "layer": "final_cls_l2",
        "preprocessing": "rgb_bicubic224_imagenet_norm_cls_l2",
    },
    "lpips": {
        "dimension": 5760,
        "layer": "relu1_relu5_weighted_normalized",
        "preprocessing": "rgb_bilinear32_lpips_v0.1_alex_weighted",
    },
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _split_csv(value):
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def resolve_image_metric_protocol(config, metrics, feature_spaces):
    """Resolve public metric aliases to one dataset-compatible protocol."""
    dataset = str(config.data.dataset).lower()
    metrics = _split_csv(metrics)
    feature_spaces = _split_csv(feature_spaces)
    if not metrics:
        raise ValueError("Select at least one image metric.")
    if not feature_spaces:
        feature_spaces = ("dataset_default",)

    if dataset == "mnist":
        if int(config.data.image_size) not in (32, 64):
            raise ValueError("MNIST JointAE evaluation supports MNIST32 and MNIST64.")
        unsupported_spaces = set(feature_spaces) - MNIST_FEATURE_SPACES
        if unsupported_spaces:
            names = ", ".join(sorted(unsupported_spaces))
            raise ValueError(
                f"MNIST evaluation does not use natural-image spaces: {names}."
            )
        aliases = {
            "fid": MNIST_JOINTAE_FD_NAME,
            "mnist_fd": MNIST_JOINTAE_FD_NAME,
            MNIST_JOINTAE_FD_NAME: MNIST_JOINTAE_FD_NAME,
            "class_fd": MNIST_JOINTAE_CLASS_FD_NAME,
            MNIST_JOINTAE_CLASS_FD_NAME: MNIST_JOINTAE_CLASS_FD_NAME,
            "f_mem": MNIST_JOINTAE_F_MEM_NAME,
            MNIST_JOINTAE_F_MEM_NAME: MNIST_JOINTAE_F_MEM_NAME,
        }
        unknown = [metric for metric in metrics if metric not in aliases]
        if unknown:
            raise ValueError(f"Unsupported MNIST metrics: {', '.join(unknown)}.")
        metric_manifest = load_mnist_jointae32_manifest()
        resolved_metrics = list(dict.fromkeys(aliases[item] for item in metrics))
        return {
            "dataset_family": "mnist",
            "dataset": dataset,
            "image_size": int(config.data.image_size),
            "metrics": resolved_metrics,
            "feature_spaces": ["mnist_jointae32"],
            "quality_metric": (
                MNIST_JOINTAE_FD_NAME
                if MNIST_JOINTAE_FD_NAME in resolved_metrics
                else None
            ),
            "encoder_sha256": metric_manifest["weight_asset"]["sha256"],
            "feature_layer": "raw_linear_128",
            "feature_dim": 128,
            "preprocessing": metric_manifest["preprocessing"],
        }

    if dataset not in NATURAL_IMAGE_DATASETS:
        raise ValueError(f"No image metric protocol is registered for {dataset!r}.")
    unsupported_spaces = set(feature_spaces) - NATURAL_FEATURE_SPACES
    if unsupported_spaces:
        names = ", ".join(sorted(unsupported_spaces))
        raise ValueError(f"Unsupported natural-image feature spaces: {names}.")
    aliases = {
        "fid": "inception_fid",
        "inception_fid": "inception_fid",
        "f_mem": "feature_f_mem",
        "feature_f_mem": "feature_f_mem",
    }
    unknown = [metric for metric in metrics if metric not in aliases]
    if unknown:
        raise ValueError(
            f"Unsupported natural-image metrics: {', '.join(unknown)}."
        )
    resolved_metrics = list(dict.fromkeys(aliases[item] for item in metrics))
    requested_spaces = list(
        dict.fromkeys(
            "inception_v3" if item == "dataset_default" else item
            for item in feature_spaces
        )
    )
    learned_spaces = [
        item for item in requested_spaces if item in {"sscd", "dinov2", "lpips"}
    ]
    if "inception_fid" in resolved_metrics and "inception_v3" not in requested_spaces:
        raise ValueError(
            "Natural-image fid requires dataset_default or inception_v3."
        )
    if "feature_f_mem" in resolved_metrics and not learned_spaces:
        raise ValueError(
            "Natural-image f_mem requires at least one of sscd, dinov2, or lpips."
        )
    if "feature_f_mem" not in resolved_metrics and learned_spaces:
        raise ValueError(
            "SSCD, DINOv2, and LPIPS are memorization spaces; request f_mem."
        )
    active_spaces = []
    if "inception_fid" in resolved_metrics:
        active_spaces.append("inception_v3")
    if "feature_f_mem" in resolved_metrics:
        active_spaces.extend(learned_spaces)
    return {
        "dataset_family": "natural_rgb",
        "dataset": dataset,
        "image_size": int(config.data.image_size),
        "metrics": resolved_metrics,
        "feature_spaces": active_spaces,
        "memorization_spaces": learned_spaces,
        "feature_definitions": {
            space: NATURAL_SPACE_SPECS[space] for space in active_spaces
        },
        "quality_metric": (
            "inception_fid" if "inception_fid" in resolved_metrics else None
        ),
    }


def resolve_checkpoint_selection(run_dir, selection):
    """Return selected saved checkpoint records in optimizer-step order."""
    available = dict(list_checkpoints(run_dir))
    if selection is None or str(selection).strip().lower() == "all":
        return sorted(available.items())
    if isinstance(selection, str):
        requested = [int(value.strip()) for value in selection.split(",")]
    else:
        requested = [int(value) for value in selection]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Checkpoint selection must contain unique optimizer steps.")
    missing = [step for step in requested if step not in available]
    if missing:
        raise ValueError(f"Requested checkpoints do not exist: {missing}.")
    return [(step, available[step]) for step in requested]


def _raw_sample_statistics(samples):
    finite = torch.isfinite(samples)
    values = samples[finite]
    return {
        "finite_fraction": float(finite.float().mean()),
        "raw_min": float(values.min()) if values.numel() else None,
        "raw_max": float(values.max()) if values.numel() else None,
        "raw_rms": (
            float(values.square().mean().sqrt()) if values.numel() else None
        ),
        "outside_minus1_1_fraction": float(
            (((samples < -1.0) | (samples > 1.0)) & finite).float().mean()
        ),
    }


def _load_noise(run_dir, noise_path):
    path = Path(noise_path) if noise_path else Path(run_dir) / "z_init.pt"
    path = path.expanduser().resolve()
    payload = load_torch(path)
    seed = None
    if torch.is_tensor(payload):
        noise = payload
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("tensor")):
        noise = payload["tensor"]
        if payload.get("seed") is not None:
            seed = int(payload["seed"])
    else:
        raise ValueError("Evaluation noise must be a tensor or contain key 'tensor'.")
    return noise.contiguous(), path, seed


def _evaluation_root(run_dir, protocol_id, output_dir):
    if output_dir is None:
        if not _PROTOCOL_ID_PATTERN.fullmatch(str(protocol_id)):
            raise ValueError(
                "protocol_id must be a path-safe name containing only letters, "
                "digits, dots, underscores, and hyphens."
            )
        root = Path(run_dir) / "evaluations" / str(protocol_id)
    else:
        root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_identity(
    run_dir,
    config,
    protocol,
    checkpoints,
    sample_count,
    noise_path,
    noise,
    noise_seed,
    encoder_assets=None,
):
    identity = {
        "schema_version": IMAGE_EVALUATION_SCHEMA_VERSION,
        "source_run": str(Path(run_dir).resolve()),
        "source_config_sha256": file_sha256(Path(run_dir) / "config.json"),
        "source_data_bundle_sha256": file_sha256(
            Path(run_dir) / "data_bundle.pt"
        ),
        "protocol": protocol,
        "checkpoint_steps": [step for step, _ in checkpoints],
        "sample_count": int(sample_count),
        "noise": {
            "path": str(noise_path),
            "file_sha256": file_sha256(noise_path),
            "tensor_sha256": tensor_sha256(noise[:sample_count]),
            "seed": noise_seed,
        },
        "sampling": config_to_dict(config.sampling),
        "image_range": "-1_1",
        "environment": {
            "mem_param": __version__,
            "torch": torch.__version__,
        },
    }
    if encoder_assets is not None:
        identity["encoder_assets"] = encoder_assets
    return identity


def _prepare_manifest(root, identity):
    path = root / "eval_manifest.json"
    if path.exists():
        existing = read_json(path)
        comparable = {
            key: value
            for key, value in existing.items()
            if key
            not in {
                "created_at",
                "elapsed_seconds",
                "failure",
                "status",
                "updated_at",
            }
        }
        if comparable != identity:
            raise ValueError(
                "Existing evaluation manifest does not match this protocol request."
            )
        return existing
    manifest = {
        **identity,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "status": "running",
    }
    write_json(path, manifest)
    return manifest


def _load_or_sample(
    step_dir,
    checkpoint_path,
    model,
    noise,
    config,
    device,
    sample_count,
    noise_seed,
):
    sample_path = step_dir / "samples.pt"
    manifest_path = step_dir / "sample_manifest.json"
    expected = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "noise_tensor_sha256": tensor_sha256(noise[:sample_count]),
        "noise_seed": noise_seed,
        "sample_count": int(sample_count),
        "sampling": config_to_dict(config.sampling),
    }
    if sample_path.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        samples = load_torch(sample_path)
        if (
            torch.is_tensor(samples)
            and {key: manifest.get(key) for key in expected} == expected
            and manifest.get("generated_tensor_sha256") == tensor_sha256(samples)
        ):
            return samples, manifest, True

    checkpoint = load_torch(checkpoint_path)
    if int(checkpoint.get("step", -1)) != int(checkpoint_path.stem.split("_")[-1]):
        raise ValueError(f"Checkpoint step does not match its path: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    started = time.perf_counter()
    samples = _sample_checkpoint(model, noise[:sample_count], config, device)
    samples = samples.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(samples).all()):
        raise FloatingPointError("Generated samples contain non-finite values.")
    manifest = {
        **expected,
        "generated_tensor_sha256": tensor_sha256(samples),
        "raw_statistics": _raw_sample_statistics(samples),
        "runtime_seconds": time.perf_counter() - started,
        "created_at": _utc_now(),
    }
    save_torch(sample_path, samples)
    write_json(manifest_path, manifest)
    return samples, manifest, False


def _load_or_encode(path, images, source_hash, encoder, batch_size, device):
    manifest = load_mnist_jointae32_manifest()
    metadata = {
        "source_tensor_sha256": source_hash,
        "encoder": "mnist_jointae32_v1",
        "encoder_sha256": manifest["weight_asset"]["sha256"],
        "feature_layer": "raw_linear_128",
        "preprocessing": manifest["preprocessing"],
        "feature_dim": 128,
        "package_version": __version__,
    }
    if path.exists():
        payload = load_torch(path)
        features = payload.get("features") if isinstance(payload, dict) else None
        logits = payload.get("logits") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("metadata") == metadata
            and torch.is_tensor(features)
            and payload.get("feature_tensor_sha256") == tensor_sha256(features)
            and torch.is_tensor(logits)
            and payload.get("logit_tensor_sha256") == tensor_sha256(logits)
        ):
            return features, logits, True

    started = time.perf_counter()
    features, logits = extract_mnist_jointae32_features(
        images,
        encoder=encoder,
        batch_size=batch_size,
        image_range="-1_1",
        device=device,
        return_logits=True,
    )
    payload = {
        "metadata": metadata,
        "features": features,
        "logits": logits,
        "feature_tensor_sha256": tensor_sha256(features),
        "logit_tensor_sha256": tensor_sha256(logits),
        "runtime_seconds": time.perf_counter() - started,
    }
    save_torch(path, payload)
    return features, logits, False


def _load_or_neighbors(path, query, training, metadata, batch_size, device):
    expected = {
        **metadata,
        "query_feature_sha256": tensor_sha256(query),
        "training_feature_sha256": tensor_sha256(training),
        "normalization": "training_bank_coordinate_zscore_population_then_l2",
    }
    if path.exists():
        payload = load_torch(path)
        if isinstance(payload, dict) and payload.get("metadata") == expected:
            valid = True
            for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
                value = payload.get(key)
                if not torch.is_tensor(value) or payload.get(
                    f"{key}_sha256"
                ) != tensor_sha256(value):
                    valid = False
            if valid:
                return payload, True

    query_vectors, training_vectors, mean, scale = (
        normalize_mnist_memorization_features(query, training)
    )
    started = time.perf_counter()
    stats = compute_nn_stats(
        query_vectors.to(device),
        training_vectors.to(device),
        query_batch_size=batch_size,
        reference_batch_size=batch_size,
    )
    payload = {
        "metadata": expected,
        "training_feature_mean": mean.squeeze(0),
        "training_feature_scale": scale.squeeze(0),
        "runtime_seconds": time.perf_counter() - started,
    }
    for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
        payload[key] = stats[key].detach().cpu().contiguous()
        payload[f"{key}_sha256"] = tensor_sha256(payload[key])
    save_torch(path, payload)
    return payload, False


def _load_or_feature_stats(path, features, metadata):
    expected = {
        **metadata,
        "feature_tensor_sha256": tensor_sha256(features),
        "statistics_dtype": "float64",
        "covariance_normalization": "N-1",
    }
    if path.exists():
        payload = load_torch(path)
        mean = payload.get("mean") if isinstance(payload, dict) else None
        covariance = (
            payload.get("covariance") if isinstance(payload, dict) else None
        )
        if (
            isinstance(payload, dict)
            and payload.get("metadata") == expected
            and torch.is_tensor(mean)
            and payload.get("mean_sha256") == tensor_sha256(mean)
            and torch.is_tensor(covariance)
            and payload.get("covariance_sha256")
            == tensor_sha256(covariance)
        ):
            return payload, True

    started = time.perf_counter()
    statistics = compute_feature_statistics(features)
    payload = {
        "metadata": expected,
        **statistics,
        "mean_sha256": tensor_sha256(statistics["mean"]),
        "covariance_sha256": tensor_sha256(statistics["covariance"]),
        "runtime_seconds": time.perf_counter() - started,
    }
    save_torch(path, payload)
    return payload, False


def _load_or_frechet_distance(path, first, second, metadata):
    identity = {
        **metadata,
        "first_mean_sha256": tensor_sha256(first["mean"]),
        "first_covariance_sha256": tensor_sha256(first["covariance"]),
        "second_mean_sha256": tensor_sha256(second["mean"]),
        "second_covariance_sha256": tensor_sha256(second["covariance"]),
    }
    if path.exists():
        payload = read_json(path)
        if payload.get("identity") == identity:
            return float(payload["distance"]), True
    started = time.perf_counter()
    distance = compute_frechet_distance(first, second)
    write_json(
        path,
        {
            "distance": distance,
            "identity": identity,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    return distance, False


def _encoder_asset_hashes(space, assets):
    keys = {
        "inception_v3": ("fid_inception_v3",),
        "sscd": ("sscd",),
        "dinov2": ("dinov2_weights",),
        "lpips": ("lpips_calibration", "lpips_alexnet"),
    }[space]
    return {key: assets["encoders"][key]["sha256"] for key in keys}


def _git_head_commit(repository):
    git_path = Path(repository) / ".git"
    if git_path.is_file():
        pointer = git_path.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError(f"Invalid Git metadata pointer: {git_path}")
        git_path = (git_path.parent / pointer.removeprefix("gitdir: ")).resolve()
    head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_path / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_path / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise ValueError(f"Cannot resolve Git HEAD for {repository}.")


def _load_encoder_assets(manifest_path, spaces):
    if manifest_path is None:
        raise ValueError(
            "Natural-image protocol evaluation requires --assets-manifest."
        )
    path = Path(manifest_path).expanduser().resolve()
    assets = read_json(path)
    if int(assets.get("schema_version", -1)) != 3:
        raise ValueError("Encoder asset manifest must use schema version 3.")
    root = path.parent
    required = set()
    for space in spaces:
        required.update(_encoder_asset_hashes(space, assets))
    for key in sorted(required):
        record = assets.get("encoders", {}).get(key)
        if record is None:
            raise ValueError(f"Encoder asset manifest is missing {key!r}.")
        asset_path = (root / record["path"]).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(f"Missing encoder asset: {asset_path}")
        if file_sha256(asset_path) != record["sha256"]:
            raise ValueError(f"Encoder asset hash mismatch: {asset_path}")
    if "dinov2" in spaces:
        source = assets.get("dinov2_source", {})
        source_root = (root / source.get("path", "")).resolve()
        if not (source_root / "hubconf.py").is_file() or not source.get("commit"):
            raise ValueError("DINOv2 source checkout is missing or unpinned.")
        if _git_head_commit(source_root) != source["commit"]:
            raise ValueError("DINOv2 source checkout does not match its pinned commit.")
    return root, assets, {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema_version": 3,
        "encoders": {
            space: _encoder_asset_hashes(space, assets) for space in spaces
        },
        "package_versions": assets["package_versions"],
    }


def _natural_encoder(space, assets_root, assets, device):
    def asset_path(key):
        return assets_root / assets["encoders"][key]["path"]

    if space == "inception_v3":
        return TorchFidelityFeatureEncoder(
            "torch_fidelity:inception-v3-compat:2048",
            device,
            weights_path=asset_path("fid_inception_v3"),
        )
    if space == "sscd":
        return SSCDFeatureEncoder(asset_path("sscd"), input_size=320)
    if space == "lpips":
        os.environ["TORCH_HOME"] = str(assets_root / assets["torch_home"])
        return LPIPSFeatureEncoder(
            input_size=32,
            weights_path=asset_path("lpips_calibration"),
        )
    if space == "dinov2":
        return DINOv2FeatureEncoder(
            assets_root / assets["dinov2_source"]["path"],
            asset_path("dinov2_weights"),
            input_size=224,
        )
    raise ValueError(f"Unknown natural-image feature space: {space!r}.")


def _load_or_encode_natural(
    path,
    images,
    source_hash,
    *,
    space,
    encoder,
    assets,
    batch_size,
    device,
):
    spec = NATURAL_SPACE_SPECS[space]
    metadata = {
        "source_tensor_sha256": source_hash,
        "space": space,
        "encoder_asset_sha256": _encoder_asset_hashes(space, assets),
        "feature_layer": spec["layer"],
        "preprocessing": spec["preprocessing"],
        "feature_dim": spec["dimension"],
        "encoder_package_versions": assets["package_versions"],
        "package_version": __version__,
    }
    if path.exists():
        payload = load_torch(path)
        features = payload.get("features") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("metadata") == metadata
            and torch.is_tensor(features)
            and payload.get("feature_tensor_sha256") == tensor_sha256(features)
        ):
            return features, True

    safe_batch_size = min(
        int(batch_size),
        {"inception_v3": 64, "sscd": 32, "lpips": 32, "dinov2": 32}[space],
    )
    started = time.perf_counter()
    features = encode_image_features(
        images,
        encoder=encoder,
        batch_size=safe_batch_size,
        image_range="-1_1",
        device=device,
    ).float().contiguous()
    if (
        features.dim() != 2
        or len(features) != len(images)
        or int(features.shape[1]) != spec["dimension"]
        or not bool(torch.isfinite(features).all())
    ):
        raise RuntimeError(f"Invalid {space} feature matrix: {tuple(features.shape)}")
    payload = {
        "metadata": metadata,
        "features": features,
        "feature_tensor_sha256": tensor_sha256(features),
        "runtime_seconds": time.perf_counter() - started,
    }
    save_torch(path, payload)
    return features, False


def _load_or_raw_neighbors(path, query, training, metadata, batch_size, device):
    expected = {
        **metadata,
        "query_feature_sha256": tensor_sha256(query),
        "training_feature_sha256": tensor_sha256(training),
        "normalization": "none",
    }
    if path.exists():
        payload = load_torch(path)
        if isinstance(payload, dict) and payload.get("metadata") == expected:
            valid = True
            for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
                value = payload.get(key)
                if not torch.is_tensor(value) or payload.get(
                    f"{key}_sha256"
                ) != tensor_sha256(value):
                    valid = False
            if valid:
                return payload, True

    started = time.perf_counter()
    stats = compute_nn_stats(
        query.to(device),
        training.to(device),
        query_batch_size=batch_size,
        reference_batch_size=batch_size,
    )
    payload = {
        "metadata": expected,
        "runtime_seconds": time.perf_counter() - started,
    }
    for key in ("d1", "d2", "r", "nn_idx", "second_nn_idx"):
        payload[key] = stats[key].detach().cpu().contiguous()
        payload[f"{key}_sha256"] = tensor_sha256(payload[key])
    save_torch(path, payload)
    return payload, False


def _calibration_records(calibration_r, validation_r):
    records = []
    for alpha in MNIST_F_MEM_ALPHAS:
        rho = float(torch.quantile(calibration_r.float(), alpha))
        records.append(
            {
                "alpha": float(alpha),
                "rho": rho,
                "calibration_rate": compute_f_mem_by_rho(calibration_r, rho)[rho],
                "validation_rate": compute_f_mem_by_rho(validation_r, rho)[rho],
            }
        )
    return records


def _write_combined(root, records):
    combined = root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    write_json(combined / "summary.json", {"checkpoints": records})
    path = combined / "checkpoint_metrics.csv"
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = [
        "step",
        "checkpoint",
        "metric",
        "value",
        "sample_count",
        "alpha",
        "rho",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            for row in record["table_rows"]:
                writer.writerow(row)
    os.replace(temporary, path)


def _evaluate_mnist(
    root,
    checkpoints,
    bundle,
    noise,
    config,
    protocol,
    device,
    sample_count,
    feature_batch_size,
    nn_batch_size,
    noise_seed,
):
    encoder = load_mnist_jointae32(device=device)
    training_features = None
    thresholds = None
    if MNIST_JOINTAE_F_MEM_NAME in protocol["metrics"]:
        reference_root = root / "reference"
        reference_root.mkdir(parents=True, exist_ok=True)
        training = bundle["train"].contiguous()
        heldout = bundle["reference"].contiguous()
        if len(heldout) < 2:
            raise ValueError("MNIST threshold calibration requires held-out images.")

        training_features, _, _ = _load_or_encode(
            reference_root / "training_features.pt",
            training,
            tensor_sha256(training),
            encoder,
            feature_batch_size,
            device,
        )
        heldout_features, _, _ = _load_or_encode(
            reference_root / "heldout_features.pt",
            heldout,
            tensor_sha256(heldout),
            encoder,
            feature_batch_size,
            device,
        )
        calibration_count = min(1024, max(1, len(heldout_features) // 2))
        calibration, _ = _load_or_neighbors(
            reference_root / "calibration_neighbors.pt",
            heldout_features[:calibration_count],
            training_features,
            {"partition": "heldout_calibration", "count": calibration_count},
            nn_batch_size,
            device,
        )
        validation, _ = _load_or_neighbors(
            reference_root / "validation_neighbors.pt",
            heldout_features[calibration_count:],
            training_features,
            {
                "partition": "heldout_validation",
                "count": len(heldout_features) - calibration_count,
            },
            nn_batch_size,
            device,
        )
        thresholds = _calibration_records(calibration["r"], validation["r"])
        write_json(
            reference_root / "calibration.json",
            {
                "alphas": list(MNIST_F_MEM_ALPHAS),
                "calibration_count": calibration_count,
                "validation_count": len(heldout_features) - calibration_count,
                "records": thresholds,
            },
        )

    model = GDFM(config).to(device)
    records = []
    for step, checkpoint_path in checkpoints:
        step_dir = root / "checkpoints" / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        samples, sample_manifest, reused_samples = _load_or_sample(
            step_dir,
            checkpoint_path,
            model,
            noise,
            config,
            device,
            sample_count,
            noise_seed,
        )
        feature_dir = step_dir / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        features, logits, reused_features = _load_or_encode(
            feature_dir / "mnist_jointae.pt",
            samples,
            sample_manifest["generated_tensor_sha256"],
            encoder,
            feature_batch_size,
            device,
        )
        metrics = {
            "step": step,
            "checkpoint": checkpoint_path.name,
            "sample_count": sample_count,
            "sample_reused": reused_samples,
            "features_reused": reused_features,
            "raw_sample_statistics": sample_manifest["raw_statistics"],
        }
        table_rows = []

        if MNIST_JOINTAE_FD_NAME in protocol["metrics"]:
            generated_stats, reused_stats = _load_or_feature_stats(
                feature_dir / "mnist_jointae_stats.pt",
                features,
                {
                    "encoder": "mnist_jointae32_v1",
                    "source_tensor_sha256": sample_manifest[
                        "generated_tensor_sha256"
                    ],
                },
            )
            reference_stats = load_mnist_jointae32_reference_stats()["global"]
            encoder_manifest = load_mnist_jointae32_manifest()
            fd_value, reused_distance = _load_or_frechet_distance(
                step_dir / "fd_cache.json",
                generated_stats,
                reference_stats,
                {
                    "metric": MNIST_JOINTAE_FD_NAME,
                    "encoder_sha256": encoder_manifest["weight_asset"]["sha256"],
                    "reference_statistics_sha256": encoder_manifest[
                        "reference_statistics_asset"
                    ]["sha256"],
                },
            )
            fd_record = {
                "metric": MNIST_JOINTAE_FD_NAME,
                "mnist_jointae_fd": fd_value,
                "n_samples": int(generated_stats["count"]),
                "n_reference": int(reference_stats["count"]),
                "feature_dim": int(generated_stats["feature_dim"]),
                "encoder_sha256": encoder_manifest["weight_asset"]["sha256"],
                "preprocessing": encoder_manifest["preprocessing"],
                "statistics_reused": reused_stats,
                "distance_reused": reused_distance,
            }
            metrics[MNIST_JOINTAE_FD_NAME] = fd_record
            write_json(step_dir / "fd.json", fd_record)
            table_rows.append(
                {
                    "step": step,
                    "checkpoint": checkpoint_path.name,
                    "metric": MNIST_JOINTAE_FD_NAME,
                    "value": fd_record["mnist_jointae_fd"],
                    "sample_count": sample_count,
                    "alpha": "",
                    "rho": "",
                }
            )

        if MNIST_JOINTAE_CLASS_FD_NAME in protocol["metrics"]:
            class_record = compute_mnist_jointae32_class_fd(
                features,
                logits.argmax(1),
            )
            confidence = logits.softmax(1).amax(1)
            class_record["mean_classifier_confidence"] = float(confidence.mean())
            metrics[MNIST_JOINTAE_CLASS_FD_NAME] = class_record
            table_rows.append(
                {
                    "step": step,
                    "checkpoint": checkpoint_path.name,
                    "metric": MNIST_JOINTAE_CLASS_FD_NAME,
                    "value": class_record["macro_present_class_fd"],
                    "sample_count": sample_count,
                    "alpha": "",
                    "rho": "",
                }
            )

        if MNIST_JOINTAE_F_MEM_NAME in protocol["metrics"]:
            if training_features is None or thresholds is None:
                raise RuntimeError("MNIST memorization references were not prepared.")
            neighbor_dir = step_dir / "nearest_neighbors"
            neighbor_dir.mkdir(parents=True, exist_ok=True)
            neighbors, reused_neighbors = _load_or_neighbors(
                neighbor_dir / "mnist_jointae.pt",
                features,
                training_features,
                {"partition": "generated", "count": len(features)},
                nn_batch_size,
                device,
            )
            curve = []
            for threshold in thresholds:
                value = compute_f_mem_by_rho(
                    neighbors["r"], threshold["rho"]
                )[threshold["rho"]]
                row = {**threshold, "f_mem": value}
                curve.append(row)
                table_rows.append(
                    {
                        "step": step,
                        "checkpoint": checkpoint_path.name,
                        "metric": MNIST_JOINTAE_F_MEM_NAME,
                        "value": value,
                        "sample_count": sample_count,
                        "alpha": threshold["alpha"],
                        "rho": threshold["rho"],
                    }
                )
            metrics[MNIST_JOINTAE_F_MEM_NAME] = {
                "curve": curve,
                "neighbors_reused": reused_neighbors,
                "neighbor_artifact": "nearest_neighbors/mnist_jointae.pt",
            }

        write_json(
            feature_dir / "mnist_jointae.COMPLETE.json",
            {
                "completed_at": _utc_now(),
                "feature_artifact_sha256": file_sha256(
                    feature_dir / "mnist_jointae.pt"
                ),
                "metrics": protocol["metrics"],
                "source_tensor_sha256": sample_manifest[
                    "generated_tensor_sha256"
                ],
            },
        )
        metrics["table_rows"] = table_rows
        write_json(step_dir / "metrics.json", metrics)
        write_json(
            step_dir / "COMPLETE.json",
            {
                "completed_at": _utc_now(),
                "generated_tensor_sha256": sample_manifest[
                    "generated_tensor_sha256"
                ],
                "metrics_sha256": file_sha256(step_dir / "metrics.json"),
            },
        )
        records.append(metrics)
    return records


def _evaluate_natural(
    root,
    checkpoints,
    bundle,
    noise,
    config,
    protocol,
    device,
    sample_count,
    feature_batch_size,
    nn_batch_size,
    noise_seed,
    assets_root,
    assets,
):
    training = bundle["train"].contiguous()
    reference = bundle["reference"].contiguous()
    if "inception_fid" in protocol["metrics"] and len(reference) < sample_count:
        raise ValueError(
            "Natural-image FID requires at least sample_count saved reference images."
        )
    if "feature_f_mem" in protocol["metrics"] and (
        len(training) < 2 or len(reference) < 2
    ):
        raise ValueError(
            "Natural-image memorization requires at least two training and "
            "two held-out images."
        )

    encoders = {
        space: _natural_encoder(space, assets_root, assets, device)
        for space in protocol["feature_spaces"]
    }
    reference_root = root / "reference"
    reference_root.mkdir(parents=True, exist_ok=True)
    fid_reference_stats = None
    if "inception_fid" in protocol["metrics"]:
        fid_reference_features, _ = _load_or_encode_natural(
            reference_root / "inception_v3_features.pt",
            reference[:sample_count],
            tensor_sha256(reference[:sample_count]),
            space="inception_v3",
            encoder=encoders["inception_v3"],
            assets=assets,
            batch_size=feature_batch_size,
            device=device,
        )
        fid_reference_stats, _ = _load_or_feature_stats(
            reference_root / "inception_v3_stats.pt",
            fid_reference_features,
            {
                "space": "inception_v3",
                "partition": "heldout_reference",
                "count": sample_count,
                "encoder_asset_sha256": _encoder_asset_hashes(
                    "inception_v3", assets
                ),
            },
        )

    training_features = {}
    thresholds = {}
    for space in protocol["memorization_spaces"]:
        space_root = reference_root / space
        space_root.mkdir(parents=True, exist_ok=True)
        training_features[space], _ = _load_or_encode_natural(
            space_root / "training_features.pt",
            training,
            tensor_sha256(training),
            space=space,
            encoder=encoders[space],
            assets=assets,
            batch_size=feature_batch_size,
            device=device,
        )
        heldout_features, _ = _load_or_encode_natural(
            space_root / "heldout_features.pt",
            reference,
            tensor_sha256(reference),
            space=space,
            encoder=encoders[space],
            assets=assets,
            batch_size=feature_batch_size,
            device=device,
        )
        calibration_count = min(1024, max(1, len(heldout_features) // 2))
        calibration, _ = _load_or_raw_neighbors(
            space_root / "calibration_neighbors.pt",
            heldout_features[:calibration_count],
            training_features[space],
            {"space": space, "partition": "heldout_calibration"},
            nn_batch_size,
            device,
        )
        validation, _ = _load_or_raw_neighbors(
            space_root / "validation_neighbors.pt",
            heldout_features[calibration_count:],
            training_features[space],
            {"space": space, "partition": "heldout_validation"},
            nn_batch_size,
            device,
        )
        thresholds[space] = _calibration_records(
            calibration["r"], validation["r"]
        )
        write_json(
            space_root / "calibration.json",
            {
                "space": space,
                "alphas": list(MNIST_F_MEM_ALPHAS),
                "calibration_count": calibration_count,
                "validation_count": len(heldout_features) - calibration_count,
                "records": thresholds[space],
            },
        )

    model = GDFM(config).to(device)
    records = []
    for step, checkpoint_path in checkpoints:
        step_started = time.perf_counter()
        step_dir = root / "checkpoints" / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        samples, sample_manifest, reused = _load_or_sample(
            step_dir,
            checkpoint_path,
            model,
            noise,
            config,
            device,
            sample_count,
            noise_seed,
        )
        record = {
            "step": step,
            "checkpoint": checkpoint_path.name,
            "sample_count": sample_count,
            "sample_reused": reused,
            "raw_sample_statistics": sample_manifest["raw_statistics"],
            "table_rows": [],
        }
        feature_dir = step_dir / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        generated_features = {}
        reused_features = {}

        for space in protocol["feature_spaces"]:
            generated_features[space], reused_features[space] = (
                _load_or_encode_natural(
                    feature_dir / f"{space}.pt",
                    samples,
                    sample_manifest["generated_tensor_sha256"],
                    space=space,
                    encoder=encoders[space],
                    assets=assets,
                    batch_size=feature_batch_size,
                    device=device,
                )
            )

        if "inception_fid" in protocol["metrics"]:
            generated_stats, reused_stats = _load_or_feature_stats(
                feature_dir / "inception_v3_stats.pt",
                generated_features["inception_v3"],
                {
                    "space": "inception_v3",
                    "partition": "generated",
                    "encoder_asset_sha256": _encoder_asset_hashes(
                        "inception_v3", assets
                    ),
                },
            )
            if fid_reference_stats is None:
                raise RuntimeError("Inception reference statistics were not prepared.")
            fid_value, reused_distance = _load_or_frechet_distance(
                step_dir / "fd_cache.json",
                generated_stats,
                fid_reference_stats,
                {
                    "metric": "inception_fid",
                    "encoder_sha256": assets["encoders"]["fid_inception_v3"][
                        "sha256"
                    ],
                },
            )
            fid = {
                "metric": "inception_fid",
                "fid": fid_value,
                "frechet_inception_distance": fid_value,
                "n_samples": int(generated_stats["count"]),
                "n_reference": int(fid_reference_stats["count"]),
                "feature_dim": int(generated_stats["feature_dim"]),
                "feature_extractor": "inception-v3-compat",
                "feature_layer": "2048",
                "encoder_sha256": assets["encoders"]["fid_inception_v3"][
                    "sha256"
                ],
                "preprocessing": NATURAL_SPACE_SPECS["inception_v3"][
                    "preprocessing"
                ],
                "features_reused": reused_features["inception_v3"],
                "statistics_reused": reused_stats,
                "distance_reused": reused_distance,
            }
            record["inception_fid"] = fid
            record["table_rows"].append(
                {
                    "step": step,
                    "checkpoint": checkpoint_path.name,
                    "metric": "inception_fid",
                    "value": fid_value,
                    "sample_count": sample_count,
                    "alpha": "",
                    "rho": "",
                }
            )
            write_json(step_dir / "fd.json", fid)

        for space in protocol["memorization_spaces"]:
            neighbor_dir = step_dir / "nearest_neighbors"
            neighbor_dir.mkdir(parents=True, exist_ok=True)
            neighbors, reused_neighbors = _load_or_raw_neighbors(
                neighbor_dir / f"{space}.pt",
                generated_features[space],
                training_features[space],
                {"space": space, "partition": "generated"},
                nn_batch_size,
                device,
            )
            curve = []
            for threshold in thresholds[space]:
                value = compute_f_mem_by_rho(
                    neighbors["r"], threshold["rho"]
                )[threshold["rho"]]
                curve.append({**threshold, "f_mem": value})
                record["table_rows"].append(
                    {
                        "step": step,
                        "checkpoint": checkpoint_path.name,
                        "metric": f"f_mem_{space}",
                        "value": value,
                        "sample_count": sample_count,
                        "alpha": threshold["alpha"],
                        "rho": threshold["rho"],
                    }
                )
            record[f"f_mem_{space}"] = {
                "curve": curve,
                "features_reused": reused_features[space],
                "neighbors_reused": reused_neighbors,
                "neighbor_artifact": f"nearest_neighbors/{space}.pt",
                "encoder_sha256": _encoder_asset_hashes(space, assets),
                "preprocessing": NATURAL_SPACE_SPECS[space]["preprocessing"],
                "feature_dim": NATURAL_SPACE_SPECS[space]["dimension"],
            }

        for space in protocol["feature_spaces"]:
            completion = {
                "completed_at": _utc_now(),
                "feature_artifact_sha256": file_sha256(
                    feature_dir / f"{space}.pt"
                ),
                "source_tensor_sha256": sample_manifest[
                    "generated_tensor_sha256"
                ],
                "space": space,
            }
            stats_path = feature_dir / f"{space}_stats.pt"
            if stats_path.is_file():
                completion["statistics_artifact_sha256"] = file_sha256(stats_path)
            neighbor_path = step_dir / "nearest_neighbors" / f"{space}.pt"
            if neighbor_path.is_file():
                completion["neighbor_artifact_sha256"] = file_sha256(neighbor_path)
            write_json(feature_dir / f"{space}.COMPLETE.json", completion)

        record["elapsed_seconds"] = time.perf_counter() - step_started
        write_json(step_dir / "metrics.json", record)
        write_json(
            step_dir / "COMPLETE.json",
            {
                "completed_at": _utc_now(),
                "generated_tensor_sha256": sample_manifest[
                    "generated_tensor_sha256"
                ],
                "metrics_sha256": file_sha256(step_dir / "metrics.json"),
            },
        )
        records.append(record)
    return records


def evaluate_image_protocol(
    run_dir,
    *,
    checkpoints="all",
    metrics=("fid",),
    feature_spaces=("dataset_default",),
    sample_count=10000,
    protocol_id="dataset-default-v1",
    output_dir=None,
    noise_path=None,
    assets_manifest=None,
    eval_config=None,
    override_items=(),
    device="auto",
    feature_batch_size=256,
    nn_batch_size=512,
):
    """Evaluate selected checkpoints from exact saved artifacts with reuse."""
    run_dir = Path(run_dir).expanduser().resolve()
    run_record = read_json(run_dir / "run.json")
    if int(run_record.get("artifact_format_version", -1)) != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Saved run uses an unsupported artifact format version.")
    if str(run_record.get("run_kind")) != "image":
        raise ValueError("The image protocol evaluator requires an image run.")

    config, _ = resolve_evaluation_config(
        run_dir,
        eval_config=eval_config,
        override_items=override_items,
    )
    protocol = resolve_image_metric_protocol(config, metrics, feature_spaces)
    assets_root = None
    assets = None
    assets_identity = None
    if protocol["dataset_family"] == "natural_rgb":
        assets_root, assets, assets_identity = _load_encoder_assets(
            assets_manifest, protocol["feature_spaces"]
        )
    selected = resolve_checkpoint_selection(run_dir, checkpoints)
    sample_count = int(sample_count)
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2 for Fréchet statistics.")
    noise, resolved_noise_path, embedded_noise_seed = _load_noise(
        run_dir, noise_path
    )
    noise_seed = embedded_noise_seed
    if noise_seed is None and noise_path is None:
        noise_seed = run_record.get("z_init_seed")
    if len(noise) < sample_count:
        raise ValueError(
            f"Evaluation noise has {len(noise)} rows, fewer than {sample_count}."
        )

    bundle = load_torch(run_dir / "data_bundle.pt")
    validate_data_manifest(run_dir, bundle, run_kind="image", verify_hashes=True)
    root = _evaluation_root(run_dir, protocol_id, output_dir)
    identity = _manifest_identity(
        run_dir,
        config,
        protocol,
        selected,
        sample_count,
        resolved_noise_path,
        noise,
        noise_seed,
        assets_identity,
    )
    manifest = _prepare_manifest(root, identity)

    started = time.perf_counter()
    try:
        device = resolve_device(device)
        configure_runtime(config, device)
        if protocol["dataset_family"] == "mnist":
            records = _evaluate_mnist(
                root,
                selected,
                bundle,
                noise,
                config,
                protocol,
                device,
                sample_count,
                int(feature_batch_size),
                int(nn_batch_size),
                noise_seed,
            )
        else:
            if assets_root is None or assets is None:
                raise RuntimeError("Natural-image encoder assets were not prepared.")
            records = _evaluate_natural(
                root,
                selected,
                bundle,
                noise,
                config,
                protocol,
                device,
                sample_count,
                int(feature_batch_size),
                int(nn_batch_size),
                noise_seed,
                assets_root,
                assets,
            )
        _write_combined(root, records)
    except Exception as error:
        manifest.update(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {
                    "message": str(error),
                    "type": error.__class__.__name__,
                },
                "status": "failed",
                "updated_at": _utc_now(),
            }
        )
        write_json(root / "eval_manifest.json", manifest)
        raise
    manifest.pop("failure", None)
    manifest.update(
        {
            "elapsed_seconds": time.perf_counter() - started,
            "status": "complete",
            "updated_at": _utc_now(),
        }
    )
    write_json(root / "eval_manifest.json", manifest)
    return root


__all__ = [
    "IMAGE_EVALUATION_SCHEMA_VERSION",
    "evaluate_image_protocol",
    "resolve_checkpoint_selection",
    "resolve_image_metric_protocol",
]
