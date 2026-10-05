"""Frozen MNIST JointAE-32 features, distance, and memorization metrics."""

import hashlib
import io
import json
from importlib import resources
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from mem_param_src.metrics.image.preprocessing import normalize_images
from mem_param_src.metrics.nearest_neighbors import compute_f_mem, compute_nn_stats

MNIST_JOINTAE_ARCHITECTURE_VERSION = "mnist_jointae32_v1"
MNIST_JOINTAE_FEATURE_DIM = 128
MNIST_JOINTAE_FD_NAME = "mnist_fd_jointae32_v1"
MNIST_JOINTAE_CLASS_FD_NAME = "mnist_cafd_jointae32_v1"
MNIST_JOINTAE_F_MEM_NAME = "f_mem_mnist_jointae32_v1"
MNIST_F_MEM_ALPHAS = (0.005, 0.01, 0.02, 0.05)

_ASSET_PACKAGE = "mem_param_src.metrics.image"
_ASSET_DIRECTORY = "assets"
_WEIGHT_ASSET = "mnist_jointae32_v1.pt"
_MANIFEST_ASSET = "mnist_jointae32_v1.json"
_REFERENCE_ASSET = "mnist_jointae32_v1_reference.pt"


class _ResBlock(nn.Module):
    """Residual block used by the frozen JointAE architecture."""

    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs):
        return inputs + self.block(inputs)


class _MNISTJointAE32(nn.Module):
    """Exact model architecture that produced the bundled checkpoint."""

    def __init__(self, feature_dim=MNIST_JOINTAE_FEATURE_DIM):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            _ResBlock(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            _ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            _ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            _ResBlock(256),
        )
        self.feature_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.to_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, self.feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )
        self.from_feature = nn.Linear(self.feature_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            _ResBlock(256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            _ResBlock(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            _ResBlock(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images):
        hidden = self.feature_pool(self.encoder(images))
        return self.to_feature(hidden)

    def forward(self, images):
        features = self.encode(images)
        logits = self.classifier(features)
        reconstruction = self.from_feature(features).view(-1, 256, 4, 4)
        reconstruction = self.decoder(reconstruction)
        return reconstruction, logits, features


def canonicalize_mnist32(images, *, image_range):
    """Map declared MNIST images to grayscale float tensors at 32 by 32."""
    images = normalize_images(images, image_range=image_range)
    channels = int(images.shape[1])
    if channels == 3:
        if not torch.allclose(images[:, 0], images[:, 1]) or not torch.allclose(
            images[:, 1], images[:, 2]
        ):
            raise ValueError(
                "MNIST RGB input must contain three replicated grayscale channels."
            )
        images = images[:, :1]
    elif channels != 1:
        raise ValueError(f"MNIST input must have one or three channels, got {channels}.")

    images = images.clamp(0.0, 1.0)
    if tuple(images.shape[-2:]) != (32, 32):
        downsampling = images.shape[-2] > 32 or images.shape[-1] > 32
        images = F.interpolate(
            images,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
            antialias=downsampling,
        )
    return images


def _asset_resource(name):
    return resources.files(_ASSET_PACKAGE).joinpath(_ASSET_DIRECTORY, name)


def _resource_bytes(name):
    with _asset_resource(name).open("rb") as handle:
        return handle.read()


def load_mnist_jointae32_manifest():
    """Return the provenance manifest for the bundled metric asset."""
    with _asset_resource(_MANIFEST_ASSET).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MNISTJointAE32Metric(nn.Module):
    """Frozen JointAE metric with raw 128-dimensional features as its output."""

    def __init__(self, model, manifest):
        super().__init__()
        self.model = model
        self.manifest = manifest
        self.name = MNIST_JOINTAE_ARCHITECTURE_VERSION
        self.feature_layer = "raw_linear_128"
        self.feature_dim = MNIST_JOINTAE_FEATURE_DIM
        self.asset_sha256 = manifest["weight_asset"]["sha256"]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        """Keep every frozen JointAE component in evaluation mode."""
        super().train(False)
        self.model.eval()
        return self

    def canonicalize(self, images, *, image_range):
        return canonicalize_mnist32(images, image_range=image_range)

    def forward(self, images, image_range="auto"):
        """Return raw features without LayerNorm, SiLU, or L2 normalization."""
        images = self.canonicalize(images, image_range=image_range)
        return self.model.encode(images)

    def classify(self, images, image_range="auto"):
        """Return classifier logits from the same raw feature representation."""
        features = self(images, image_range=image_range)
        return self.model.classifier(features)

    def reconstruct(self, images, image_range="auto"):
        """Return reconstructions for qualification diagnostics."""
        images = self.canonicalize(images, image_range=image_range)
        reconstruction, _, _ = self.model(images)
        return reconstruction


def load_mnist_jointae32(device="cpu", *, asset_path=None, verify_hash=True):
    """Load the frozen bundled MNIST metric and return its raw-feature encoder."""
    manifest = load_mnist_jointae32_manifest()
    if asset_path is None:
        asset_bytes = _resource_bytes(_WEIGHT_ASSET)
    else:
        asset_bytes = Path(asset_path).expanduser().read_bytes()

    if verify_hash:
        actual_hash = hashlib.sha256(asset_bytes).hexdigest()
        expected_hash = manifest["weight_asset"]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                "MNIST JointAE checkpoint SHA-256 does not match its manifest."
            )

    checkpoint = torch.load(
        io.BytesIO(asset_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("architecture") != "MNISTJointAE32":
        raise ValueError("MNIST JointAE checkpoint has an incompatible architecture.")
    configuration = checkpoint.get("configuration", {})
    feature_dim = int(configuration.get("feature_dim", -1))
    if feature_dim != MNIST_JOINTAE_FEATURE_DIM:
        raise ValueError(
            f"MNIST JointAE feature dimension must be 128, got {feature_dim}."
        )

    model = _MNISTJointAE32(feature_dim=feature_dim)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metric = MNISTJointAE32Metric(model, manifest)
    return metric.to(torch.device(device))


def extract_mnist_jointae32_features(
    images,
    *,
    encoder=None,
    batch_size=256,
    image_range="auto",
    device="cpu",
    return_logits=False,
):
    """Extract reusable raw features, and optional logits, in fixed batches."""
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if encoder is None:
        encoder = load_mnist_jointae32(device=device)
    torch_device = torch.device(device)
    encoder.to(torch_device)
    encoder.eval()

    feature_blocks = []
    logit_blocks = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(torch_device)
            features = encoder(batch, image_range=image_range)
            feature_blocks.append(features.float().cpu())
            if return_logits:
                logits = encoder.model.classifier(features)
                logit_blocks.append(logits.float().cpu())

    features = torch.cat(feature_blocks)
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError("MNIST JointAE features contain non-finite values.")
    if not return_logits:
        return features
    logits = torch.cat(logit_blocks)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("MNIST JointAE logits contain non-finite values.")
    return features, logits


def compute_feature_statistics(features):
    """Compute float64 mean and covariance for one feature matrix."""
    features = torch.as_tensor(features).double()
    if features.dim() != 2 or features.shape[0] < 2:
        raise ValueError("Feature statistics require a [N, F] matrix with N >= 2.")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Feature statistics require finite values.")
    mean = features.mean(0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    return {
        "mean": mean,
        "covariance": covariance,
        "count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }


def _symmetric_psd_sqrt(matrix):
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T


def compute_frechet_distance(first, second):
    """Compute stable Fréchet distance from float64 feature statistics."""
    mean_first = torch.as_tensor(first["mean"]).double()
    mean_second = torch.as_tensor(second["mean"]).double()
    covariance_first = torch.as_tensor(first["covariance"]).double()
    covariance_second = torch.as_tensor(second["covariance"]).double()
    if mean_first.shape != mean_second.shape:
        raise ValueError("Fréchet means have different shapes.")
    if covariance_first.shape != covariance_second.shape:
        raise ValueError("Fréchet covariances have different shapes.")

    covariance_first = (covariance_first + covariance_first.T) * 0.5
    covariance_second = (covariance_second + covariance_second.T) * 0.5
    first_root = _symmetric_psd_sqrt(covariance_first)
    product_root = _symmetric_psd_sqrt(
        first_root @ covariance_second @ first_root
    )
    mean_term = (mean_first - mean_second).square().sum()
    covariance_term = torch.trace(
        covariance_first + covariance_second - 2.0 * product_root
    )
    distance = float((mean_term + covariance_term).item())
    return max(distance, 0.0)


def load_mnist_jointae32_reference_stats():
    """Load bundled global and per-class official-test reference statistics."""
    manifest = load_mnist_jointae32_manifest()
    asset_bytes = _resource_bytes(_REFERENCE_ASSET)
    actual_hash = hashlib.sha256(asset_bytes).hexdigest()
    expected_hash = manifest["reference_statistics_asset"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "MNIST JointAE reference-statistics SHA-256 does not match its manifest."
        )
    return torch.load(
        io.BytesIO(asset_bytes),
        map_location="cpu",
        weights_only=True,
    )


def compute_mnist_jointae32_fd(
    images=None,
    *,
    features=None,
    encoder=None,
    reference_stats=None,
    batch_size=256,
    image_range="auto",
    device="cpu",
):
    """Compute MNIST JointAE FD against bundled official-test statistics."""
    if (images is None) == (features is None):
        raise ValueError("Pass exactly one of images or features.")
    if features is None:
        features = extract_mnist_jointae32_features(
            images,
            encoder=encoder,
            batch_size=batch_size,
            image_range=image_range,
            device=device,
        )
    generated_stats = compute_feature_statistics(features)
    if reference_stats is None:
        reference_stats = load_mnist_jointae32_reference_stats()["global"]
    distance = compute_frechet_distance(generated_stats, reference_stats)
    manifest = load_mnist_jointae32_manifest()
    return {
        "metric": MNIST_JOINTAE_FD_NAME,
        "mnist_jointae_fd": distance,
        "n_samples": generated_stats["count"],
        "n_reference": int(reference_stats["count"]),
        "feature_dim": generated_stats["feature_dim"],
        "encoder_sha256": manifest["weight_asset"]["sha256"],
        "preprocessing": manifest["preprocessing"],
    }


def compute_mnist_jointae32_class_fd(
    features,
    predicted_labels,
    *,
    reference_stats=None,
):
    """Compute per-predicted-class JointAE FD and a present-class macro mean."""
    features = torch.as_tensor(features).float()
    predicted_labels = torch.as_tensor(predicted_labels).long().flatten()
    if len(features) != len(predicted_labels):
        raise ValueError("Features and predicted labels must have the same length.")
    if reference_stats is None:
        reference_stats = load_mnist_jointae32_reference_stats()

    per_class = []
    present_distances = []
    class_counts = torch.bincount(predicted_labels, minlength=10)
    for class_index in range(10):
        class_features = features[predicted_labels == class_index]
        count = int(class_features.shape[0])
        reference_count = int(reference_stats["per_class"][class_index]["count"])
        record = {
            "class": class_index,
            "generated_count": count,
            "reference_count": reference_count,
        }
        if count == 0:
            record.update({"status": "missing", "fd": None})
        elif count == 1:
            record.update({"status": "undersampled", "fd": None})
        else:
            generated_stats = compute_feature_statistics(class_features)
            distance = compute_frechet_distance(
                generated_stats,
                reference_stats["per_class"][class_index],
            )
            record.update({"status": "ok", "fd": distance})
            present_distances.append(distance)
        per_class.append(record)

    macro_fd = None
    if present_distances:
        macro_fd = sum(present_distances) / len(present_distances)
    manifest = load_mnist_jointae32_manifest()
    return {
        "metric": MNIST_JOINTAE_CLASS_FD_NAME,
        "macro_present_class_fd": macro_fd,
        "n_present_classes": len(present_distances),
        "n_samples": len(features),
        "feature_dim": int(features.shape[1]),
        "encoder_sha256": manifest["weight_asset"]["sha256"],
        "preprocessing": manifest["preprocessing"],
        "generated_class_counts": class_counts.tolist(),
        "per_class": per_class,
    }


def normalize_mnist_memorization_features(query_features, training_features):
    """Use training-bank standardization and row L2 normalization for neighbors."""
    query_features = torch.as_tensor(query_features).float()
    training_features = torch.as_tensor(training_features).float()
    if query_features.dim() != 2 or training_features.dim() != 2:
        raise ValueError("MNIST memorization features must be matrices.")
    if query_features.shape[1] != training_features.shape[1]:
        raise ValueError("Query and training features have different dimensions.")
    if training_features.shape[0] < 2:
        raise ValueError("MNIST memorization requires at least two training features.")

    mean = training_features.mean(0, keepdim=True)
    scale = training_features.std(0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    training_vectors = F.normalize((training_features - mean) / scale, dim=1)
    query_vectors = F.normalize((query_features - mean) / scale, dim=1)
    return query_vectors, training_vectors, mean, scale


def compute_mnist_feature_memorization(
    query_features,
    training_features,
    *,
    rho,
    query_batch_size=256,
    reference_batch_size=None,
):
    """Compute canonical JointAE-space nearest-neighbor memorization records."""
    query_vectors, training_vectors, mean, scale = (
        normalize_mnist_memorization_features(query_features, training_features)
    )
    stats = compute_nn_stats(
        query_vectors,
        training_vectors,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    stats.update(
        {
            "rho": float(rho),
            "f_mem": compute_f_mem(stats["r"], rho=rho),
            "feature_dim": int(training_vectors.shape[1]),
            "n_samples": int(query_vectors.shape[0]),
            "N": int(training_vectors.shape[0]),
            "training_feature_mean": mean.squeeze(0),
            "training_feature_scale": scale.squeeze(0),
            "metric": MNIST_JOINTAE_F_MEM_NAME,
        }
    )
    return stats


__all__ = [
    "MNIST_F_MEM_ALPHAS",
    "MNIST_JOINTAE_ARCHITECTURE_VERSION",
    "MNIST_JOINTAE_CLASS_FD_NAME",
    "MNIST_JOINTAE_FD_NAME",
    "MNIST_JOINTAE_F_MEM_NAME",
    "MNISTJointAE32Metric",
    "canonicalize_mnist32",
    "compute_feature_statistics",
    "compute_frechet_distance",
    "compute_mnist_feature_memorization",
    "compute_mnist_jointae32_class_fd",
    "compute_mnist_jointae32_fd",
    "extract_mnist_jointae32_features",
    "load_mnist_jointae32",
    "load_mnist_jointae32_manifest",
    "load_mnist_jointae32_reference_stats",
    "normalize_mnist_memorization_features",
]
