"""Train and qualify the frozen MNIST JointAE-32 metric encoder."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


class MNISTTensorDataset(Dataset):
    """Expose canonical torchvision MNIST tensors without PIL conversion."""

    def __init__(self, dataset):
        self.data = dataset.data
        self.targets = dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].unsqueeze(0).float().div(255.0)
        return image, self.targets[index]


class ResBlock(nn.Module):
    """Residual convolution block from the validated notebook."""

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


class MNISTJointAE32(nn.Module):
    """Joint autoencoder/classifier with a raw 128-D metric feature."""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            ResBlock(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            ResBlock(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            ResBlock(256),
        )
        self.feature_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.to_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )
        self.from_feature = nn.Linear(feature_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            ResBlock(64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images):
        """Return the raw linear feature projection used by MNIST FD."""
        hidden = self.feature_pool(self.encoder(images))
        return self.to_feature(hidden)

    def forward(self, images):
        features = self.encode(images)
        logits = self.classifier(features)
        reconstruction = self.from_feature(features).view(-1, 256, 4, 4)
        reconstruction = self.decoder(reconstruction)
        return reconstruction, logits, features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2003)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=55000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--min-validation-accuracy", type=float, default=0.99)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonicalize_mnist32(images):
    """Apply the repository's float bilinear 28-to-32 transform."""
    if tuple(images.shape[-2:]) == (32, 32):
        return images
    return F.interpolate(
        images,
        size=(32, 32),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_loader(dataset, indices, batch_size, workers, shuffle, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def run_epoch(
    model,
    loader,
    device,
    classification_weight,
    optimizer=None,
    max_batches=0,
):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "classification_loss": 0.0,
        "correct": 0,
        "count": 0,
    }
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            images = canonicalize_mnist32(images.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, logits, _ = model(images)
            reconstruction_loss = F.smooth_l1_loss(
                reconstruction,
                images,
                beta=0.1,
            )
            classification_loss = F.cross_entropy(logits, labels)
            loss = reconstruction_loss + classification_weight * classification_loss
            if training:
                loss.backward()
                optimizer.step()
            batch_size = len(images)
            totals["loss"] += float(loss.detach()) * batch_size
            totals["reconstruction_loss"] += (
                float(reconstruction_loss.detach()) * batch_size
            )
            totals["classification_loss"] += (
                float(classification_loss.detach()) * batch_size
            )
            totals["correct"] += int((logits.argmax(1) == labels).sum())
            totals["count"] += batch_size
    if totals["count"] == 0:
        raise RuntimeError("No images were processed in this epoch.")
    return {
        "loss": totals["loss"] / totals["count"],
        "reconstruction_loss": totals["reconstruction_loss"] / totals["count"],
        "classification_loss": totals["classification_loss"] / totals["count"],
        "accuracy": totals["correct"] / totals["count"],
        "n_samples": totals["count"],
    }


def is_better_candidate(candidate, incumbent, minimum_accuracy):
    if incumbent is None:
        return True
    candidate_qualified = candidate["accuracy"] >= minimum_accuracy
    incumbent_qualified = incumbent["accuracy"] >= minimum_accuracy
    if candidate_qualified != incumbent_qualified:
        return candidate_qualified
    if candidate_qualified:
        candidate_key = (
            candidate["reconstruction_loss"],
            -candidate["accuracy"],
        )
        incumbent_key = (
            incumbent["reconstruction_loss"],
            -incumbent["accuracy"],
        )
        return candidate_key < incumbent_key
    candidate_key = (-candidate["accuracy"], candidate["reconstruction_loss"])
    incumbent_key = (-incumbent["accuracy"], incumbent["reconstruction_loss"])
    return candidate_key < incumbent_key


@torch.no_grad()
def collect_features(model, loader, device):
    model.eval()
    feature_blocks = []
    label_blocks = []
    logit_blocks = []
    for images, labels in loader:
        images = canonicalize_mnist32(images.to(device, non_blocking=True))
        features = model.encode(images)
        logits = model.classifier(features)
        feature_blocks.append(features.cpu())
        label_blocks.append(labels.clone())
        logit_blocks.append(logits.cpu())
    return (
        torch.cat(feature_blocks),
        torch.cat(label_blocks),
        torch.cat(logit_blocks),
    )


def covariance(features):
    features = features.double()
    centered = features - features.mean(0)
    return centered.T @ centered / (len(features) - 1)


def raw_dataset_hashes(data_root):
    raw_root = data_root / "MNIST" / "raw"
    names = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    hashes = {}
    for name in names:
        path = raw_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256_file(path)
    return hashes


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.feature_dim < 1:
        raise ValueError("epochs, batch size, and feature dimension must be positive.")
    if args.train_size < 2 or args.valid_size < 2:
        raise ValueError("Training and validation splits must each contain at least two images.")
    if args.train_size + args.valid_size > 60000:
        raise ValueError("MNIST train and validation sizes exceed the 60,000-image split.")
    if not 0.0 <= args.min_validation_accuracy <= 1.0:
        raise ValueError("Minimum validation accuracy must lie in [0, 1].")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no allocated GPU is visible.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    started_at = utc_now()
    print(f"started: {started_at}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"output: {args.output_dir}", flush=True)

    train_source = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=False,
    )
    test_source = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=False,
    )
    train_dataset = MNISTTensorDataset(train_source)
    test_dataset = MNISTTensorDataset(test_source)
    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_dataset), generator=split_generator)
    train_indices = permutation[: args.train_size].tolist()
    valid_indices = permutation[
        args.train_size : args.train_size + args.valid_size
    ].tolist()
    test_count = len(test_dataset) if args.test_limit == 0 else args.test_limit
    if test_count < 2 or test_count > len(test_dataset):
        raise ValueError("test-limit must be zero or lie in [2, 10000].")
    test_indices = list(range(test_count))

    train_loader = make_loader(
        train_dataset,
        train_indices,
        args.batch_size,
        args.workers,
        True,
        args.seed + 1,
        device,
    )
    valid_loader = make_loader(
        train_dataset,
        valid_indices,
        args.batch_size,
        args.workers,
        False,
        args.seed + 2,
        device,
    )
    test_loader = make_loader(
        test_dataset,
        test_indices,
        args.batch_size,
        args.workers,
        False,
        args.seed + 3,
        device,
    )

    model = MNISTJointAE32(args.feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    best_epoch = None
    best_validation = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.classification_weight,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        validation_metrics = run_epoch(
            model,
            valid_loader,
            device,
            args.classification_weight,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if is_better_candidate(
            validation_metrics,
            best_validation,
            args.min_validation_accuracy,
        ):
            best_epoch = epoch
            best_validation = dict(validation_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = run_epoch(
        model,
        test_loader,
        device,
        args.classification_weight,
    )
    print(f"selected epoch: {best_epoch}", flush=True)
    print(f"test: {json.dumps(test_metrics, sort_keys=True)}", flush=True)

    train_feature_loader = make_loader(
        train_dataset,
        train_indices,
        args.batch_size,
        args.workers,
        False,
        args.seed + 4,
        device,
    )
    train_features, _, _ = collect_features(model, train_feature_loader, device)
    feature_mean = train_features.mean(0, keepdim=True)
    feature_scale = train_features.std(0, keepdim=True).clamp_min(1e-6)
    test_features, test_labels, test_logits = collect_features(
        model,
        test_loader,
        device,
    )

    configuration = {
        "architecture": "MNISTJointAE32",
        "batch_size": args.batch_size,
        "classification_weight": args.classification_weight,
        "epochs": args.epochs,
        "feature_dim": args.feature_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_batches": args.max_train_batches,
        "minimum_validation_accuracy": args.min_validation_accuracy,
        "seed": args.seed,
        "test_size": test_count,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
    }
    split_path = args.output_dir / "split_indices.pt"
    reference_path = args.output_dir / "mnist_test_reference_features.pt"
    checkpoint_path = args.output_dir / "mnist_jointae32_v1.pt"
    history_path = args.output_dir / "history.json"
    manifest_path = args.output_dir / "manifest.json"
    atomic_torch_save(
        {
            "seed": args.seed,
            "train_indices": torch.tensor(train_indices, dtype=torch.int64),
            "valid_indices": torch.tensor(valid_indices, dtype=torch.int64),
            "test_indices": torch.tensor(test_indices, dtype=torch.int64),
        },
        split_path,
    )
    atomic_torch_save(
        {
            "features": test_features,
            "labels": test_labels,
            "logits": test_logits,
            "mean": test_features.double().mean(0),
            "covariance": covariance(test_features),
            "source_split": "torchvision_mnist:test",
            "preprocessing": "uint8/255 then bilinear 28_to_32 align_corners_false antialias_false",
        },
        reference_path,
    )
    atomic_torch_save(
        {
            "architecture": "MNISTJointAE32",
            "state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "configuration": configuration,
            "best_epoch": best_epoch,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    atomic_json(history, history_path)

    qualified = best_validation["accuracy"] >= args.min_validation_accuracy
    completed_at = utc_now()
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (checkpoint_path, reference_path, split_path, history_path)
    }
    manifest = {
        "artifact_hashes": artifact_hashes,
        "completed_at": completed_at,
        "configuration": configuration,
        "dataset": {
            "name": "MNIST",
            "raw_sha256": raw_dataset_hashes(args.data_root),
            "test_split_role": "frozen real reference; never used for weight optimization or epoch selection",
            "train_split_role": "55k encoder training plus 5k encoder validation",
        },
        "environment": {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "metric_names": {
            "class_aware": "mnist_cafd_jointae32_v1",
            "frechet": "mnist_fd_jointae32_v1",
            "memorization": "f_mem_mnist_jointae32_v1",
        },
        "preprocessing": {
            "channels": 1,
            "input_range": [0.0, 1.0],
            "interpolation": "bilinear",
            "align_corners": False,
            "antialias_28_to_32": False,
            "canonical_resolution": 32,
        },
        "qualification": {
            "best_epoch": best_epoch,
            "minimum_validation_accuracy": args.min_validation_accuracy,
            "passed": qualified,
            "test_metrics": test_metrics,
            "validation_metrics": best_validation,
        },
        "started_at": started_at,
        "status": "qualified" if qualified else "failed_qualification",
    }
    atomic_json(manifest, manifest_path)
    if qualified:
        (args.output_dir / "TRAINING_COMPLETE").write_text(
            completed_at + "\n",
            encoding="utf-8",
        )
        print("MNIST JointAE-32 asset passed qualification.", flush=True)
        return
    (args.output_dir / "FAILED_QUALIFICATION").write_text(
        json.dumps(manifest["qualification"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise RuntimeError(
        "Best validation accuracy did not satisfy the frozen qualification threshold."
    )


if __name__ == "__main__":
    main()
