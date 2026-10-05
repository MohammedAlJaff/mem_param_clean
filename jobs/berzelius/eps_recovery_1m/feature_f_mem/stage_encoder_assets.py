#!/usr/bin/env python3
"""Stage and hash every learned feature encoder asset before Slurm submission."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DINO_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_REPOSITORY = "https://github.com/facebookresearch/dinov2.git"
DINO_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_pretrain.pth"
)
SSCD_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/"
    "sscd_disc_mixup.torchscript.pt"
)
ALEXNET_FILENAME = "alexnet-owt-7be5be79.pth"

INCEPTION_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/"
    "weights-inception-2015-12-05-6726825d.pth"
)

def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(path)


def stage_dinov2(root):
    repo = root / "dinov2"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", DINO_REPOSITORY, str(repo)],
            check=True,
        )
    subprocess.run(["git", "-C", str(repo), "fetch", "origin", DINO_COMMIT], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", DINO_COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if actual != DINO_COMMIT:
        raise RuntimeError(f"DINOv2 checkout is {actual}, expected {DINO_COMMIT}.")
    return repo


def stage_lpips(root):
    import lpips

    torch_home = root / "torch_home"
    os.environ["TORCH_HOME"] = str(torch_home)
    lpips.LPIPS(net="alex", version="0.1", verbose=False)
    backbone = torch_home / "hub" / "checkpoints" / ALEXNET_FILENAME
    if not backbone.is_file():
        raise FileNotFoundError(f"LPIPS did not stage its AlexNet backbone: {backbone}")
    package_weight = Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / "alex.pth"
    calibration = root / "lpips" / "alex_v0.1.pth"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_weight, calibration)
    return torch_home, backbone, calibration


def relative_record(root, path, **metadata):
    path = Path(path).resolve()
    record = {
        "path": str(path.relative_to(root.resolve())),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    record.update(metadata)
    return record


def stage(output_root):
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if int(existing.get("schema_version", -1)) == 3:
            verify(manifest_path)
            print(manifest_path)
            return manifest_path

    dino_repo = stage_dinov2(output_root)
    dino_weights = output_root / "weights" / "dinov2_vits14_pretrain.pth"
    sscd_weights = output_root / "weights" / "sscd_disc_mixup.torchscript.pt"
    download(DINO_WEIGHTS_URL, dino_weights)
    download(SSCD_WEIGHTS_URL, sscd_weights)
    torch_home, alexnet_weights, lpips_weights = stage_lpips(output_root)
    dino_cache = torch_home / "hub" / "checkpoints" / dino_weights.name
    dino_cache.parent.mkdir(parents=True, exist_ok=True)
    if (
        not dino_cache.is_file()
        or sha256(dino_cache) != sha256(dino_weights)
    ):
        shutil.copy2(dino_weights, dino_cache)

    inception_weights = output_root / "weights" / "inception_v3_compat.pth"
    download(INCEPTION_WEIGHTS_URL, inception_weights)
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(output_root),
        "encoders": {
            "sscd": relative_record(
                output_root,
                sscd_weights,
                source_url=SSCD_WEIGHTS_URL,
                architecture="sscd_disc_mixup",
            ),
            "lpips_calibration": relative_record(
                output_root,
                lpips_weights,
                package_version=importlib.metadata.version("lpips"),
                network="alex",
                version="0.1",
            ),
            "lpips_alexnet": relative_record(
                output_root,
                alexnet_weights,
                torchvision_cache=True,
            ),
            "fid_inception_v3": relative_record(
                output_root,
                inception_weights,
                source_url=INCEPTION_WEIGHTS_URL,
                architecture="inception-v3-compat",
            ),
            "dinov2_weights": relative_record(
                output_root,
                dino_cache,
                source_url=DINO_WEIGHTS_URL,
                architecture="dinov2_vits14",
            ),
        },
        "dinov2_source": {
            "path": str(dino_repo.relative_to(output_root)),
            "repository": DINO_REPOSITORY,
            "commit": DINO_COMMIT,
        },
        "torch_home": str(torch_home.relative_to(output_root)),
        "package_versions": {
            "torch": importlib.metadata.version("torch"),
            "torchvision": importlib.metadata.version("torchvision"),
            "timm": importlib.metadata.version("timm"),
            "lpips": importlib.metadata.version("lpips"),
            "torch_fidelity": "0.4.0-beta-vendored",
        },
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    verify(manifest_path)
    print(manifest_path)
    return manifest_path


def verify(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    if int(manifest.get("schema_version", -1)) != 3:
        raise RuntimeError("Unsupported encoder-asset manifest schema.")
    for name, record in manifest["encoders"].items():
        path = root / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Encoder asset failed SHA-256 verification: {name}")
    repo = root / manifest["dinov2_source"]["path"]
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    if actual != manifest["dinov2_source"]["commit"]:
        raise RuntimeError("DINOv2 source checkout no longer matches its manifest.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--output-root", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("choose stage or verify")
    if args.command == "stage":
        stage(args.output_root)
    else:
        verify(args.manifest)
        print(Path(args.manifest).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
