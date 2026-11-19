"""Build and verify canonical fixed-noise artifacts for checkpoint evaluation."""

import argparse
from pathlib import Path

import torch

from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    save_torch,
    tensor_sha256,
    write_json,
)

EVALUATION_NOISE_SCHEMA_VERSION = 1


def _noise_tensor(payload, path):
    """Return one batched CPU noise tensor from a saved artifact."""
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict) and torch.is_tensor(payload.get("tensor")):
        tensor = payload["tensor"]
    else:
        raise ValueError(f"Evaluation noise is not a tensor: {path}")
    if tensor.dim() < 2 or len(tensor) < 1:
        raise ValueError(f"Evaluation noise must be a non-empty batch: {path}")
    return tensor.detach().cpu().contiguous()


def build_canonical_noise(source_noise_paths, output_artifact, *, count, seed):
    """Build one fixed CPU Gaussian stream after verifying all saved prefixes."""
    output_artifact = Path(output_artifact).expanduser().resolve()
    manifest_path = output_artifact.with_suffix(".json")
    if output_artifact.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to replace canonical evaluation noise: {output_artifact}"
        )

    source_paths = [Path(path).expanduser().resolve() for path in source_noise_paths]
    if not source_paths:
        raise ValueError("Pass at least one saved source-noise artifact.")
    sources = []
    reference = None
    for path in source_paths:
        tensor = _noise_tensor(load_torch(path), path)
        if reference is None:
            reference = tensor
        elif tensor.dtype != reference.dtype or tensor.shape != reference.shape:
            raise ValueError("Saved evaluation-noise prefixes have different shapes.")
        elif not torch.equal(tensor, reference):
            raise ValueError("Saved evaluation-noise prefixes are not identical.")
        sources.append(
            {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "tensor_sha256": tensor_sha256(tensor),
                "shape": list(tensor.shape),
            }
        )

    count = int(count)
    seed = int(seed)
    if count < len(reference):
        raise ValueError("Canonical noise count is smaller than the saved prefix.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reproduced = torch.randn(
        reference.shape,
        generator=generator,
        dtype=reference.dtype,
    )
    if not torch.equal(reproduced, reference):
        raise ValueError(
            "The canonical seed does not reproduce the saved evaluation-noise prefix."
        )
    tail_shape = (count - len(reference), *reference.shape[1:])
    tail = torch.randn(tail_shape, generator=generator, dtype=reference.dtype)
    tensor = torch.cat((reference, tail), dim=0).contiguous()
    generator_state = generator.get_state().contiguous()
    payload = {
        "schema_version": EVALUATION_NOISE_SCHEMA_VERSION,
        "tensor": tensor,
        "tensor_sha256": tensor_sha256(tensor),
        "seed": seed,
        "generator": "torch.Generator(device='cpu').manual_seed",
        "generator_state_after_bank": generator_state,
        "generator_state_sha256": tensor_sha256(generator_state),
        "torch_version": torch.__version__,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "source_prefix_count": len(reference),
        "source_prefixes": sources,
    }
    save_torch(output_artifact, payload)
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"tensor", "generator_state_after_bank"}
    }
    manifest.update(
        {
            "artifact": str(output_artifact),
            "artifact_sha256": file_sha256(output_artifact),
        }
    )
    write_json(manifest_path, manifest)
    return output_artifact, manifest_path


def load_canonical_noise(path, *, sample_count, expected_shape=None):
    """Load one verified canonical bank and return its requested prefix."""
    path = Path(path).expanduser().resolve()
    manifest_path = path.with_suffix(".json")
    payload = load_torch(path)
    manifest = read_json(manifest_path)
    if int(payload.get("schema_version", -1)) != EVALUATION_NOISE_SCHEMA_VERSION:
        raise ValueError("Canonical evaluation-noise schema version is unsupported.")
    if manifest.get("artifact_sha256") != file_sha256(path):
        raise ValueError("Canonical evaluation-noise artifact hash does not match.")
    tensor = _noise_tensor(payload, path)
    if payload.get("tensor_sha256") != tensor_sha256(tensor):
        raise ValueError("Canonical evaluation-noise tensor hash does not match.")
    state = payload.get("generator_state_after_bank")
    if not torch.is_tensor(state) or payload.get(
        "generator_state_sha256"
    ) != tensor_sha256(state):
        raise ValueError("Canonical evaluation-noise generator state is invalid.")
    for key in (
        "schema_version",
        "tensor_sha256",
        "seed",
        "generator",
        "generator_state_sha256",
        "torch_version",
        "dtype",
        "shape",
        "source_prefix_count",
        "source_prefixes",
    ):
        if manifest.get(key) != payload.get(key):
            raise ValueError(f"Canonical evaluation-noise manifest differs for {key}.")
    sample_count = int(sample_count)
    if sample_count < 1 or len(tensor) < sample_count:
        raise ValueError("Canonical noise has fewer rows than the requested count.")
    if expected_shape is not None and tuple(tensor.shape[1:]) != tuple(expected_shape):
        raise ValueError("Canonical noise shape differs from the saved run shape.")
    prefix = tensor[:sample_count].clone()
    identity = {
        "path": str(path),
        "file_sha256": manifest["artifact_sha256"],
        "tensor_sha256": payload["tensor_sha256"],
        "prefix_sha256": tensor_sha256(prefix),
        "seed": int(payload["seed"]),
        "sample_count": sample_count,
        "shape": list(prefix.shape),
    }
    return prefix, identity


def build_parser():
    """Build the canonical evaluation-noise command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-noise", action="append", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2003)
    return parser


def main(argv=None):
    """Build one canonical evaluation-noise artifact."""
    args = build_parser().parse_args(argv)
    build_canonical_noise(
        args.source_noise,
        args.output_artifact,
        count=args.count,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    main()
