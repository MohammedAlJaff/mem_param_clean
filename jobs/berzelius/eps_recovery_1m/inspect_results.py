"""Print raw sample measurements and matching-data hashes; create no files."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch


def read_json(path, default):
    """Use a default when a job has not yet written the requested record."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_tensor(path):
    """Read a saved tensor on CPU, without loading model or dataset code."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value) or value.numel() == 0:
        raise ValueError(f"Expected a nonempty tensor: {path}")
    return value.detach().contiguous()


def tensor_sha256(value):
    """Match the runner hash: dtype, shape, then exact contiguous tensor bytes."""
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(metadata)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inspect_run(run_dir, writer):
    """Print one row per saved sample batch; report unfinished observations."""
    record = read_json(run_dir / "run.json", {})
    manifest = read_json(run_dir / "data_manifest.json", {})
    train = manifest.get("partitions", {}).get("train", {}).get("tensor", {})
    metadata = {
        "run": run_dir.name, "state": record.get("state", "missing run.json"),
        "final_step": record.get("final_step"),
        "recorded_bundle_sha256": record.get("data_bundle_sha256")
        or manifest.get("bundle", {}).get("sha256"),
        "recorded_train_tensor_sha256": train.get("sha256"),
    }
    noise_path = run_dir / "z_init.pt"
    metadata["z_init_tensor_sha256"] = (
        tensor_sha256(load_tensor(noise_path)) if noise_path.is_file() else "missing"
    )
    print("# " + json.dumps(metadata, sort_keys=True))
    if record.get("state") not in ("trained", "evaluated"):
        print(f"# {run_dir.name}: job incomplete or failed; {record.get('error', '')}")
    history = read_json(run_dir / "train_history.json", [])
    losses = {int(row["step"]): row["loss"] for row in history}
    if not history:
        print(f"# {run_dir.name}: no saved training loss records yet")
    monitor = run_dir / "monitoring"
    sample_dir = monitor / "generated_samples"
    sample_paths = sorted(sample_dir.glob("step_*.pt"))
    if not sample_paths:
        print(f"# {run_dir.name}: no saved raw sample batches yet")
    for checkpoint in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        if not (sample_dir / checkpoint.name).is_file():
            print(f"# {run_dir.name}: raw samples missing for {checkpoint.stem}")
    for path in sample_paths:
        step = int(path.stem.removeprefix("step_"))
        samples = load_tensor(path).double()
        metrics = read_json(monitor / f"f_mem_step_{step:06d}.json", {})
        writer.writerow([
            run_dir.name, step, losses.get(step, ""),
            bool(torch.isfinite(samples).all()), samples.min().item(),
            samples.max().item(), samples.square().mean().sqrt().item(),
            (samples.abs() > 1).double().mean().item(), metrics.get("f_mem", ""),
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Inspect raw one-million-step samples on CPU. Prints CSV with # metadata "
        "and status comments. Hashes check matched inputs; metrics do not prove convergence."
    )
    parser.add_argument("batch_dir", type=Path, help="Batch directory containing runs/.")
    args = parser.parse_args()
    runs_dir = args.batch_dir.expanduser() / "runs"
    if not runs_dir.is_dir():
        parser.error(f"No runs directory yet: {runs_dir}")
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not run_dirs:
        print(f"# No run directories yet: {runs_dir}")
    writer = csv.writer(sys.stdout)
    writer.writerow(["run", "step", "loss", "finite", "min", "max", "rms",
                     "fraction_abs_gt_1", "projected_f_mem"])
    for run_dir in run_dirs:
        try:
            inspect_run(run_dir, writer)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"# {run_dir.name}: inspection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
