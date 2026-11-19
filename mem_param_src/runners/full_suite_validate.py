"""Validate one completed full metric-suite evaluation."""

import argparse
import csv
import math
from pathlib import Path

from PIL import Image

from mem_param_src.runners.artifacts import load_torch, read_json, write_json
from mem_param_src.runners.full_suite_retention import (
    validate_retained_checkpoint,
)

VALIDATION_SCHEMA_VERSION = 1
SAMPLE_GRID_STEPS = (0, 10_000, 100_000, 1_000_000)


def _validate_plots(root, protocol):
    manifest = read_json(root / "plots" / "plot_manifest.json")
    files = manifest.get("files", {})
    if not files:
        raise ValueError("The plot manifest contains no plots.")
    expected = {
        "plots/quality_trajectories.png",
        "plots/heldout_loss_trajectories.png",
        "plots/raw_range_diagnostics.png",
        "plots/runtime_diagnostics.png",
    }
    selected_steps = {
        int(path.name.removeprefix("step_"))
        for path in (root / "checkpoints").glob("step_*")
    }
    for step in SAMPLE_GRID_STEPS:
        if step in selected_steps:
            expected.add(f"plots/samples_step_{step:06d}.png")
    for space in protocol["memorization_spaces"]:
        plot_space = "mnist_jointae32_v1" if space == "mnist_jointae32" else space
        expected.add(f"plots/memorization_{plot_space}.png")
        expected.add(f"plots/nearest_neighbors_{space}.png")
    if "f_mem_raw_pixels" in protocol["metrics"]:
        expected.add("plots/memorization_raw_pixels.png")
        expected.add("plots/nearest_neighbors_raw_pixels.png")
    missing = expected - set(files)
    if missing:
        raise ValueError(f"Expected plots are missing: {sorted(missing)}")
    for name in sorted(files):
        path = root / name
        with Image.open(path) as image:
            image.verify()
    return sorted(files)


def _validate_metric_rows(root):
    path = root / "combined" / "checkpoint_metrics.csv"
    row_count = 0
    f_mem_count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric value in row {row_count}.")
            if row["metric"].startswith("f_mem"):
                f_mem_count += 1
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"f_mem is outside [0, 1] in row {row_count}.")
    if row_count == 0 or f_mem_count == 0:
        raise ValueError("The combined metric table is incomplete.")
    return row_count, f_mem_count


def _validate_reuse(metrics, protocol):
    if not metrics.get("sample_reused"):
        raise ValueError("Repeated invocation did not reuse generated samples.")
    if protocol["dataset_family"] == "mnist":
        if not metrics.get("features_reused"):
            raise ValueError("Repeated invocation did not reuse JointAE features.")
        quality = metrics["mnist_fd_jointae32_v1"]
        if not quality.get("statistics_reused"):
            raise ValueError("Repeated invocation did not reuse JointAE statistics.")
        if not quality.get("distance_reused"):
            raise ValueError("Repeated invocation did not reuse JointAE distance.")
        if not metrics["f_mem_raw_pixels"].get("neighbors_reused"):
            raise ValueError("Repeated invocation did not reuse raw neighbors.")
        learned = metrics["f_mem_mnist_jointae32_v1"]
        if not learned.get("neighbors_reused"):
            raise ValueError("Repeated invocation did not reuse JointAE neighbors.")
        if not metrics["heldout_loss"].get("reused"):
            raise ValueError("Repeated invocation did not reuse held-out losses.")
        return
    for name, label in (
        ("inception_fid", "Inception"),
        ("dinov2_fd", "DINOv2"),
    ):
        quality = metrics[name]
        if not quality.get("features_reused"):
            raise ValueError(f"Repeated invocation did not reuse {label} features.")
        if not quality.get("statistics_reused"):
            raise ValueError(f"Repeated invocation did not reuse {label} statistics.")
        if not quality.get("distance_reused"):
            raise ValueError(f"Repeated invocation did not reuse {label} distance.")
    if not metrics["f_mem_raw_pixels"].get("neighbors_reused"):
        raise ValueError("Repeated invocation did not reuse raw neighbors.")
    if not metrics["heldout_loss"].get("reused"):
        raise ValueError("Repeated invocation did not reuse held-out losses.")
    for space in protocol["memorization_spaces"]:
        learned = metrics[f"f_mem_{space}"]
        if not learned.get("features_reused"):
            raise ValueError(f"Repeated invocation did not reuse {space} features.")
        if not learned.get("neighbors_reused"):
            raise ValueError(f"Repeated invocation did not reuse {space} neighbors.")


def validate_full_suite(
    root,
    *,
    expected_checkpoints,
    sample_count,
    evaluation_tier="development",
):
    """Validate one full-suite root and write its acceptance record."""
    root = Path(root).expanduser().resolve()
    manifest = read_json(root / "eval_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Evaluation manifest is not complete.")
    if evaluation_tier not in {"development", "paper"}:
        raise ValueError("evaluation_tier must be 'development' or 'paper'.")
    if evaluation_tier == "paper":
        if manifest.get("evaluation_tier") != "paper":
            raise ValueError("Evaluation manifest is not marked paper-tier.")
        if manifest.get("development_only") is not False:
            raise ValueError("Paper evaluation is marked development-only.")
        if manifest.get("publication_ready") is not True:
            raise ValueError("Paper evaluation is not marked publication-ready.")
    else:
        if manifest.get("development_only") is not True:
            raise ValueError("Evaluation is not marked development_only.")
        if manifest.get("publication_ready") is not False:
            raise ValueError("Evaluation is incorrectly marked publication-ready.")
    if int(manifest["sample_count"]) != int(sample_count):
        raise ValueError("Evaluation sample count differs from the request.")
    protocol = manifest["protocol"]
    checkpoint_roots = sorted((root / "checkpoints").glob("step_*"))
    if len(checkpoint_roots) != int(expected_checkpoints):
        raise ValueError("Evaluation checkpoint count differs from the request.")

    streaming_retention = bool(manifest.get("streaming_retention", False))
    selected_steps = {int(path.name.removeprefix("step_")) for path in checkpoint_roots}
    protected_steps = {
        int(step) for step in SAMPLE_GRID_STEPS if int(step) in selected_steps
    }
    protected_steps.add(max(selected_steps))
    expected_shape = (
        int(sample_count),
        int(protocol.get("image_channels", 1)),
        int(protocol["image_size"]),
        int(protocol["image_size"]),
    )
    retained_checkpoint_count = 0
    for checkpoint_root in checkpoint_roots:
        if not (checkpoint_root / "COMPLETE.json").is_file():
            raise ValueError(f"Incomplete checkpoint: {checkpoint_root.name}")
        step = int(checkpoint_root.name.removeprefix("step_"))
        if streaming_retention:
            retention = validate_retained_checkpoint(checkpoint_root)
            if bool(retention["retain_samples"]) != (step in protected_steps):
                raise ValueError(
                    f"Checkpoint sample-retention policy differs at step {step}."
                )
            retained_checkpoint_count += 1
        else:
            metrics = read_json(checkpoint_root / "metrics.json")
            _validate_reuse(metrics, protocol)
        if not streaming_retention or step in protected_steps:
            samples = load_torch(checkpoint_root / "samples.pt")
            if tuple(samples.shape) != expected_shape:
                raise ValueError(
                    f"Generated tensor shape differs at {checkpoint_root.name}."
                )

    row_count, f_mem_count = _validate_metric_rows(root)
    plots = _validate_plots(root, protocol)
    record = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "accepted": True,
        "streaming_retention": streaming_retention,
        "retained_checkpoint_count": retained_checkpoint_count,
        "retention_mode": "checkpoint_identity_after_completion"
        if streaming_retention
        else "two_pass_cache_reuse",
        "evaluation_tier": evaluation_tier,
        "publication_ready": evaluation_tier == "paper",
        "checkpoint_count": len(checkpoint_roots),
        "sample_count": int(sample_count),
        "metric_row_count": row_count,
        "f_mem_row_count": f_mem_count,
        "plot_count": len(plots),
        "plots": plots,
    }
    write_json(root / "ACCEPTED.json", record)
    return record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--expected-checkpoints", required=True, type=int)
    parser.add_argument("--sample-count", default=512, type=int)
    parser.add_argument(
        "--evaluation-tier",
        choices=("development", "paper"),
        default="development",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_full_suite(
        args.evaluation_root,
        expected_checkpoints=args.expected_checkpoints,
        sample_count=args.sample_count,
        evaluation_tier=args.evaluation_tier,
    )
    return 0


if __name__ == "__main__":
    main()
