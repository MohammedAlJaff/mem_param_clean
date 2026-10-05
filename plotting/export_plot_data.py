"""Export accepted scalar metrics into clean, Git-diffable experiment folders."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

EVALUATION_ROOTS = (
    Path("outputs/evals/celeba32_streaming_full_10000/20261004T110904Z_1979099"),
    Path("outputs/evals/all_1m_full_10000/20261004T005702Z_3271331"),
    Path("outputs/evals/celeba32_vv_weighting_full_10000/20261003T134718Z_2109890"),
    Path(
        "outputs/evals/celeba32_dit_seed1_streaming_full_10000/20261005T070254Z_4059124"
    ),
)
DEFAULT_OUTPUT_ROOT = Path("plot_data")
GROUP_TO_EXPERIMENT = {
    "celeba32_unet_a": "celeba32_unet_3x3",
    "celeba32_unet_b": "celeba32_unet_3x3",
    "celeba32_unet_c": "celeba32_unet_3x3",
    "dit_seed0": "celeba32_dit_3x3",
    "dit_seed1": "celeba32_dit_3x3",
    "celeba64_rgb": "celeba64_rgb_diagonal",
    "gate_mnist32": "mnist32_diagonal",
    "mnist32_rest": "mnist32_diagonal",
    "mnist64": "mnist64_diagonal",
    "weighting": "celeba32_vv_weighting",
}
RUN_FIELDS = (
    "run_id",
    "evaluation_group",
    "result_name",
    "source_run_name",
    "dataset",
    "model_arch",
    "n_train",
    "seed",
    "prediction_space",
    "loss_space",
    "loss_weighting",
    "training_steps",
    "learning_rate",
    "batch_size",
    "effective_batch_size",
    "image_channels",
    "image_size",
    "evaluation_checkpoint_count",
    "evaluation_sample_count",
    "evaluation_tier",
    "source_state",
    "source_final_step",
    "wandb_project",
    "wandb_group",
    "wandb_name",
    "source_run_path",
    "evaluation_result_path",
    "source_config_sha256",
    "source_data_bundle_sha256",
)
TRAIN_FIELDS = ("run_id", "step", "metric", "value")
EVALUATION_FIELDS = (
    "run_id",
    "step",
    "checkpoint",
    "metric",
    "space",
    "value",
    "sample_count",
    "reference_count",
    "loss_count",
    "alpha",
    "rho",
)
DIAGNOSTIC_FIELDS = (
    "run_id",
    "step",
    "category",
    "metric",
    "value",
    "alpha",
    "rho",
)
MONITOR_METRICS = (
    "f_mem",
    "mean_d1",
    "mean_d2",
    "mean_r",
    "median_d1",
    "median_d2",
    "median_r",
    "std_r",
)


def _read_json(path):
    """Read one UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(mapping, *keys, default=""):
    """Read a nested value without changing the source."""
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _sha256(path):
    """Hash one exported file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _accepted_results(evaluation_root):
    """Return accepted result directories without scanning checkpoint trees."""
    evaluation_root = Path(evaluation_root)
    paths = set(evaluation_root.glob("results/*/*/ACCEPTED.json"))
    paths.update(evaluation_root.glob("*/ACCEPTED.json"))
    return sorted(path.parent for path in paths)


def _relative_result(evaluation_root, result_root):
    """Return group and result names from either supported result layout."""
    relative = result_root.relative_to(evaluation_root)
    parts = relative.parts
    if parts and parts[0] == "results":
        parts = parts[1:]
    if len(parts) == 1:
        return "weighting", parts[0]
    return parts[-2], parts[-1]


def _experiment_id(group):
    """Map one evaluation group to a clean experiment directory."""
    try:
        return GROUP_TO_EXPERIMENT[group]
    except KeyError as error:
        raise ValueError(f"No plotting experiment mapping for group {group}") from error


def _identity(evaluation_root, result_root, config):
    """Build one short run identifier inside its experiment."""
    group, result_name = _relative_result(evaluation_root, result_root)
    return {
        "experiment": _experiment_id(group),
        "run_id": f"{group}/{result_name}",
        "evaluation_group": group,
        "result_name": result_name,
        "source_run_name": _nested(config, "outputs", "run_name", default=result_name),
    }


def _run_row(identity, manifest, config, run_record, source_run, result_root):
    """Build one complete run-index row."""
    row = {
        "run_id": identity["run_id"],
        "evaluation_group": identity["evaluation_group"],
        "result_name": identity["result_name"],
        "source_run_name": identity["source_run_name"],
        "dataset": _nested(config, "data", "dataset"),
        "model_arch": _nested(config, "network", "arch"),
        "n_train": _nested(config, "data", "n_train"),
        "seed": _nested(config, "training", "seed"),
        "prediction_space": _nested(config, "framework", "prediction_space"),
        "loss_space": _nested(config, "framework", "loss_space"),
        "loss_weighting": _nested(config, "framework", "loss_weighting"),
        "training_steps": _nested(config, "training", "n_training_steps"),
        "learning_rate": _nested(config, "training", "lr"),
        "batch_size": _nested(config, "training", "batch_size"),
        "effective_batch_size": _nested(config, "training", "effective_batch_size"),
        "image_channels": _nested(config, "data", "image_channels"),
        "image_size": _nested(config, "data", "image_size"),
        "evaluation_checkpoint_count": len(manifest["checkpoint_steps"]),
        "evaluation_sample_count": manifest["sample_count"],
        "evaluation_tier": manifest["evaluation_tier"],
        "source_state": run_record.get("state", ""),
        "source_final_step": run_record.get("final_step", ""),
        "wandb_project": _nested(config, "tracking", "wandb", "project"),
        "wandb_group": _nested(config, "tracking", "wandb", "group"),
        "wandb_name": _nested(config, "tracking", "wandb", "name"),
        "source_run_path": str(source_run),
        "evaluation_result_path": str(result_root),
        "source_config_sha256": manifest["source_config_sha256"],
        "source_data_bundle_sha256": manifest["source_data_bundle_sha256"],
    }
    return row


def _training_rows(run_id, source_run):
    """Yield saved training loss and projected-monitor records."""
    history_path = source_run / "train_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    for record in _read_json(history_path):
        step = int(record["step"])
        for key, value in sorted(record.items()):
            if key == "step" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                yield {
                    "run_id": run_id,
                    "step": step,
                    "metric": key,
                    "value": value,
                }

    monitoring_root = source_run / "monitoring"
    for path in sorted(monitoring_root.glob("f_mem_step_*.json")):
        record = _read_json(path)
        step = int(record["step"])
        for key in MONITOR_METRICS:
            if key in record:
                yield {
                    "run_id": run_id,
                    "step": step,
                    "metric": f"monitor/{key}",
                    "value": record[key],
                }


def _evaluation_rows(run_id, result_root):
    """Yield the accepted combined evaluation metric table."""
    path = result_root / "combined" / "checkpoint_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation metric table: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {"run_id": run_id, **row}


def _diagnostic_rows(run_id, result_root):
    """Yield raw-range, runtime, and threshold-calibration scalars."""
    path = result_root / "combined" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    for checkpoint in _read_json(path)["checkpoints"]:
        step = int(checkpoint["step"])
        for category in ("raw_sample_statistics", "stage_runtimes"):
            for key, value in sorted(checkpoint.get(category, {}).items()):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                yield {
                    "run_id": run_id,
                    "step": step,
                    "category": category,
                    "metric": key,
                    "value": value,
                    "alpha": "",
                    "rho": "",
                }

        for key in ("elapsed_seconds", "sample_count"):
            value = checkpoint.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                yield {
                    "run_id": run_id,
                    "step": step,
                    "category": "checkpoint",
                    "metric": key,
                    "value": value,
                    "alpha": "",
                    "rho": "",
                }

        for space in (
            "f_mem_raw_pixels",
            "f_mem_sscd",
            "f_mem_lpips",
            "f_mem_dinov2",
        ):
            payload = checkpoint.get(space, {})
            curve = payload.get("curve", payload.get("calibrated", ()))
            for point in curve:
                for key in ("calibration_rate", "validation_rate"):
                    if key not in point:
                        continue
                    yield {
                        "run_id": run_id,
                        "step": step,
                        "category": "threshold_calibration",
                        "metric": f"{space}/{key}",
                        "value": point[key],
                        "alpha": point.get("alpha", ""),
                        "rho": point.get("rho", ""),
                    }


def _collect_records(evaluation_roots):
    """Collect and group accepted result identities before writing tables."""
    grouped = {}
    for evaluation_root in evaluation_roots:
        if not evaluation_root.is_dir():
            continue
        for result_root in _accepted_results(evaluation_root):
            accepted = _read_json(result_root / "ACCEPTED.json")
            if not accepted.get("accepted"):
                continue
            manifest = _read_json(result_root / "eval_manifest.json")
            if not manifest.get("publication_ready"):
                raise ValueError(
                    f"Accepted result is not publication ready: {result_root}"
                )
            source_run = Path(manifest["source_run"])
            config = _read_json(source_run / "config.json")
            run_record = _read_json(source_run / "run.json")
            identity = _identity(evaluation_root, result_root, config)
            grouped.setdefault(identity["experiment"], []).append(
                {
                    "identity": identity,
                    "manifest": manifest,
                    "config": config,
                    "run_record": run_record,
                    "source_run": source_run,
                    "result_root": result_root,
                }
            )

    for records in grouped.values():
        records.sort(key=lambda record: record["identity"]["run_id"])
    return grouped


def _write_experiment(output_root, experiment, records):
    """Write one self-contained, deterministic experiment directory."""
    experiment_root = output_root / experiment
    experiment_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": experiment_root / "runs.csv",
        "training": experiment_root / "training_metrics.csv",
        "evaluation": experiment_root / "evaluation_metrics.csv",
        "diagnostics": experiment_root / "evaluation_diagnostics.csv",
    }
    counts = {key: 0 for key in paths}

    with (
        paths["runs"].open("w", encoding="utf-8", newline="") as run_handle,
        paths["training"].open("w", encoding="utf-8", newline="") as train_handle,
        paths["evaluation"].open(
            "w", encoding="utf-8", newline=""
        ) as evaluation_handle,
        paths["diagnostics"].open(
            "w", encoding="utf-8", newline=""
        ) as diagnostic_handle,
    ):
        writer_options = {"lineterminator": "\n"}
        run_writer = csv.DictWriter(run_handle, fieldnames=RUN_FIELDS, **writer_options)
        train_writer = csv.DictWriter(
            train_handle, fieldnames=TRAIN_FIELDS, **writer_options
        )
        evaluation_writer = csv.DictWriter(
            evaluation_handle, fieldnames=EVALUATION_FIELDS, **writer_options
        )
        diagnostic_writer = csv.DictWriter(
            diagnostic_handle, fieldnames=DIAGNOSTIC_FIELDS, **writer_options
        )
        for writer in (
            run_writer,
            train_writer,
            evaluation_writer,
            diagnostic_writer,
        ):
            writer.writeheader()

        for record in records:
            identity = record["identity"]
            run_id = identity["run_id"]
            run_writer.writerow(
                _run_row(
                    identity,
                    record["manifest"],
                    record["config"],
                    record["run_record"],
                    record["source_run"],
                    record["result_root"],
                )
            )
            counts["runs"] += 1

            for row in _training_rows(run_id, record["source_run"]):
                train_writer.writerow(row)
                counts["training"] += 1
            for row in _evaluation_rows(run_id, record["result_root"]):
                evaluation_writer.writerow(row)
                counts["evaluation"] += 1
            for row in _diagnostic_rows(run_id, record["result_root"]):
                diagnostic_writer.writerow(row)
                counts["diagnostics"] += 1

    readme = f"""# {experiment}

This directory contains scalar training and evaluation records for one experiment.
Join metric tables to runs.csv through run_id.

It contains no checkpoints, generated sample banks, feature matrices, or datasets.
All CSV files are plain text with deterministic row ordering for clean Git diffs.
"""
    (experiment_root / "README.md").write_text(readme, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "accepted_run_count": counts["runs"],
        "row_counts": counts,
        "files": {},
    }
    for name, path in sorted(paths.items()):
        manifest["files"][path.name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "row_count": counts[name],
        }
    (experiment_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_root_readme(output_root):
    """Write the short navigation guide."""
    text = """# Plot data

Each subdirectory is one experiment with the same small relational layout:

- runs.csv: one row per accepted run and all condition labels.
- training_metrics.csv: saved training loss and monitoring trajectories.
- evaluation_metrics.csv: all accepted checkpoint-level evaluation metrics.
- evaluation_diagnostics.csv: raw-range, runtime, and calibration diagnostics.
- manifest.json: row counts and SHA-256 hashes.

Join each metric table to runs.csv with run_id.
No checkpoints or large tensors are stored here.

From the repository root, reproduce the current CelebA32 U-Net figure with:

```bash
python plotting/celeba32_3x3/plot_f_mem_grid.py
```

The PNG and its compact plotted table are written under
`outputs/figures/celeba32_unet_prediction_loss_3x3/`.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def export(output_root, evaluation_roots):
    """Export currently accepted results into experiment directories."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    grouped = _collect_records(evaluation_roots)
    index = {
        "schema_version": 1,
        "experiments": {},
    }
    for experiment, records in sorted(grouped.items()):
        manifest = _write_experiment(
            output_root,
            experiment,
            records,
        )
        index["experiments"][experiment] = {
            "accepted_run_count": manifest["accepted_run_count"],
            "row_counts": manifest["row_counts"],
            "manifest": f"{experiment}/manifest.json",
        }

    _write_root_readme(output_root)
    (output_root / "INDEX.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    index = export(args.output_root, EVALUATION_ROOTS)
    total = sum(
        record["accepted_run_count"] for record in index["experiments"].values()
    )
    print(f"Wrote {total} accepted runs to {args.output_root}")
    for experiment, record in sorted(index["experiments"].items()):
        print(
            f"  {experiment}: {record['accepted_run_count']} runs, "
            f"{record['row_counts']['evaluation']} evaluation rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
