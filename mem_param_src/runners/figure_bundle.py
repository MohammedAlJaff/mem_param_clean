"""Export completed checkpoint metrics as one portable figure bundle."""

import argparse
import csv
import hashlib
import math
import re
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from mem_param_src.runners.artifacts import (
    file_sha256,
    load_torch,
    read_json,
    tensor_sha256,
    write_json,
)
from mem_param_src.runners.checkpoint_metrics_benchmark import (
    _expected_generated_bank_identity,
    _load_generated_bank,
)

FIGURE_BUNDLE_SCHEMA_VERSION = 1
NEIGHBOR_KEYS = ("d1", "d2", "r", "nn_idx", "second_nn_idx")
QUANTILES = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    1.0,
)
EXAMPLE_SELECTION = {
    "space": "sscd",
    "strongest_count": 16,
    "boundary_count": 8,
    "non_memorized_count": 8,
    "alpha": 0.01,
}


def _utc_now():
    """Return one UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value):
    """Return one path-safe stable identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    if not cleaned:
        raise ValueError("Figure-bundle identifiers must contain a safe character.")
    return cleaned


def _nested(mapping, *keys, default=None):
    """Read one nested mapping value without changing the source."""
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _write_csv(path, fieldnames, rows):
    """Write one portable CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_neighbors(path, expected_count):
    """Load and verify one compact top-two neighbor artifact."""
    path = Path(path).expanduser().resolve()
    payload = load_torch(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Neighbor artifact must contain a dictionary: {path}")
    for key in NEIGHBOR_KEYS:
        value = payload.get(key)
        if not torch.is_tensor(value) or value.numel() != int(expected_count):
            raise ValueError(f"Neighbor field {key} is missing or misaligned: {path}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Neighbor field {key} contains non-finite values: {path}")
    return payload, {
        "path": str(path),
        "sha256": file_sha256(path),
        "count": int(expected_count),
    }


def _load_evaluation(root, allow_non_test_reference, example_keys):
    """Load one completed checkpoint gate and verify every reusable input."""
    root = Path(root).expanduser().resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint summary: {summary_path}")
    summary = read_json(summary_path)
    run_dir = Path(summary["run_dir"]).expanduser().resolve()
    config_path = run_dir / "config.json"
    run_path = run_dir / "run.json"
    bundle_path = run_dir / "data_bundle.pt"
    data_manifest_path = run_dir / "data_manifest.json"
    for path in (config_path, run_path, bundle_path, data_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved-run input: {path}")

    config = read_json(config_path)
    run_record = read_json(run_path)
    step = int(summary["checkpoint_step"])
    sample_count = int(summary["sample_count"])
    run_name = _nested(config, "outputs", "run_name", default=run_dir.name)
    run_id = _safe_id(run_name)
    final_step = int(_nested(config, "training", "n_training_steps", default=-1))
    example_key = f"{run_id}:{step}"
    examples_required = step == final_step or example_key in example_keys
    reference_split = str(summary["inception_fid"]["reference_split"])
    if reference_split != "test" and not bool(allow_non_test_reference):
        raise ValueError(
            "Publication figure bundles require reference_split='test'. "
            "Use --allow-non-test-reference only for development."
        )

    generated_bank = summary.get("generated_bank")
    if generated_bank is None:
        generated_bank = root / "generated_samples.pt"
    generated_bank = Path(generated_bank).expanduser().resolve()
    identity = _expected_generated_bank_identity(run_dir, step, sample_count)
    if generated_bank.is_file():
        samples, sampling_summary, generated_bank = _load_generated_bank(
            generated_bank, identity
        )
        generated_hash = tensor_sha256(samples)
        expected_generated_hash = summary.get("generated_tensor_sha256", generated_hash)
        if expected_generated_hash != generated_hash:
            raise ValueError("Evaluation summary and generated bank hashes differ.")
    else:
        if examples_required:
            raise FileNotFoundError(
                "A final or selected figure checkpoint must retain its generated "
                f"bank: {generated_bank}"
            )
        samples = None
        sampling_summary = _nested(summary, "sampling", "summary")
        generated_hash = summary.get("generated_tensor_sha256")
        if not generated_hash:
            raise ValueError(
                "An evaluation without a generated bank must save its tensor hash."
            )

    bundle = load_torch(bundle_path)
    training = bundle.get("train")
    if not torch.is_tensor(training):
        raise ValueError(f"Saved run has no training tensor: {run_dir}")
    training = training.contiguous()
    training_hash = tensor_sha256(training)
    if int(summary["training_count"]) != len(training):
        raise ValueError("Evaluation summary and training tensor counts differ.")

    calibration_count = int(summary["raw_pixel_f_mem"]["calibration_count"])
    validation_count = int(summary["raw_pixel_f_mem"]["validation_count"])
    spaces = {}
    input_files = []
    for space in ("raw_pixels", "sscd"):
        space_root = root / "metrics" / space
        partitions = {}
        for partition, count in (
            ("generated", sample_count),
            ("calibration", calibration_count),
            ("validation", validation_count),
        ):
            payload, source = _load_neighbors(
                space_root / f"{partition}_neighbors.pt", count
            )
            metadata = payload.get("metadata", {})
            if metadata.get("train_tensor_sha256") != training_hash:
                raise ValueError(
                    f"{space} {partition} neighbors use another training tensor."
                )
            if (
                partition == "generated"
                and metadata.get("generated_tensor_sha256") != generated_hash
            ):
                raise ValueError(f"{space} generated neighbors use another bank.")
            partitions[partition] = payload
            input_files.append(source)
        spaces[space] = partitions

    checkpoint_path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
    reference_path = Path(summary["reference_artifact"]).expanduser().resolve()
    expected_reference_hash = summary["reference_artifact_sha256"]
    if (
        not reference_path.is_file()
        or file_sha256(reference_path) != expected_reference_hash
    ):
        raise ValueError("Evaluation reference artifact failed hash verification.")
    if file_sha256(checkpoint_path) != identity["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint failed hash verification.")

    source_paths = [
        summary_path,
        config_path,
        run_path,
        bundle_path,
        data_manifest_path,
        checkpoint_path,
        reference_path,
        root / "metrics" / "raw_pixels" / "metrics.json",
        root / "metrics" / "sscd" / "metrics.json",
        root / "metrics" / "fid" / "metrics.json",
    ]
    if generated_bank.is_file():
        source_paths.append(generated_bank)
    for path in source_paths:
        input_files.append({"path": str(path), "sha256": file_sha256(path)})

    return {
        "root": root,
        "summary": summary,
        "config": config,
        "run_record": run_record,
        "run_dir": run_dir,
        "run_id": run_id,
        "step": step,
        "sample_count": sample_count,
        "reference_split": reference_split,
        "samples": samples,
        "training": training,
        "spaces": spaces,
        "generated_tensor_sha256": generated_hash,
        "training_tensor_sha256": training_hash,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "reference_artifact_sha256": expected_reference_hash,
        "sampling_summary": sampling_summary,
        "input_files": sorted(input_files, key=lambda record: record["path"]),
        "generated_bank_retained": generated_bank.is_file(),
        "examples_required": examples_required,
    }


def _common_row(evaluation):
    """Return identifiers shared by all portable metric rows."""
    config = evaluation["config"]
    return {
        "run_id": evaluation["run_id"],
        "step": evaluation["step"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "generated_tensor_sha256": evaluation["generated_tensor_sha256"],
        "reference_artifact_sha256": evaluation["reference_artifact_sha256"],
        "reference_split": evaluation["reference_split"],
        "sample_count": evaluation["sample_count"],
        "dataset": _nested(config, "data", "dataset", default=""),
        "architecture": _nested(config, "network", "arch", default=""),
        "n_train": _nested(config, "data", "n_train", default=""),
        "seed": _nested(config, "training", "seed", default=""),
        "prediction_space": _nested(
            config, "framework", "prediction_space", default=""
        ),
        "loss_space": _nested(config, "framework", "loss_space", default=""),
        "loss_weighting": _nested(config, "framework", "loss_weighting", default=""),
    }


def _metric_rows(evaluation):
    """Return long scalar and calibration tables for one checkpoint."""
    summary = evaluation["summary"]
    common = _common_row(evaluation)
    metrics = []
    calibration = []

    fid = summary["inception_fid"]
    metrics.append(
        {
            **common,
            "metric": "inception_fid",
            "space": "inception_v3",
            "alpha": "",
            "rho": "",
            "value": fid["fid_10k"],
            "calibration_rate": "",
            "validation_rate": "",
        }
    )
    raw = summary["raw_pixel_f_mem"]
    for rho, value in raw["fixed_rho"].items():
        metrics.append(
            {
                **common,
                "metric": "f_mem_fixed_rho",
                "space": "raw_pixels",
                "alpha": "",
                "rho": float(rho),
                "value": value,
                "calibration_rate": "",
                "validation_rate": "",
            }
        )
    for space, record in (
        ("raw_pixels", raw),
        ("sscd", summary["sscd_f_mem"]),
    ):
        for row in record["heldout_curve"]:
            curve = {
                **common,
                "space": space,
                "alpha": row["heldout_quantile"],
                "rho": row["rho"],
                "generated_f_mem": row["generated_f_mem"],
                "calibration_rate": row["calibration_rate"],
                "validation_rate": row["validation_rate"],
            }
            calibration.append(curve)
            metrics.append(
                {
                    **common,
                    "metric": "f_mem_calibrated",
                    "space": space,
                    "alpha": row["heldout_quantile"],
                    "rho": row["rho"],
                    "value": row["generated_f_mem"],
                    "calibration_rate": row["calibration_rate"],
                    "validation_rate": row["validation_rate"],
                }
            )
    return metrics, calibration


def _distribution_rows(evaluation):
    """Return fixed quantiles for all saved neighbor distributions."""
    common = _common_row(evaluation)
    rows = []
    for space, partitions in evaluation["spaces"].items():
        for partition, payload in partitions.items():
            for quantity in ("d1", "d2", "r"):
                values = payload[quantity].detach().cpu().float().numpy()
                quantile_values = np.quantile(values, QUANTILES)
                for quantile, value in zip(QUANTILES, quantile_values):
                    rows.append(
                        {
                            **common,
                            "space": space,
                            "partition": partition,
                            "quantity": quantity,
                            "quantile": quantile,
                            "value": float(value),
                        }
                    )
    return rows


def _tensor_numpy(tensor):
    """Convert one saved tensor to a contiguous NumPy array."""
    return tensor.detach().cpu().contiguous().numpy()


def _save_generated_arrays(path, payload):
    """Save generated top-two evidence in a portable compressed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: _tensor_numpy(payload[key]) for key in NEIGHBOR_KEYS}
    )


def _save_reference_arrays(path, calibration, validation):
    """Save shared calibration and validation evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for partition, payload in (
        ("calibration", calibration),
        ("validation", validation),
    ):
        for key in NEIGHBOR_KEYS:
            arrays[f"{partition}_{key}"] = _tensor_numpy(payload[key])
    np.savez_compressed(path, **arrays)


def _deterministic_order(indices, identity):
    """Order integer candidates with one stable identity hash."""
    return sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"{identity}:{int(index)}".encode()
        ).hexdigest(),
    )


def _select_examples(evaluation):
    """Select fixed strong, boundary, and non-memorized examples."""
    neighbors = evaluation["spaces"][EXAMPLE_SELECTION["space"]]["generated"]
    ratios = neighbors["r"].detach().cpu().float()
    rho = float(evaluation["summary"]["sscd_f_mem"]["headline_alpha_0.01"]["rho"])
    used = set()
    selected = []

    strongest = sorted(
        range(len(ratios)), key=lambda index: (float(ratios[index]), index)
    )
    strongest = strongest[: EXAMPLE_SELECTION["strongest_count"]]
    for index in strongest:
        used.add(index)
        selected.append((index, "strongest"))

    boundary = sorted(
        (index for index in range(len(ratios)) if index not in used),
        key=lambda index: (abs(float(ratios[index]) - rho), index),
    )[: EXAMPLE_SELECTION["boundary_count"]]
    for index in boundary:
        used.add(index)
        selected.append((index, "boundary"))

    non_memorized = [
        index
        for index in range(len(ratios))
        if index not in used and float(ratios[index]) >= rho
    ]
    identity = f"{evaluation['run_id']}:{evaluation['step']}:non-memorized"
    non_memorized = _deterministic_order(non_memorized, identity)[
        : EXAMPLE_SELECTION["non_memorized_count"]
    ]
    for index in non_memorized:
        selected.append((index, "non_memorized"))
    return selected, rho


def _display_image(tensor, size=96):
    """Convert one exact tensor to a clamped nearest-neighbor preview."""
    tensor = torch.as_tensor(tensor).detach().cpu().float().clamp(-1.0, 1.0)
    array = ((tensor + 1.0) * 127.5).round().to(torch.uint8).numpy()
    if array.shape[0] == 1:
        image = Image.fromarray(array[0], mode="L").convert("RGB")
    elif array.shape[0] == 3:
        image = Image.fromarray(np.transpose(array, (1, 2, 0)), mode="RGB")
    else:
        raise ValueError(f"Unsupported example channel count: {array.shape[0]}")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _save_example_panel(path, rows, arrays):
    """Save one compact preview of the exact exported example tensors."""
    examples_per_row = 4
    image_size = 96
    label_height = 18
    group_width = image_size * 3 + 12
    row_height = image_size + label_height + 8
    row_count = math.ceil(len(rows) / examples_per_row)
    panel = Image.new(
        "RGB",
        (examples_per_row * group_width, row_count * row_height),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    for position, row in enumerate(rows):
        column = position % examples_per_row
        panel_row = position // examples_per_row
        left = column * group_width
        top = panel_row * row_height
        draw.text(
            (left, top),
            f"{row['category']} | r={float(row['r']):.3f}",
            fill="black",
        )
        for offset, key in enumerate(("generated", "nearest", "second_nearest")):
            image = _display_image(arrays[key][position], size=image_size)
            panel.paste(image, (left + offset * image_size, top + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def _export_examples(root, evaluation):
    """Export exact figure tensors, a readable table, and one PNG preview."""
    selected, rho = _select_examples(evaluation)
    neighbors = evaluation["spaces"][EXAMPLE_SELECTION["space"]]["generated"]
    query_indices = torch.as_tensor([index for index, _ in selected], dtype=torch.long)
    nearest_indices = neighbors["nn_idx"][query_indices].to(torch.long)
    second_indices = neighbors["second_nn_idx"][query_indices].to(torch.long)
    training = evaluation["training"]
    if int(nearest_indices.max()) >= len(training) or int(second_indices.max()) >= len(
        training
    ):
        raise ValueError("Saved example neighbor index is outside the training bank.")

    arrays = {
        "query_index": _tensor_numpy(query_indices),
        "generated": _tensor_numpy(evaluation["samples"][query_indices]),
        "nearest_train_index": _tensor_numpy(nearest_indices),
        "nearest": _tensor_numpy(training[nearest_indices]),
        "second_train_index": _tensor_numpy(second_indices),
        "second_nearest": _tensor_numpy(training[second_indices]),
        "d1": _tensor_numpy(neighbors["d1"][query_indices]),
        "d2": _tensor_numpy(neighbors["d2"][query_indices]),
        "r": _tensor_numpy(neighbors["r"][query_indices]),
        "category": np.asarray([category for _, category in selected]),
    }
    example_root = (
        root
        / "examples"
        / evaluation["run_id"]
        / f"step_{evaluation['step']:06d}"
        / EXAMPLE_SELECTION["space"]
    )
    example_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(example_root / "examples.npz", **arrays)
    rows = []
    for position, (query_index, category) in enumerate(selected):
        rows.append(
            {
                "run_id": evaluation["run_id"],
                "step": evaluation["step"],
                "space": EXAMPLE_SELECTION["space"],
                "position": position,
                "category": category,
                "query_index": query_index,
                "nearest_train_index": int(nearest_indices[position]),
                "second_train_index": int(second_indices[position]),
                "d1": float(arrays["d1"][position]),
                "d2": float(arrays["d2"][position]),
                "r": float(arrays["r"][position]),
                "rho": rho,
            }
        )
    _write_csv(
        example_root / "examples.csv",
        list(rows[0]) if rows else [],
        rows,
    )
    _save_example_panel(example_root / "panel.png", rows, arrays)
    return rows


def _run_row(evaluation):
    """Return one stable run-level metadata row."""
    config = evaluation["config"]
    return {
        "run_id": evaluation["run_id"],
        "run_dir": str(evaluation["run_dir"]),
        "study": _nested(config, "study", "name", default=""),
        "dataset": _nested(config, "data", "dataset", default=""),
        "architecture": _nested(config, "network", "arch", default=""),
        "n_train": _nested(config, "data", "n_train", default=""),
        "seed": _nested(config, "training", "seed", default=""),
        "prediction_space": _nested(
            config, "framework", "prediction_space", default=""
        ),
        "loss_space": _nested(config, "framework", "loss_space", default=""),
        "loss_weighting": _nested(config, "framework", "loss_weighting", default=""),
        "training_steps": _nested(config, "training", "n_training_steps", default=""),
        "config_sha256": file_sha256(evaluation["run_dir"] / "config.json"),
        "data_bundle_sha256": file_sha256(evaluation["run_dir"] / "data_bundle.pt"),
        "training_tensor_sha256": evaluation["training_tensor_sha256"],
    }


def _write_bundle_readme(root, study_id, publication_ready):
    """Write short portable usage instructions."""
    readiness = (
        "yes" if publication_ready else "no; this bundle uses a non-test reference"
    )
    text = f"""# Figure bundle: {study_id}

Publication ready: {readiness}.

This archive contains saved numbers and selected image tensors. It does not contain models or encoder features.

Load the main table:

```python
import csv

with open("tables/checkpoint_metrics.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

print(rows[0]["metric"], rows[0]["value"])
```

Load a neighbor array:

```python
import numpy as np

data = np.load("arrays/RUN_ID/step_STEP/sscd.npz")
print(data["r"].shape)
```

Use `checksums.sha256` to verify all files. Use `manifest.json` for source hashes and selection rules.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_checksums(root):
    """Write hashes for every portable file except the checksum list."""
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}" for path in paths
    ]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(paths)


def _write_archive(root, archive_path):
    """Create one compressed archive with the bundle directory as its root."""
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    temporary.replace(archive_path)


def build_figure_bundle(
    evaluation_roots,
    output_dir,
    *,
    study_id,
    example_keys=(),
    allow_non_test_reference=False,
):
    """Build one portable, hash-checked figure bundle without recomputation."""
    output_dir = Path(output_dir).expanduser().resolve()
    archive_path = Path(f"{output_dir}.tar.gz")
    if output_dir.exists() or archive_path.exists():
        raise FileExistsError(f"Refusing to replace a figure bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    example_keys = {str(value) for value in example_keys}
    evaluations = [
        _load_evaluation(root, allow_non_test_reference, example_keys)
        for root in evaluation_roots
    ]
    if not evaluations:
        raise ValueError("At least one completed evaluation root is required.")
    keys = [(item["run_id"], item["step"]) for item in evaluations]
    if len(keys) != len(set(keys)):
        raise ValueError("Figure-bundle evaluation roots contain duplicate run steps.")
    known_keys = {f"{run_id}:{step}" for run_id, step in keys}
    unknown_keys = example_keys.difference(known_keys)
    if unknown_keys:
        raise ValueError(f"Unknown example keys: {sorted(unknown_keys)}")

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        root = Path(temporary) / output_dir.name
        root.mkdir()
        metric_rows = []
        calibration_rows = []
        distribution_rows = []
        example_rows = []
        run_rows = {}
        reference_exports = {}

        for evaluation in evaluations:
            run_id = evaluation["run_id"]
            step = evaluation["step"]
            run_row = _run_row(evaluation)
            if run_id in run_rows and run_rows[run_id] != run_row:
                raise ValueError(f"Run identifier maps to different sources: {run_id}")
            run_rows[run_id] = run_row
            rows, curves = _metric_rows(evaluation)
            metric_rows.extend(rows)
            calibration_rows.extend(curves)
            distribution_rows.extend(_distribution_rows(evaluation))

            for space, partitions in evaluation["spaces"].items():
                generated_path = (
                    root / "arrays" / run_id / f"step_{step:06d}" / f"{space}.npz"
                )
                _save_generated_arrays(generated_path, partitions["generated"])
                reference_key = (run_id, space)
                reference_identity = (
                    partitions["calibration"]["metadata"].get(
                        "reference_tensor_sha256"
                    ),
                    partitions["validation"]["metadata"].get("reference_tensor_sha256"),
                    evaluation["training_tensor_sha256"],
                )
                if reference_key not in reference_exports:
                    reference_path = (
                        root / "arrays" / run_id / "reference" / f"{space}.npz"
                    )
                    _save_reference_arrays(
                        reference_path,
                        partitions["calibration"],
                        partitions["validation"],
                    )
                    reference_exports[reference_key] = reference_identity
                elif reference_exports[reference_key] != reference_identity:
                    raise ValueError(
                        f"Run {run_id} uses inconsistent {space} reference evidence."
                    )

            final_step = int(
                _nested(
                    evaluation["config"],
                    "training",
                    "n_training_steps",
                    default=-1,
                )
            )
            key = f"{run_id}:{step}"
            if step == final_step or key in example_keys:
                example_rows.extend(_export_examples(root, evaluation))

        run_fieldnames = list(next(iter(run_rows.values())))
        metric_fieldnames = list(metric_rows[0])
        calibration_fieldnames = list(calibration_rows[0])
        distribution_fieldnames = list(distribution_rows[0])
        _write_csv(
            root / "tables" / "runs.csv",
            run_fieldnames,
            [run_rows[key] for key in sorted(run_rows)],
        )
        _write_csv(
            root / "tables" / "checkpoint_metrics.csv",
            metric_fieldnames,
            sorted(
                metric_rows,
                key=lambda row: (
                    row["run_id"],
                    row["step"],
                    row["metric"],
                    str(row["space"]),
                    str(row["alpha"]),
                    str(row["rho"]),
                ),
            ),
        )
        _write_csv(
            root / "tables" / "calibration_curves.csv",
            calibration_fieldnames,
            sorted(
                calibration_rows,
                key=lambda row: (
                    row["run_id"],
                    row["step"],
                    row["space"],
                    row["alpha"],
                ),
            ),
        )
        _write_csv(
            root / "tables" / "distribution_quantiles.csv",
            distribution_fieldnames,
            sorted(
                distribution_rows,
                key=lambda row: (
                    row["run_id"],
                    row["step"],
                    row["space"],
                    row["partition"],
                    row["quantity"],
                    row["quantile"],
                ),
            ),
        )
        if example_rows:
            _write_csv(
                root / "tables" / "examples.csv",
                list(example_rows[0]),
                example_rows,
            )

        reference_splits = sorted(
            {evaluation["reference_split"] for evaluation in evaluations}
        )
        publication_ready = reference_splits == ["test"]
        manifest = {
            "schema_version": FIGURE_BUNDLE_SCHEMA_VERSION,
            "bundle_kind": "portable_paper_figure_inputs",
            "study_id": str(study_id),
            "created_at": _utc_now(),
            "publication_ready": publication_ready,
            "reference_splits": reference_splits,
            "portable_formats": ["csv", "json", "npz", "png"],
            "consumer_requires_model": False,
            "consumer_requires_torch": False,
            "example_selection": {
                **EXAMPLE_SELECTION,
                "automatic_checkpoints": "final_training_step",
                "additional_keys": sorted(example_keys),
                "non_memorized_order": "sha256(run_id:step:non-memorized:index)",
                "display": "clamp[-1,1], uint8, nearest-neighbor resize",
                "exact_float_tensors_saved": True,
            },
            "quantiles": list(QUANTILES),
            "row_counts": {
                "runs": len(run_rows),
                "checkpoint_metrics": len(metric_rows),
                "calibration_curves": len(calibration_rows),
                "distribution_quantiles": len(distribution_rows),
                "examples": len(example_rows),
            },
            "evaluations": [
                {
                    "root": str(evaluation["root"]),
                    "run_id": evaluation["run_id"],
                    "step": evaluation["step"],
                    "reference_split": evaluation["reference_split"],
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "generated_tensor_sha256": evaluation["generated_tensor_sha256"],
                    "training_tensor_sha256": evaluation["training_tensor_sha256"],
                    "reference_artifact_sha256": evaluation[
                        "reference_artifact_sha256"
                    ],
                    "generated_bank_retained": evaluation["generated_bank_retained"],
                    "examples_exported": evaluation["examples_required"],
                    "input_files": evaluation["input_files"],
                }
                for evaluation in evaluations
            ],
            "exporter": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        }
        write_json(root / "manifest.json", manifest)
        _write_bundle_readme(root, study_id, publication_ready)
        portable_file_count = _write_checksums(root)
        manifest["portable_file_count_without_checksum_list"] = portable_file_count
        write_json(root / "manifest.json", manifest)
        _write_checksums(root)
        root.replace(output_dir)

    _write_archive(output_dir, archive_path)
    print(f"Figure bundle: {output_dir}", flush=True)
    print(f"Archive: {archive_path}", flush=True)
    print(f"Archive SHA-256: {file_sha256(archive_path)}", flush=True)
    return output_dir, archive_path


def build_parser():
    """Build the portable figure-bundle command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--example-key",
        action="append",
        default=[],
        help="Also export examples for RUN_ID:STEP.",
    )
    parser.add_argument(
        "--allow-non-test-reference",
        action="store_true",
        help="Allow a development bundle from a non-test reference.",
    )
    return parser


def main(argv=None):
    """Create one portable bundle from completed checkpoint evaluations."""
    args = build_parser().parse_args(argv)
    build_figure_bundle(
        args.evaluation_root,
        args.output_dir,
        study_id=args.study_id,
        example_keys=args.example_key,
        allow_non_test_reference=args.allow_non_test_reference,
    )
    return 0


if __name__ == "__main__":
    main()
