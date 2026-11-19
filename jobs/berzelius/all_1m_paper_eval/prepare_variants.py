"""Build and validate the frozen matrix for all completed one-million-step runs."""

import argparse
import csv
import json
import os
from pathlib import Path

BLOCKS = (
    {
        "name": "celeba32_unet",
        "root": "outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324/runs",
        "run_count": 108,
        "checkpoint_count": 72,
        "dataset": "celeba32_grey",
        "shape": (1, 32),
        "noise": "outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt",
        "reference": "outputs/reference_banks/celeba_test_seed1_clean_n10000.pt",
    },
    {
        "name": "mnist",
        "root": "outputs/job_sets/mnist_diagonal_1m/20260927T165106Z_1991997/runs",
        "run_count": 36,
        "checkpoint_count": 20,
        "dataset": "mnist",
        "shape": None,
        "noise": None,
        "reference": "-",
    },
    {
        "name": "celeba64_rgb",
        "root": "outputs/job_sets/celeba64_rgb_diagonal_1m/20260928T131627Z_3579761/runs",
        "run_count": 18,
        "checkpoint_count": 75,
        "dataset": "celeba32_grey",
        "shape": (3, 64),
        "noise": "outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000_c3x64.pt",
        "reference": "outputs/reference_banks/celeba64_rgb_test_seed1_clean_n10000.pt",
    },
    {
        "name": "dit_seed0",
        "root": "outputs/job_sets/celeba32_dit_3x3_nscale_1m_viax/20260926T104429Z_1787399/runs",
        "run_count": 36,
        "checkpoint_count": 72,
        "dataset": "celeba32_grey",
        "shape": (1, 32),
        "noise": "outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt",
        "reference": "outputs/reference_banks/celeba_test_seed1_clean_n10000.pt",
    },
)

FIELDNAMES = (
    "group",
    "task_id",
    "task_key",
    "run_path",
    "expected_checkpoints",
    "noise_path",
    "reference_path",
    "dataset",
    "image_channels",
    "image_size",
)


def _numeric_key(path):
    return int(path.name.split("_", 1)[0])


def _run_record(run_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return config, record


def _validate_run(repo_root, run_dir, block):
    config, record = _run_record(run_dir)
    if config["data"]["dataset"] != block["dataset"]:
        raise ValueError(f"Dataset mismatch: {run_dir}")
    shape = (
        int(config["data"]["image_channels"]),
        int(config["data"]["image_size"]),
    )
    if block["shape"] is not None and shape != block["shape"]:
        raise ValueError(f"Image shape mismatch: {run_dir}")
    if (
        record.get("state") != "trained"
        or int(record.get("final_step", -1)) != 1_000_000
        or record.get("error") is not None
    ):
        raise ValueError(f"Run is not complete at one million steps: {run_dir}")
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if len(checkpoints) != int(block["checkpoint_count"]):
        raise ValueError(f"Checkpoint count mismatch: {run_dir}")
    if not (run_dir / "checkpoints" / "step_1000000.pt").is_file():
        raise FileNotFoundError(f"Final checkpoint is missing: {run_dir}")
    try:
        relative = run_dir.resolve().relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"Run is outside the repository: {run_dir}") from error
    return {
        "run_path": str(relative),
        "dataset": block["dataset"],
        "image_channels": shape[0],
        "image_size": shape[1],
    }


def _assign_groups(records):
    by_block = {}
    for record in records:
        by_block.setdefault(record.pop("block"), []).append(record)
    rows = []

    mnist = sorted(by_block["mnist"], key=lambda item: _numeric_key(Path(item["run_path"])))
    mnist32 = [item for item in mnist if item["image_size"] == 32]
    mnist64 = [item for item in mnist if item["image_size"] == 64]
    groups = [
        ("gate_mnist32", mnist32[:1]),
        ("mnist32_rest", mnist32[1:]),
        ("mnist64", mnist64),
        ("celeba64_rgb", sorted(by_block["celeba64_rgb"], key=lambda item: _numeric_key(Path(item["run_path"])))),
        ("dit_seed0", sorted(by_block["dit_seed0"], key=lambda item: _numeric_key(Path(item["run_path"])))),
    ]
    unet = sorted(by_block["celeba32_unet"], key=lambda item: _numeric_key(Path(item["run_path"])))
    groups.extend(
        [
            ("celeba32_unet_a", unet[:36]),
            ("celeba32_unet_b", unet[36:72]),
            ("celeba32_unet_c", unet[72:]),
        ]
    )

    expected_counts = {
        "gate_mnist32": 1,
        "mnist32_rest": 17,
        "mnist64": 18,
        "celeba64_rgb": 18,
        "dit_seed0": 36,
        "celeba32_unet_a": 36,
        "celeba32_unet_b": 36,
        "celeba32_unet_c": 36,
    }
    for group, items in groups:
        if len(items) != expected_counts[group]:
            raise ValueError(f"Group {group} has {len(items)} runs.")
        for task_id, item in enumerate(items):
            rows.append(
                {
                    **item,
                    "group": group,
                    "task_id": task_id,
                    "task_key": item["run_path"].split("/")[-1],
                }
            )
    return rows


def build_variants(repo_root, output_path):
    """Validate the completed matrix and write its frozen task table."""
    repo_root = Path(repo_root).expanduser().resolve()
    records = []
    for block in BLOCKS:
        run_root = repo_root / block["root"]
        run_dirs = sorted(
            (path for path in run_root.iterdir() if (path / "config.json").is_file()),
            key=_numeric_key,
        )
        if len(run_dirs) != int(block["run_count"]):
            raise ValueError(f"Block {block['name']} has {len(run_dirs)} runs.")
        for run_dir in run_dirs:
            record = _validate_run(repo_root, run_dir, block)
            channels, size = record["image_channels"], record["image_size"]
            if block["name"] == "mnist":
                if (channels, size) == (1, 32):
                    noise = "outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt"
                elif (channels, size) == (1, 64):
                    noise = "outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000_c1x64.pt"
                else:
                    raise ValueError(f"Unsupported MNIST shape: {run_dir}")
            else:
                noise = block["noise"]
            records.append(
                {
                    **record,
                    "block": block["name"],
                    "expected_checkpoints": int(block["checkpoint_count"]),
                    "noise_path": noise,
                    "reference_path": block["reference"],
                }
            )
    rows = _assign_groups(records)
    if len(rows) != 198 or len({item["run_path"] for item in rows}) != 198:
        raise ValueError("The completed evaluation matrix is not exactly 198 unique runs.")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_path)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = build_variants(args.repo_root, args.output)
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(json.dumps({"run_count": len(rows), "groups": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
