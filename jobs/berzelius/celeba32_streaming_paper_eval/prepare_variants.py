"""Build a frozen task matrix for completed CelebA32 one-million-step runs."""

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

DIT_SEED1_BLOCK = {
    **BLOCKS[1],
    "name": "dit_seed1",
    "root": "outputs/job_sets/celeba32_dit_robustness_s1_1m_viax/20260929T124425Z_2240178/runs",
    "run_count": 18,
}

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
    if "dit_seed1" in by_block:
        items = sorted(
            by_block["dit_seed1"],
            key=lambda item: _numeric_key(Path(item["run_path"])),
        )
        return [
            {
                **item,
                "group": "dit_seed1",
                "task_id": task_id,
                "task_key": Path(item["run_path"]).name,
            }
            for task_id, item in enumerate(items)
        ]

    groups = [
        (
            "dit_seed0",
            sorted(
                by_block["dit_seed0"],
                key=lambda item: _numeric_key(Path(item["run_path"])),
            ),
        ),
    ]
    unet = sorted(
        by_block["celeba32_unet"],
        key=lambda item: _numeric_key(Path(item["run_path"])),
    )
    groups.extend(
        [
            ("celeba32_unet_a", unet[:36]),
            ("celeba32_unet_b", unet[36:72]),
            ("celeba32_unet_c", unet[72:]),
        ]
    )

    expected_counts = {
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


def build_variants(repo_root, output_path, dit_seed1=False):
    """Validate the completed matrix and write its frozen task table."""
    repo_root = Path(repo_root).expanduser().resolve()
    records = []
    cells = set()
    blocks = (DIT_SEED1_BLOCK,) if dit_seed1 else BLOCKS
    expected_count = sum(block["run_count"] for block in blocks)
    for block in blocks:
        run_root = repo_root / block["root"]
        run_dirs = sorted(
            (path for path in run_root.iterdir() if (path / "config.json").is_file()),
            key=_numeric_key,
        )
        if len(run_dirs) != int(block["run_count"]):
            raise ValueError(f"Block {block['name']} has {len(run_dirs)} runs.")
        for run_dir in run_dirs:
            record = _validate_run(repo_root, run_dir, block)
            if dit_seed1:
                config, state = _run_record(run_dir)
                if state["seed"] != 1 or config["network"]["arch"] != "dit":
                    raise ValueError(f"Expected a seed-1 DiT run: {run_dir}")
                cells.add(
                    (
                        config["data"]["n_train"],
                        config["framework"]["prediction_space"],
                        config["framework"]["loss_space"],
                    )
                )
                expected_steps = set(config["training"]["resolved_ckpt_steps"])
                checkpoints = list((run_dir / "checkpoints").glob("step_*.pt"))
                actual_steps = {int(path.stem.split("_")[1]) for path in checkpoints}
                if actual_steps != expected_steps or any(
                    path.stat().st_size == 0 for path in checkpoints
                ):
                    raise ValueError(f"Invalid saved checkpoint schedule: {run_dir}")
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
    if (
        len(rows) != expected_count
        or len({item["run_path"] for item in rows}) != expected_count
    ):
        raise ValueError(f"Expected exactly {expected_count} unique evaluation runs.")
    if dit_seed1 and cells != {
        (n_train, prediction, loss)
        for n_train in (512, 2048)
        for prediction in ("x", "eps", "v")
        for loss in ("x", "eps", "v")
    }:
        raise ValueError("Seed-1 DiT runs do not match the required 18-cell matrix.")

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
    parser.add_argument("--dit-seed1", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = build_variants(args.repo_root, args.output, dit_seed1=args.dit_seed1)
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(json.dumps({"run_count": len(rows), "groups": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
