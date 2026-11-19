"""Hash every frozen run input for the all-1M paper evaluation."""

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_hashes(path, files):
    missing = [item for item in files if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen input is missing: {missing[0]}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for source in files:
            handle.write(f"{_sha256(source)}  {source}\n")
    os.replace(temporary, path)
    return path


def _task_files(repo_root, row):
    run_dir = repo_root / row["run_path"]
    files = [
        run_dir / "config.json",
        run_dir / "run.json",
        run_dir / "data_bundle.pt",
        run_dir / "data_manifest.json",
        run_dir / "z_init.pt",
        repo_root / row["noise_path"],
        repo_root / Path(row["noise_path"]).with_suffix(".json"),
    ]
    files.extend(sorted((run_dir / "checkpoints").glob("step_*.pt")))
    if row["reference_path"] != "-":
        reference = repo_root / row["reference_path"]
        files.extend([reference, reference.with_suffix(".json")])
    if len(list((run_dir / "checkpoints").glob("step_*.pt"))) != int(
        row["expected_checkpoints"]
    ):
        raise ValueError(f"Checkpoint count changed: {run_dir}")
    return sorted({path.resolve() for path in files})


def freeze_inputs(repo_root, job_root, variants_path, workers=8):
    """Hash common assets and each task's immutable source artifacts."""
    repo_root = Path(repo_root).expanduser().resolve()
    job_root = Path(job_root).expanduser().resolve()
    variants_path = Path(variants_path).expanduser().resolve()
    with variants_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 198:
        raise ValueError(f"Expected 198 frozen tasks; found {len(rows)}.")

    assets = repo_root / "outputs" / "model_assets" / "feature_f_mem"
    common = [
        assets / "manifest.json",
        assets / "weights" / "inception_v3_compat.pth",
        assets / "weights" / "sscd_disc_mixup.torchscript.pt",
        assets / "torch_home" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth",
        assets / "torch_home" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth",
        assets / "lpips" / "alex_v0.1.pth",
        assets / "dinov2" / "hubconf.py",
    ]
    _write_hashes(job_root / "common_inputs.sha256", common)

    def freeze_task(row):
        task_root = (
            job_root
            / "tasks"
            / row["group"]
            / f"{row['task_id']}_{row['task_key']}"
        )
        task_root.mkdir(parents=True, exist_ok=True)
        return _write_hashes(
            task_root / "source_inputs.sha256",
            _task_files(repo_root, row),
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(freeze_task, row) for row in rows]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(rows):
                print(f"Hashed {completed}/{len(rows)} task input sets.", flush=True)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = freeze_inputs(
        args.repo_root,
        args.job_root,
        args.variants,
        workers=args.workers,
    )
    print(f"Frozen input identities for {len(rows)} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
