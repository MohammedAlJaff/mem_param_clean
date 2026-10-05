#!/usr/bin/env python3
"""Require selected CelebA32 runs to have valid resumable state."""

import csv
import hashlib
import json
import sys
from pathlib import Path

from validate_array_spec import expand_array_spec


def file_sha256(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Check run state, error state, and the newest exact recovery payload."""
    batch_root = Path(sys.argv[1]).resolve()
    selected = expand_array_spec(sys.argv[2])
    with (batch_root / "variants.tsv").open(newline="") as handle:
        rows = {int(row["task_id"]): row for row in csv.DictReader(handle, delimiter="\t")}
    for task_id in sorted(selected):
        row = rows[task_id]
        run_dir = batch_root / "runs" / f"{task_id}_{row['label']}"
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("state") not in {"interrupted", "paused"}:
            raise AssertionError(f"Task {task_id} is not resumable: {record.get('state')}")
        if record.get("error") is not None:
            raise AssertionError(f"Task {task_id} records an error: {record['error']}")
        index = json.loads((run_dir / "recovery/index.json").read_text())
        latest = max(index["slots"].values(), key=lambda item: int(item["step"]))
        recovery = run_dir / "recovery" / latest["path"]
        if not recovery.is_file() or file_sha256(recovery) != latest["sha256"]:
            raise AssertionError(f"Task {task_id} has invalid latest recovery state.")
        if int(latest["step"]) >= 1_000_000:
            raise AssertionError(f"Task {task_id} is already at one million.")
        print(f"task {task_id}: resumable from exact step {latest['step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
