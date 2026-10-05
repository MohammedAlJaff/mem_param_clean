#!/usr/bin/env python3
"""Build a two-task manifest for the saved v/v and x/x reference runs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_REFERENCES = {
    6: ("v", "v"),
    7: ("x", "x"),
}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def build_reference_manifest(source_manifest):
    source_manifest = Path(source_manifest).expanduser().resolve()
    source = read_json(source_manifest)
    sources = {int(row["source_index"]): row for row in source["sources"]}
    tasks = []
    for source_index, spaces in EXPECTED_REFERENCES.items():
        if source_index not in sources:
            raise ValueError(f"Source manifest is missing reference {source_index}.")
        task = dict(sources[source_index])
        if (task["prediction_space"], task["loss_space"]) != spaces:
            raise ValueError(f"Reference {source_index} does not have spaces {spaces}.")
        run_dir = Path(task["run_dir"])
        for step in source["checkpoint_steps"]:
            checkpoint = run_dir / "checkpoints" / f"step_{int(step):06d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        task["evaluation_profiles"] = [
            {
                "profile": "saved",
                "description": "Use the sampler saved with this reference run",
            }
        ]
        task["task_index"] = len(tasks)
        tasks.append(task)

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "run_root": source["run_root"],
        "max_step": source["max_step"],
        "checkpoint_steps": source["checkpoint_steps"],
        "checkpoint_count": source["checkpoint_count"],
        "tasks": tasks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    payload = build_reference_manifest(args.source_manifest)
    if args.output:
        write_json(args.output, payload)
    print(
        f"tasks={len(payload['tasks'])} checkpoints={payload['checkpoint_count']} "
        f"sources=6,7",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
