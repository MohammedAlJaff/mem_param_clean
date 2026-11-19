"""Split one accepted weighting aggregate into small metric tables."""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

DEFAULT_AGGREGATE = Path(
    "outputs/evals/celeba32_vv_weighting_full_10000/"
    "20261003T134718Z_2109890/aggregate/seed_trajectories/"
    "seed_aggregated_metrics.csv"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, fieldnames, rows):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def prepare_tables(aggregate_csv, output_dir=None):
    aggregate_csv = Path(aggregate_csv).expanduser().resolve()
    if output_dir is None:
        output_dir = aggregate_csv.parent / "plot_tables"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with aggregate_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows_by_metric = defaultdict(list)
        for row in reader:
            rows_by_metric[row["metric"]].append(row)
    if not fieldnames or not rows_by_metric:
        raise ValueError(f"No metric rows in {aggregate_csv}")

    tables = {}
    for metric, rows in sorted(rows_by_metric.items()):
        path = output_dir / f"{metric}.csv"
        _write_csv(path, fieldnames, rows)
        tables[path.name] = {"rows": len(rows), "sha256": _sha256(path)}

    index = {
        "schema_version": 1,
        "source_csv": str(aggregate_csv),
        "source_sha256": _sha256(aggregate_csv),
        "source_rows": sum(item["rows"] for item in tables.values()),
        "tables": tables,
    }
    index_path = output_dir / "INDEX.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), **index}, indent=2))
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    prepare_tables(args.aggregate_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
