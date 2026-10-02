#!/usr/bin/env python3
"""Select physical batches from complete six-profile CUDA probe results."""

import csv
import json
import sys
from pathlib import Path


HEADROOM_GIB = {"thin": 32.0, "fat": 64.0}
CANDIDATES = (256, 128, 64)


def read_tsv(path):
    """Read one tab-separated manifest."""
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    """Require all probe records, choose batches, and record launch segmentation."""
    batch_root = Path(sys.argv[1]).resolve()
    profiles = read_tsv(batch_root / "profiles.tsv")
    if len(profiles) != 6:
        raise AssertionError(f"Expected six profiles, found {len(profiles)}")

    selected = []
    all_results = []
    for profile in profiles:
        profile_id = int(profile["profile_id"])
        results = {}
        for physical_batch in (512, 256, 128, 64):
            path = (
                batch_root
                / "validation/probes/results"
                / f"profile_{profile_id}_batch_{physical_batch}.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing probe result: {path}")
            result = json.loads(path.read_text())
            if int(result["profile_id"]) != profile_id:
                raise AssertionError(f"Profile mismatch in {path}")
            if int(result["physical_batch"]) != physical_batch:
                raise AssertionError(f"Batch mismatch in {path}")
            results[physical_batch] = result
            all_results.append(result)

        threshold = HEADROOM_GIB[profile["constraint"]]
        chosen = None
        for physical_batch in CANDIDATES:
            result = results[physical_batch]
            reserved_gib = float(result.get("peak_reserved_bytes", 0)) / 2**30
            if result.get("fits") and result.get("finite") and reserved_gib <= threshold:
                chosen = result
                break
        if chosen is None:
            raise RuntimeError(
                f"No batch in {CANDIDATES} satisfies the {threshold:.0f} GiB "
                f"headroom rule for profile {profile_id}."
            )

        physical_batch = int(chosen["physical_batch"])
        seconds = float(chosen["seconds_per_optimizer_step"])
        estimate = seconds * 400_000 * 1.30 / 3600.0
        selected.append(
            {
                "profile_id": profile_id,
                "capacity": profile["capacity"],
                "attention": profile["attention"],
                "constraint": profile["constraint"],
                "physical_batch": physical_batch,
                "effective_batch": 256,
                "accumulation_steps": 256 // physical_batch,
                "peak_reserved_gib": float(chosen["peak_reserved_bytes"]) / 2**30,
                "seconds_per_optimizer_step": seconds,
                "estimated_400k_hours_with_30pct_margin": estimate,
            }
        )

    validation_dir = batch_root / "validation"
    selection_path = validation_dir / "selected_batches.tsv"
    fields = list(selected[0])
    with selection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected)

    maximum_hours = max(row["estimated_400k_hours_with_30pct_margin"] for row in selected)
    summary = {
        "probe_result_count": len(all_results),
        "selected_profile_count": len(selected),
        "headroom_gib": HEADROOM_GIB,
        "monitor_margin": 0.30,
        "maximum_estimated_400k_hours": maximum_hours,
        "launch_mode": "split" if maximum_hours > 40.0 else "direct",
        "split_rule": "0-200k then exact resume 200k-400k" if maximum_hours > 40.0 else None,
        "selected": selected,
    }
    (validation_dir / "probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (batch_root / "PROBE_COMPLETE").write_text("six profiles x four batches complete\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
