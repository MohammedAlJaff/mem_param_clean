"""Collect weighting figures into one flat directory with descriptive names."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_AGGREGATE_ROOT = Path(
    "outputs/evals/celeba32_vv_weighting_full_10000/20261003T134718Z_2109890/aggregate"
)
DEFAULT_OUTPUT_DIR = Path("outputs/figures/celeba32_vv_weighting_experiment")

TRAJECTORY_FIGURES = {
    "primary_overview.png": "primary_overview__seed_mean_sd.png",
    "quality.png": "quality_fid_dinov2__seed_mean_sd.png",
    "heldout_losses.png": "heldout_losses__seed_mean_sd.png",
    "memorization_raw_pixels_calibrated.png": (
        "fmem_raw_pixels__calibrated_alphas__seed_mean_sd.png"
    ),
    "memorization_raw_pixels_fixed_rho.png": (
        "fmem_raw_pixels__fixed_rhos__seed_mean_sd.png"
    ),
    "memorization_sscd_calibrated.png": (
        "fmem_sscd__calibrated_alphas__seed_mean_sd.png"
    ),
    "memorization_lpips_calibrated.png": (
        "fmem_lpips__calibrated_alphas__seed_mean_sd.png"
    ),
    "memorization_dinov2_calibrated.png": (
        "fmem_dinov2__calibrated_alphas__seed_mean_sd.png"
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_figures(aggregate_root, output_dir):
    aggregate_root = Path(aggregate_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for n_train in (512, 2048):
        source_dir = aggregate_root / "seed_trajectories" / f"n{n_train}"
        for source_name, descriptive_name in TRAJECTORY_FIGURES.items():
            source = source_dir / source_name
            destination = output_dir / f"n{n_train}__{descriptive_name}"
            shutil.copy2(source, destination)
            records.append(
                {
                    "file": destination.name,
                    "source": str(source),
                    "sha256": _sha256(destination),
                }
            )

        final_source = (
            aggregate_root / "quick_figures" / f"n{n_train}_final_f_mem_dinov2.png"
        )
        final_destination = (
            output_dir
            / f"n{n_train}__final_step__fmem_dinov2__alpha_0p01__seed_points.png"
        )
        shutil.copy2(final_source, final_destination)
        records.append(
            {
                "file": final_destination.name,
                "source": str(final_source),
                "sha256": _sha256(final_destination),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": "CelebA32 native-v/v loss-weighting experiment",
        "train_sizes": [512, 2048],
        "weightings": ["x_equivalent", "unit", "eps_equivalent"],
        "seeds": [0, 1, 2],
        "trajectory_line": "arithmetic seed mean",
        "trajectory_band": "plus or minus one sample standard deviation",
        "files": records,
    }
    (output_dir / "FIGURE_INDEX.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = """# CelebA32 v/v loss-weighting figures

This flat directory contains the current viewable PNG figures for the complete
loss-weighting experiment. Each filename starts with the training-set size.
Trajectory lines are means across seeds 0, 1, and 2. Shaded regions are plus or
minus one sample standard deviation. The three curves are x-equivalent, unit,
and epsilon-equivalent weighting.

`FIGURE_INDEX.json` records the exact source and SHA-256 hash of every image.
Regenerate this directory with:

```bash
python plotting/weighting/collect_figures.py
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Collected {len(records)} figures in {output_dir}")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-root", type=Path, default=DEFAULT_AGGREGATE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    collect_figures(args.aggregate_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
