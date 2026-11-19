"""Summarize paired memorization at each seed's lowest-FID saved checkpoint."""

import csv
import statistics

from plot_f_mem_grid import (
    DEFAULT_FIGURE_ROOT,
    DEFAULT_PLOT_DATA_ROOT,
    N_TRAINS,
    TARGETS,
    _load_series,
    _selected_runs,
)

RHO = 0.3


def summarize(experiment="celeba32_unet_3x3", metric="f_mem_raw_pixels",
              rho=RHO, alpha=None, n_trains=N_TRAINS):
    """Return per-seed records and means; sample SD is absent for a single seed."""
    model_arch, seeds = {
        "celeba32_unet_3x3": ("unet", (0, 1, 2)),
        "celeba32_dit_3x3": ("dit", (0,)),
    }[experiment]
    if not ((metric == "f_mem_raw_pixels" and rho == RHO and alpha is None)
            or (metric == "f_mem_sscd" and rho is None and alpha == 0.01)):
        raise ValueError("Use fixed raw-pixel rho=0.3 or calibrated SSCD alpha=0.01.")
    root = DEFAULT_PLOT_DATA_ROOT.parent / experiment
    runs = _selected_runs(root, n_trains, model_arch, seeds)
    fid = _load_series(root, runs, None, "inception_fid")
    f_mem = _load_series(root, runs, rho, metric, alpha)
    selected_rows = []
    summary_rows = []

    for n_train in n_trains:
        for prediction in TARGETS:
            for loss in TARGETS:
                condition = {
                    "n_train": n_train,
                    "prediction_target": prediction,
                    "loss_target": loss,
                    **({"rho": rho} if alpha is None else {"alpha": alpha}),
                }
                seed_rows = []
                for seed in seeds:
                    key = (n_train, prediction, loss, seed)
                    if set(fid[key]) != set(f_mem[key]):
                        raise ValueError(f"FID and f_mem checkpoint grids differ: {key}")
                    # Select within each seed; exact FID ties use the earlier step.
                    step = min(fid[key], key=lambda step: (fid[key][step], step))
                    seed_rows.append(
                        {
                            **condition,
                            "seed": seed,
                            "run_id": runs[key],
                            "step": step,
                            "inception_fid": fid[key][step],
                            metric: f_mem[key][step],
                        }
                    )
                selected_rows.extend(seed_rows)
                summary = {**condition, "n_seeds": len(seed_rows)}
                for field in (metric, "inception_fid"):
                    values = [row[field] for row in seed_rows]
                    summary[f"{field}_mean"] = statistics.fmean(values)
                    summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else None
                summary_rows.append(summary)
    return selected_rows, summary_rows


def main():
    selected_rows, summary_rows = summarize()
    DEFAULT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("per_seed", selected_rows), ("mean_sd", summary_rows)):
        path = DEFAULT_FIGURE_ROOT / f"fmem_at_best_fid__rho_0p3__{suffix}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path.resolve()} ({len(rows)} rows)")

    cells = {
        (row["n_train"], row["prediction_target"], row["loss_target"]): row
        for row in summary_rows
    }
    print("\nRaw-pixel f_mem at each seed's best FID; rho=0.3, mean ± sample SD.")
    print("Each cell: f_mem mean ± SD (best FID mean ± SD).")
    print("| Prediction / loss | " + " | ".join(f"N={n}" for n in N_TRAINS) + " |")
    print("|---|" + "---:|" * len(N_TRAINS))
    for prediction in TARGETS:
        for loss in TARGETS:
            values = []
            for n_train in N_TRAINS:
                row = cells[(n_train, prediction, loss)]
                values.append(
                    f"{row['f_mem_raw_pixels_mean']:.4f} ± "
                    f"{row['f_mem_raw_pixels_std']:.4f} "
                    f"({row['inception_fid_mean']:.2f} ± {row['inception_fid_std']:.2f})"
                )
            print(f"| {prediction}/{loss} | " + " | ".join(values) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
