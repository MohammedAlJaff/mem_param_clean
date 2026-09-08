"""Measure epsilon error and induced velocity error on exact path states."""

import argparse
import math
from pathlib import Path

import torch
from ml_collections import ConfigDict

from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime, resolve_device
from mem_param_src.runners.artifacts import (
    list_checkpoints,
    load_torch,
    read_json,
    write_json,
)


DEFAULT_TIMES = (0.01, 0.0125, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99)


def build_parser():
    """Build the endpoint-conditioning command parser."""
    parser = argparse.ArgumentParser(
        description="Measure epsilon and velocity errors on exact path states."
    )
    parser.add_argument("--run-dir", required=True, help="Saved image run directory.")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Checkpoint step. The saved final step is the default.",
    )
    parser.add_argument(
        "--times",
        nargs="+",
        type=float,
        default=DEFAULT_TIMES,
        help="Path times in the open interval from zero to one.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, mps, or cuda.",
    )
    return parser


def _select_checkpoint(run_dir, requested_step):
    """Select one existing checkpoint by its recorded step."""
    checkpoints = dict(list_checkpoints(run_dir))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints exist in {run_dir}")
    step = max(checkpoints) if requested_step is None else requested_step
    if step not in checkpoints:
        raise ValueError(f"Checkpoint step {step} does not exist.")
    return step, checkpoints[step]


def _validate_inputs(config, times):
    """Require an eps/eps model and valid controlled probe times."""
    if config.framework.prediction_space != "eps":
        raise ValueError("This diagnostic requires prediction_space='eps'.")
    if config.framework.loss_space != "eps":
        raise ValueError("This diagnostic requires loss_space='eps'.")
    if any(not math.isfinite(time) or not 0.0 < time < 1.0 for time in times):
        raise ValueError("Every probe time must be finite and between zero and one.")
    if len(set(times)) != len(times):
        raise ValueError("Probe times must be unique.")


def _matching_report(path, expected):
    """Reuse a complete deterministic report when its probe settings match."""
    if not path.exists():
        return False
    report = read_json(path)
    keys = ("checkpoint_step", "device", "probe_count", "times")
    if any(report.get(key) != expected[key] for key in keys):
        raise FileExistsError(f"Diagnostic report settings do not match: {path}")
    print(f"reuse {path}", flush=True)
    return True


@torch.no_grad()
def main(argv=None):
    """Evaluate one checkpoint on fixed, exact interpolation states."""
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = ConfigDict(read_json(run_dir / "config.json"))
    times = tuple(sorted(args.times))
    _validate_inputs(config, times)

    bundle = load_torch(run_dir / "data_bundle.pt")
    z_init = load_torch(run_dir / "z_init.pt")
    train = bundle["train"]
    probe_count = min(train.shape[0], z_init.shape[0])
    if probe_count < 1:
        raise ValueError("The saved tensors do not contain a probe sample.")

    step, checkpoint_path = _select_checkpoint(run_dir, args.checkpoint_step)
    device = resolve_device(args.device)
    report_path = run_dir / "diagnostics" / f"eps_eps_conditioning_step_{step:06d}.json"
    expected = {
        "checkpoint_step": step,
        "device": str(device),
        "probe_count": probe_count,
        "times": list(times),
    }
    if _matching_report(report_path, expected):
        return 0

    configure_runtime(config, device)
    model = GDFM(config).to(device)
    checkpoint = load_torch(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    # Reuse saved tensors to keep every time probe paired and deterministic.
    clean = train[:probe_count].to(device)
    eps = z_init[:probe_count].to(device)
    velocity = clean - eps
    records = []

    for time_value in times:
        time = torch.full(
            (probe_count,),
            time_value,
            device=device,
            dtype=clean.dtype,
        )
        state = model.z_t_interpolate(clean, eps, time)
        eps_prediction = model.predict(state, time)
        eps_error = eps_prediction - eps

        velocity_prediction = model.convert(
            eps_prediction,
            state,
            time,
            source="eps",
            target="v",
            use_safe_division=False,
            safe_div_denom_eps=float(config.sampling.safe_div_denom_eps),
        )
        velocity_error = velocity_prediction - velocity

        time_b = time.view([probe_count] + [1] * (clean.dim() - 1))
        direct_velocity = (state - eps_prediction) / time_b
        eps_mse = eps_error.square().mean()
        velocity_mse = velocity_error.square().mean()
        induced_velocity_mse = eps_mse / (time_value**2)

        records.append(
            {
                "t": time_value,
                "amplification": 1.0 / time_value,
                "eps_mse": float(eps_mse.cpu()),
                "eps_rmse": float(eps_mse.sqrt().cpu()),
                "velocity_mse": float(velocity_mse.cpu()),
                "velocity_rmse": float(velocity_mse.sqrt().cpu()),
                "induced_velocity_mse": float(induced_velocity_mse.cpu()),
                "conversion_max_abs_difference": float(
                    (velocity_prediction - direct_velocity).abs().max().cpu()
                ),
            }
        )

    report = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_path.name,
        "checkpoint_step": step,
        "device": str(device),
        "probe_count": probe_count,
        "times": list(times),
        "probe_source": "saved train prefix paired with saved z_init prefix",
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
