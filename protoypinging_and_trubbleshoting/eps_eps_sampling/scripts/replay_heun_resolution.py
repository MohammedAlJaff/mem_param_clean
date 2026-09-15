"""Replay one saved eps/eps image run with finer Heun grids."""

import argparse
from pathlib import Path

from mem_param_src.runners import resolve_device
from mem_param_src.runners.artifacts import read_json, write_json
from mem_param_src.runners.evaluation import evaluate_image_run


def build_parser():
    """Build the solver-resolution command parser."""
    parser = argparse.ArgumentParser(
        description="Replay one saved eps/eps image run with finer Heun grids."
    )
    parser.add_argument("--run-dir", required=True, help="Saved image run directory.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=(200, 500),
        help="Positive Heun step counts. The default values are 200 and 500.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, mps, or cuda.",
    )
    return parser


def _validate_saved_run(run_dir):
    """Require the saved eps/eps exact-division experiment."""
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing saved configuration: {config_path}")

    config = read_json(config_path)
    framework = config["framework"]
    sampling = config["sampling"]
    if framework["prediction_space"] != "eps" or framework["loss_space"] != "eps":
        raise ValueError("This diagnostic requires a saved eps/eps run.")
    if sampling["method"] != "heun":
        raise ValueError("This diagnostic requires the saved Heun method.")
    if sampling["use_safe_div"]:
        raise ValueError("This diagnostic requires exact sampling division.")
    if sampling["output"] != "x_1":
        raise ValueError("This diagnostic requires sampling.output='x_1'.")
    return config


def _reuse_complete_evaluation(eval_dir, expected_settings, resolved_device):
    """Reuse one complete evaluation only when every effective setting matches."""
    if not eval_dir.exists():
        return False

    required = (
        eval_dir / "settings.json",
        eval_dir / "checkpoint_metrics.json",
        eval_dir / "final_samples_x_1.pt",
        eval_dir / "diagnostic_context.json",
    )
    if not all(path.is_file() for path in required):
        raise FileExistsError(f"Evaluation directory is incomplete: {eval_dir}")

    actual = read_json(eval_dir / "settings.json")
    actual_settings = {
        "sampling": actual["sampling"],
        "evaluation": actual["evaluation"],
    }
    if actual_settings != expected_settings:
        raise FileExistsError(f"Evaluation settings do not match: {eval_dir}")
    context = read_json(eval_dir / "diagnostic_context.json")
    if context.get("resolved_device") != resolved_device:
        raise FileExistsError(f"Evaluation device does not match: {eval_dir}")
    print(f"reuse {eval_dir}", flush=True)
    return True


def main(argv=None):
    """Run each requested resolution without rebuilding data or training."""
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    saved_config = _validate_saved_run(run_dir)
    resolved_device = str(resolve_device(args.device))

    steps = tuple(dict.fromkeys(args.steps))
    if any(step < 1 for step in steps):
        raise ValueError("Every solver step count must be positive.")

    for step_count in steps:
        eval_name = f"heun_{step_count}_exact"
        eval_dir = run_dir / "evaluations" / eval_name
        expected_settings = {
            "sampling": dict(saved_config["sampling"]),
            "evaluation": dict(saved_config["evaluation"]),
        }
        expected_settings["sampling"]["n_steps"] = step_count

        if _reuse_complete_evaluation(eval_dir, expected_settings, resolved_device):
            continue

        print(f"evaluate n_steps={step_count}", flush=True)
        created_eval_dir = evaluate_image_run(
            run_dir,
            eval_name=eval_name,
            override_items=(f"sampling.n_steps={step_count}",),
            device=resolved_device,
        )
        write_json(
            created_eval_dir / "diagnostic_context.json",
            {
                "purpose": "eps/eps Heun solver-resolution test",
                "resolved_device": resolved_device,
                "changed_setting": "sampling.n_steps",
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
