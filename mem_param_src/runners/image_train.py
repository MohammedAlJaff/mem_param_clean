"""Command-line entry point for image training runs."""

import argparse

from mem_param_src.config import (
    apply_overrides,
    load_python_config,
    parse_cli_overrides,
)
from mem_param_src.runners.evaluation import evaluate_image_run
from mem_param_src.runners.training import train_run


def build_parser():
    """Build the image training command parser."""
    parser = argparse.ArgumentParser(description="Train one saved image run.")
    parser.add_argument("--config", required=True, help="Python experiment config.")
    parser.add_argument("--run-dir", required=True, help="New run directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument(
        "--set",
        dest="override_items",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one existing experiment config field.",
    )
    parser.add_argument(
        "--evaluate-after-training",
        action="store_true",
        help="Evaluate every saved checkpoint after training.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable W&B training logs.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        help="Override the configured W&B mode.",
    )
    parser.add_argument(
        "--generation-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable fixed-noise checkpoint image grids.",
    )
    parser.add_argument(
        "--save-generated-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save full generated tensors at monitoring and evaluation checkpoints.",
    )
    parser.add_argument(
        "--training-grid-samples",
        type=int,
        help="Show this many fixed training images above each generated grid; 0 disables.",
    )
    parser.add_argument(
        "--f-mem-monitor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compute rough projected-pixel f_mem with generation monitoring.",
    )
    parser.add_argument(
        "--f-mem-samples",
        type=int,
        help="Number of fixed-noise samples used by the optional f_mem monitor.",
    )
    training = parser.add_argument_group("training controls")
    training.add_argument("--steps", type=int, help="Number of training updates.")
    training.add_argument("--lr", type=float, help="Optimizer learning rate.")
    training.add_argument("--seed", type=int, help="Data, model, and noise-bank seed.")
    training.add_argument(
        "--batch-size", type=int, help="Maximum training batch size, capped at N."
    )
    training.add_argument(
        "--checkpoint-every",
        type=int,
        help="Use interval checkpoints, replacing any configured explicit schedule.",
    )
    sampling = parser.add_argument_group("sampling controls")
    sampling.add_argument("--sampler", choices=("heun", "euler"), help="ODE solver.")
    sampling.add_argument("--sampling-steps", type=int, help="ODE solver intervals.")
    sampling.add_argument(
        "--sampling-safe-division",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clamp sampling conversion denominators; does not change the training loss.",
    )
    sampling.add_argument(
        "--sampling-denom-eps",
        type=float,
        help="Denominator floor when sampling safe division is enabled.",
    )
    sampling.add_argument("--sampling-start-time", type=float, help="Sampling start time.")
    sampling.add_argument("--sampling-end-time", type=float, help="Sampling end time.")
    sampling.add_argument(
        "--sampling-output",
        choices=("x_1", "x_t_max"),
        help="Return the clean endpoint estimate or the final integrated state.",
    )
    return parser


def main(argv=None):
    """Run image training and optional default evaluation."""
    args = build_parser().parse_args(argv)
    config = load_python_config(args.config)
    apply_overrides(config, parse_cli_overrides(args.override_items))
    # Explicit convenience flags override the corresponding --set fields.
    for argument, field in (
        ("steps", "training.n_training_steps"),
        ("lr", "training.lr"),
        ("seed", "training.seed"),
        ("batch_size", "training.batch_cap"),
        ("checkpoint_every", "training.ckpt_every"),
        ("sampler", "sampling.method"),
        ("sampling_steps", "sampling.n_steps"),
        ("sampling_safe_division", "sampling.use_safe_division"),
        ("sampling_denom_eps", "sampling.safe_div_denom_eps"),
        ("sampling_start_time", "sampling.start_time"),
        ("sampling_end_time", "sampling.end_time"),
        ("sampling_output", "sampling.output"),
    ):
        value = getattr(args, argument)
        if value is not None:
            apply_overrides(config, {field: value})
    if args.checkpoint_every is not None:
        config.training.ckpt_schedule = "interval"
        config.training.ckpt_steps = ()
    if args.wandb is not None:
        config.tracking.wandb.enabled = bool(args.wandb)
        if args.wandb and args.wandb_mode is None:
            config.tracking.wandb.mode = "online"
    if args.wandb_mode is not None:
        config.tracking.wandb.mode = str(args.wandb_mode)
    if args.generation_monitor is not None:
        config.generation_monitor.enabled = bool(args.generation_monitor)
    if args.save_generated_samples is not None:
        config.evaluation.save_generated_samples = bool(args.save_generated_samples)
    if args.training_grid_samples is not None:
        config.generation_monitor.n_train_samples = args.training_grid_samples
    if args.f_mem_monitor is not None:
        config.random_projected_f_mem.enabled = bool(args.f_mem_monitor)
    if args.f_mem_samples is not None:
        config.random_projected_f_mem.n_samples = args.f_mem_samples
    run_dir = train_run(config, args.run_dir, run_kind="image", device=args.device)
    if args.evaluate_after_training:
        print("[eval] Evaluating saved checkpoints...", flush=True)
        evaluate_image_run(run_dir, device=args.device)
    print(f"[done] Results: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    main()
