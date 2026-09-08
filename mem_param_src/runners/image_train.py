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
    return parser


def main(argv=None):
    """Run image training and optional default evaluation."""
    args = build_parser().parse_args(argv)
    config = load_python_config(args.config)
    apply_overrides(config, parse_cli_overrides(args.override_items))
    if args.wandb is not None:
        config.tracking.wandb.enabled = bool(args.wandb)
        if args.wandb and args.wandb_mode is None:
            config.tracking.wandb.mode = "online"
    if args.wandb_mode is not None:
        config.tracking.wandb.mode = str(args.wandb_mode)
    if args.generation_monitor is not None:
        config.generation_monitor.enabled = bool(args.generation_monitor)
    if args.f_mem_monitor is not None:
        config.random_projected_f_mem.enabled = bool(args.f_mem_monitor)
    if args.f_mem_samples is not None:
        config.random_projected_f_mem.n_samples = args.f_mem_samples
    run_dir = train_run(config, args.run_dir, run_kind="image", device=args.device)
    if args.evaluate_after_training:
        evaluate_image_run(run_dir, device=args.device)
    print(run_dir)
    return 0


if __name__ == "__main__":
    main()
