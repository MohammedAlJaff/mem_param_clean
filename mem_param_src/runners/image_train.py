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
    return parser


def main(argv=None):
    """Run image training and optional default evaluation."""
    args = build_parser().parse_args(argv)
    config = load_python_config(args.config)
    apply_overrides(config, parse_cli_overrides(args.override_items))
    run_dir = train_run(config, args.run_dir, run_kind="image", device=args.device)
    if args.evaluate_after_training:
        evaluate_image_run(run_dir, device=args.device)
    print(run_dir)
    return 0


if __name__ == "__main__":
    main()
