"""Standalone command-line entry point for saved image evaluation."""

import argparse

from mem_param_src.runners.evaluation import evaluate_image_run


def build_parser():
    """Build the image evaluation command parser."""
    parser = argparse.ArgumentParser(description="Evaluate one saved image run.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory.")
    parser.add_argument(
        "--eval-config",
        help="Optional Python config containing sampling or evaluation settings.",
    )
    parser.add_argument("--eval-name", help="New name under run_dir/evaluations.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda.")
    parser.add_argument(
        "--set",
        dest="override_items",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one sampling or evaluation field.",
    )
    return parser


def main(argv=None):
    """Evaluate one saved image run without rebuilding its dataset."""
    args = build_parser().parse_args(argv)
    eval_dir = evaluate_image_run(
        args.run_dir,
        eval_name=args.eval_name,
        eval_config=args.eval_config,
        override_items=args.override_items,
        device=args.device,
    )
    print(eval_dir)
    return 0


if __name__ == "__main__":
    main()
