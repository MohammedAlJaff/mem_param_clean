"""Standalone command-line entry point for saved image evaluation."""

import argparse

from mem_param_src.runners.evaluation import evaluate_image_run
from mem_param_src.runners.image_protocol import evaluate_image_protocol


def build_parser():
    """Build the image evaluation command parser."""
    parser = argparse.ArgumentParser(description="Evaluate one saved image run.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory.")
    parser.add_argument(
        "--eval-config",
        help="Optional Python config containing sampling or evaluation settings.",
    )
    parser.add_argument("--eval-name", help="New name under run_dir/evaluations.")
    parser.add_argument(
        "--checkpoints",
        help="Comma-separated optimizer steps, or 'all', for protocol evaluation.",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated protocol metrics, such as fid,f_mem,class_fd.",
    )
    parser.add_argument(
        "--feature-spaces",
        default="dataset_default",
        help="Comma-separated feature spaces for protocol evaluation.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        help="Explicit generated and reference count for protocol evaluation.",
    )
    parser.add_argument(
        "--protocol-id",
        help="Restartable protocol directory name under run_dir/evaluations.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional evaluation root outside the source run directory.",
    )
    parser.add_argument(
        "--noise-path",
        help="Optional saved noise bank when run_dir/z_init.pt is too small.",
    )
    parser.add_argument(
        "--noise-artifact",
        help="Verified canonical evaluation-noise artifact.",
    )
    parser.add_argument(
        "--reference-artifact",
        help="Frozen external held-out reference artifact.",
    )
    parser.add_argument(
        "--loss-count",
        type=int,
        help="Held-out images used for each loss estimate.",
    )
    parser.add_argument(
        "--write-plots",
        action="store_true",
        help="Write full-suite trajectories, grids, and neighbor panels.",
    )
    parser.add_argument(
        "--publication-evaluation",
        action="store_true",
        help=(
            "Require the 10,000-sample full-suite publication contract and "
            "mark its manifest as paper-tier."
        ),
    )
    parser.add_argument(
        "--assets-manifest",
        help="Pinned encoder asset manifest required by natural-image protocols.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--nn-batch-size", type=int, default=512)
    parser.add_argument(
        "--streaming-retention",
        action="store_true",
        help="Compact dense tensors immediately after each checkpoint completes.",
    )
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
    protocol_mode = args.streaming_retention or any(
        value is not None
        for value in (
            args.checkpoints,
            args.metrics,
            args.sample_count,
            args.protocol_id,
            args.output_dir,
            args.noise_path,
            args.noise_artifact,
            args.reference_artifact,
            args.loss_count,
            args.write_plots,
            args.assets_manifest,
        )
    )
    if protocol_mode:
        if args.eval_name is not None:
            raise ValueError("Use --protocol-id, not --eval-name, in protocol mode.")
        eval_dir = evaluate_image_protocol(
            args.run_dir,
            checkpoints=args.checkpoints or "all",
            metrics=args.metrics or "fid",
            feature_spaces=args.feature_spaces,
            sample_count=(
                args.sample_count if args.sample_count is not None else 10000
            ),
            protocol_id=args.protocol_id or "dataset-default-v1",
            output_dir=args.output_dir,
            noise_path=args.noise_path,
            noise_artifact=args.noise_artifact,
            reference_artifact=args.reference_artifact,
            loss_count=args.loss_count,
            write_plots=args.write_plots,
            publication_evaluation=args.publication_evaluation,
            streaming_retention=args.streaming_retention,
            assets_manifest=args.assets_manifest,
            eval_config=args.eval_config,
            override_items=args.override_items,
            device=args.device,
            feature_batch_size=args.feature_batch_size,
            nn_batch_size=args.nn_batch_size,
        )
    else:
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
