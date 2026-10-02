#!/usr/bin/env python3
"""Measure one physical batch in one fresh CUDA process."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from mem_param_src.config import apply_overrides, load_python_config
from mem_param_src.datasets import build_data_bundle
from mem_param_src.model import GDFM
from mem_param_src.runners import configure_runtime


def build_parser():
    """Build the capacity-probe argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--profile-id", type=int, required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--constraint", choices=("thin", "fat"), required=True)
    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--channel-mult", required=True)
    parser.add_argument("--time-emb-dim", type=int, required=True)
    parser.add_argument("--attention", choices=("on", "off"), required=True)
    parser.add_argument("--middle-attention", choices=("true", "false"), required=True)
    parser.add_argument("--attn-resolutions", required=True)
    parser.add_argument("--physical-batch", type=int, required=True)
    parser.add_argument("--timed-steps", type=int, default=3)
    return parser


def write_result(path, record):
    """Write one independent result without a shared append operation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None):
    """Run warmup plus timed optimizer updates, or record a CUDA OOM."""
    args = build_parser().parse_args(argv)
    if args.physical_batch not in (64, 128, 256, 512):
        raise ValueError("physical batch must be one of 64, 128, 256, or 512")
    if args.timed_steps < 1:
        raise ValueError("timed steps must be positive")

    base_record = {
        "profile_id": args.profile_id,
        "capacity": args.capacity,
        "attention": args.attention,
        "constraint": args.constraint,
        "physical_batch": args.physical_batch,
        "effective_batch": max(256, args.physical_batch),
        "timed_steps": args.timed_steps,
        "accumulation_steps": max(256, args.physical_batch) // args.physical_batch,
    }
    model = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the capacity probe")
        config = load_python_config(args.config)
        apply_overrides(
            config,
            {
                "data.root": args.data_root,
                "data.dataset": "celeba32_grey",
                "data.train_split": "train",
                "data.reference_split": "valid",
                "data.image_size": 64,
                "data.resize_size": 64,
                "data.image_channels": 3,
                "network.base_channels": args.base_channels,
                "network.channel_mult": json.loads(args.channel_mult),
                "network.time_emb_dim": args.time_emb_dim,
                "network.middle_attention": args.middle_attention == "true",
                "network.attn_resolutions": json.loads(args.attn_resolutions),
            },
        )
        configure_runtime(config, torch.device("cuda"))
        random.seed(int(config.training.seed))
        np.random.seed(int(config.training.seed))
        torch.manual_seed(int(config.training.seed))
        torch.cuda.manual_seed_all(int(config.training.seed))

        train = build_data_bundle(config)["train"].contiguous()
        if train.shape[0] < args.physical_batch:
            raise ValueError("The real N=512 probe data cannot fill this batch")
        device = torch.device("cuda")
        model = GDFM(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )
        accumulation_steps = max(256, args.physical_batch) // args.physical_batch
        microbatches = [
            train[index * args.physical_batch : (index + 1) * args.physical_batch]
            for index in range(accumulation_steps)
        ]

        def optimizer_step():
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for microbatch in microbatches:
                loss = model.compute_loss(microbatch.to(device, non_blocking=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Capacity probe loss became non-finite")
                (loss / accumulation_steps).backward()
                value += float(loss.detach().cpu()) / accumulation_steps
            optimizer.step()
            return value

        torch.cuda.reset_peak_memory_stats()
        warmup_loss = optimizer_step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses = [optimizer_step() for _ in range(args.timed_steps)]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record = {
            **base_record,
            "status": "fit",
            "fits": True,
            "finite": all(math.isfinite(value) for value in [warmup_loss, *losses]),
            "warmup_loss": warmup_loss,
            "timed_loss_last": losses[-1],
            "elapsed_seconds": elapsed,
            "seconds_per_optimizer_step": elapsed / args.timed_steps,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "parameter_count": int(model.n_params()),
            "accumulation_steps": accumulation_steps,
        }
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        if not is_oom:
            raise
        record = {
            **base_record,
            "status": "oom",
            "fits": False,
            "finite": False,
            "error": str(exc),
            "seconds_per_optimizer_step": None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "parameter_count": int(model.n_params()) if model is not None else None,
        }
    write_result(args.output, record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
