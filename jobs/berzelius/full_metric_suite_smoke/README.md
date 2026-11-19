# Full metric-suite smoke

This frozen three-task array evaluates all saved checkpoints for:

- CelebA32 U-Net-S;
- MNIST32 U-Net-S;
- CelebA32 Small-DiT.

Each checkpoint uses the first 512 rows of the canonical seed-2003 noise bank.
The two natural-image tasks use the complete clean CelebA test bank. Results
are diagnostic only.

The optional `final-replica10000` profile evaluates only step 1,000,000 for
the same three runs. It uses all 10,000 canonical noise rows and writes to
`outputs/smoke/final_metric_suite_replica_10000/`. This profile tests the
production-scale per-checkpoint path. It remains a development-only dummy run
and does not resume the final paper evaluation or archive.

Preview the initial submission without writing files:

```bash
bash jobs/berzelius/full_metric_suite_smoke/submit.sh --dry-run
```

Submit all three tasks without a concurrency throttle:

```bash
bash jobs/berzelius/full_metric_suite_smoke/submit.sh
```

Submit the isolated final-checkpoint replica smoke:

```bash
bash jobs/berzelius/full_metric_suite_smoke/submit.sh \
  --profile final-replica10000
```

Restart one failed task from its first incomplete checkpoint:

```bash
bash jobs/berzelius/full_metric_suite_smoke/submit.sh \
  --resume-root outputs/smoke/full_metric_suite_512/TIMESTAMP \
  --tasks 1
```

Each worker invokes the evaluator twice. The second pass must reuse all samples,
features, neighbor results, distances, and losses. The acceptance validator
checks checkpoint counts, finite metrics, f_mem ranges, tensor shapes, plots,
and second-pass reuse. Source input hashes are checked before and after work.
