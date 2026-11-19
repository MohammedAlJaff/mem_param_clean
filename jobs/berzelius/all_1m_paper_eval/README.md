# All completed 1M-run paper evaluation

This launcher evaluates every production run that has reached 1,000,000
updates. It uses the complete metric suite and 10,000 generated samples at
every saved checkpoint.

The frozen matrix contains 198 runs:

- 108 CelebA32 U-Net-S runs;
- 18 MNIST32 U-Net-S runs;
- 18 MNIST64 U-Net-S runs;
- 18 CelebA64 RGB U-Net runs;
- 36 CelebA32 Small-DiT seed-0 runs.

The 18 Small-DiT seed-1 runs are not in this matrix because they have not yet
reached 1,000,000 updates.

## Execution order

The launcher first runs one MNIST32 gate. The gate must pass paper-tier
validation, cache-only replay, and exact-path cache retention. Later arrays use
Slurm `afterok` dependencies. At most 36 storage-heavy tasks run at once.

Each successful task keeps all metrics, tables, plots, feature statistics,
nearest-neighbor records, and samples from steps 0, 10k, 100k, and 1M. It
deletes only generated feature matrices, reference feature matrices, and other
checkpoint sample tensors from an identity-checked manifest.

## Commands

Preview the submission:

```bash
bash jobs/berzelius/all_1m_paper_eval/submit.sh --dry-run
```

Submit the frozen dependency chain:

```bash
bash jobs/berzelius/all_1m_paper_eval/submit.sh
```

Restart selected tasks from an existing frozen batch:

```bash
bash jobs/berzelius/all_1m_paper_eval/submit.sh \
  --resume-root outputs/evals/all_1m_full_10000/TIMESTAMP \
  --group celeba32_unet_a --tasks 2,7,11
```

The restart uses the frozen source and input hashes. The evaluator resumes at
the first incomplete checkpoint. If compaction was interrupted, it resumes
from the saved deletion manifest without rebuilding or resampling.

## CelebA64 finalization repair

The original CelebA64 workers completed their first pass before a live-source
change made the reuse pass reject the existing manifests. The repair worker
uses the batch's original frozen source, performs one cache-reuse pass, runs
paper-tier validation, and applies the verified retention manifest.

```bash
batch_root=outputs/evals/all_1m_full_10000/20261004T005702Z_3271331
sbatch --array=0-17 --account=berzelius-2026-269 --time=12:00:00 \
  --qos=normal --constraint=fat --gres=gpu:A100-SXM4-80GB:1 \
  --cpus-per-task=8 --mem=64G --no-requeue \
  --chdir="$batch_root" \
  --output="$batch_root/_job/slurm/repair_celeba64_%A_%a.out" \
  --error="$batch_root/_job/slurm/repair_celeba64_%A_%a.err" \
  "$batch_root/_job/repair_celeba64_finalization.sbatch" "$batch_root"
```

Expected result: each repair task writes `status\tcomplete`, `ACCEPTED.json`,
and `RETENTION_COMPLETE.json` without regenerating checkpoint samples.
