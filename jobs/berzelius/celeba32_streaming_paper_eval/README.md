# CelebA32 streaming paper evaluation

This launcher replaces the cancelled CelebA32 evaluation arrays. It evaluates
all 144 completed one-million-step CelebA32 runs with the complete metric suite
and 10,000 generated samples at every saved checkpoint:

- 108 CelebA32 U-Net-S runs across seeds 0, 1, and 2;
- 36 CelebA32 Small-DiT seed-0 runs.

The separate `--dit-seed1` selection evaluates only the 18 completed Small-DiT
seed-1 runs: the full prediction/loss 3x3 matrix at N=512 and N=2048.
It does not resubmit the 144 runs above. All runs have 72 checkpoints.

## Execution and retention

The four 36-task arrays submit independently, without Slurm dependencies or
array throttles. The seed-1 selection submits one independent 18-task array.
Each task requests one A100-40GB, eight CPUs, 64 GB RAM, and 12 hours.
Workers stay in the frozen batch and verify that evaluator imports come from
its source snapshot. Each checkpoint is identity-recorded immediately after its
metrics complete. Dense feature tensors are then deleted, along with the sample
tensor unless the checkpoint is needed for a sample grid or is the final
checkpoint.

Each successful task keeps all metrics, tables, plots, feature statistics,
nearest-neighbor records, and samples from steps 0, 10k, 100k, and 1M. After
paper-tier acceptance, a final retention pass removes reproducible reference
feature matrices.

## Commands

Preview the submission:

```bash
bash jobs/berzelius/celeba32_streaming_paper_eval/submit.sh --dry-run
```

Submit all four independent arrays:

```bash
bash jobs/berzelius/celeba32_streaming_paper_eval/submit.sh
```

Preview and submit only the 18 seed-1 DiT runs:

```bash
bash jobs/berzelius/celeba32_streaming_paper_eval/submit.sh --dit-seed1 --dry-run
bash jobs/berzelius/celeba32_streaming_paper_eval/submit.sh --dit-seed1
```

The preview reports 18 runs and one independent array. Submission prints the
array job ID and a frozen batch under
`outputs/evals/celeba32_dit_seed1_streaming_full_10000/TIMESTAMP`.
Results appear in `results/dit_seed1/RUN_NAME`. The full suite uses 10,000
generated samples and 512 held-out losses at each of the 72 checkpoints.
Use `--group dit_seed1 --tasks 0-17` to restart this selection.

Restart selected tasks from an existing frozen batch:

```bash
bash jobs/berzelius/celeba32_streaming_paper_eval/submit.sh \
  --resume-root outputs/evals/celeba32_streaming_full_10000/TIMESTAMP \
  --group celeba32_unet_a --tasks 2,7,11
```

The restart uses the frozen source and input hashes. Completed compacted
checkpoints are validated and skipped. If compaction was interrupted after a
checkpoint completed, deletion resumes from its saved manifest without
rebuilding or resampling.
