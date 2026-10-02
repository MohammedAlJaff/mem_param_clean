#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# > 1 )); then
    echo "Usage: bash submit.sh [--dry-run] [array, e.g. 0 or 0-4]" >&2
    exit 2
fi

array_spec="${1:-0-4}"
if [[ ! "$array_spec" =~ ^[0-4](-[0-4])?(,[0-4](-[0-4])?)*(%[1-5])?$ ]]; then
    echo "Choose task IDs 0 through 4, optionally with a concurrency limit." >&2
    exit 2
fi

account="${MP_SMOKE_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_SMOKE_TIME:-01:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="$repo_root/outputs/job_sets/celeba32_n16_10k_sampler_smoke/$batch_tag"
command=(
    sbatch --account="$account" --time="$walltime"
    --chdir="$repo_root" --array="$array_spec"
    --output="$batch_root/slurm/%x_%A_%a.out"
    --error="$batch_root/slurm/%x_%A_%a.err"
    "$script_dir/run_array.sbatch" "$repo_root" "$batch_root"
)

printf 'CelebA32, N=batch=16, seed=0, 10000 updates per task.\n'
printf 'Fixed: Adam lr=5e-4, Heun-50, final Euler, base32_m124 UNet.\n'
printf 'Selected task IDs: %s\n' "$array_spec"
printf 'Batch directory: %s\n' "$batch_root"
cat "$script_dir/variants.tsv"
printf 'Submission command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "$batch_root/slurm" "$batch_root/commands"
cp "$script_dir/variants.tsv" "$batch_root/variants.tsv"
cp "$repo_root/configs/runners/celeba32_n16_10k_sampler_smoke.py" "$batch_root/smoke_config.py"
git -C "$repo_root" rev-parse HEAD > "$batch_root/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src > "$batch_root/source.diff"
"${command[@]}"
printf 'Inspect Slurm logs and W&B URLs in %s/runs/*/run.json\n' "$batch_root"
