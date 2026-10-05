#!/bin/bash
# Submit the strict completion verifier after all training tasks finish.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
if (( $# != 1 )); then
    echo "Usage: bash verify.sh BATCH_DIRECTORY" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
account="${MP_CELEBA32_3X3_ACCOUNT:-berzelius-2026-269}"
job_id=$(sbatch --parsable --account="$account" --time=00:30:00 \
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=1 --mem=4G \
    --no-requeue --chdir="$repo_root" \
    --output="${batch_root}/slurm/verify_%j.out" \
    --error="${batch_root}/slurm/verify_%j.err" \
    "${batch_root}/job/verify_complete.sbatch" "$batch_root")
job_id="${job_id%%;*}"
printf 'Submitted completion verifier %s.\n' "$job_id"
