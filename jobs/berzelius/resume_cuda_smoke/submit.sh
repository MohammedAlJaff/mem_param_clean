#!/bin/bash
# Submit one exact CUDA resume comparison.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
account="${MP_RESUME_SMOKE_ACCOUNT:-berzelius-2026-269}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/resume_cuda_smoke/${batch_tag}"

mkdir -p "${batch_root}/source"
cp "${repo_root}/configs/runners/resume_cuda_smoke.py" "${batch_root}/run_config.py"
cp "${script_dir}/compare.py" "${batch_root}/compare.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src configs/runners > "${batch_root}/source.diff"

sbatch --account="$account" --time=00:30:00 --constraint=thin \
    --no-requeue --chdir="$repo_root" \
    --output="${batch_root}/slurm.out" --error="${batch_root}/slurm.err" \
    "${script_dir}/run.sbatch" "$repo_root" "$batch_root"
printf 'Batch directory: %s\n' "$batch_root"
