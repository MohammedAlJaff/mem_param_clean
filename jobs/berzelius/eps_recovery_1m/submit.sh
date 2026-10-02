#!/bin/bash
# Submit only when the researcher executes this file. --dry-run prints only.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# > 1 )); then
    echo "Usage: bash submit.sh [--dry-run] [array, e.g. 0 or 0-7]" >&2
    exit 2
fi
array_spec="${1:-0-7}"
if [[ ! "$array_spec" =~ ^[0-7](-[0-7])?(,[0-7](-[0-7])?)*(%[1-8])?$ ]]; then
    echo "Choose task IDs 0 through 7, optionally with a concurrency limit." >&2
    exit 2
fi

# This account comes from the existing Berzelius jobs. Override if needed.
account="${MP_EPS_1M_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS_1M_TIME:-48:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/eps_recovery_1m/${batch_tag}"
command=(
    sbatch --account="$account" --time="$walltime"
    --chdir="$repo_root" --array="$array_spec"
    --output="${batch_root}/slurm/%x_%A_%a.out"
    --error="${batch_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch" "$repo_root" "$batch_root"
)

printf 'CelebA32, N=batch=256, seed=0, 1000000 updates per task.
'
printf 'Selected task IDs: %s
' "$array_spec"
printf 'Batch directory: %s
' "$batch_root"
cat "${script_dir}/variants.tsv"
printf 'Submission command: '
printf '%q ' "${command[@]}"
printf '
'
if [[ "$preview" == true ]]; then
    exit 0
fi

# Slurm needs its log directory before the worker starts.
mkdir -p "${batch_root}/slurm" "${batch_root}/commands"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "${repo_root}/configs/runners/celeba32_eps_recovery_1m.py"     "${batch_root}/run_config.py"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src > "${batch_root}/source.diff"
"${command[@]}"
printf 'Inspect the Slurm logs and saved W&B URL in each run.json under %s/runs.
' "$batch_root"
