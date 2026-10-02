#!/bin/bash
# Submit only when the researcher executes this file. --dry-run writes nothing.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# > 1 )); then
    echo "Usage: bash submit.sh [--dry-run] [array, e.g. 0-8]" >&2
    exit 2
fi
array_spec="${1:-0-8}"
if [[ ! "$array_spec" =~ ^[0-8](-[0-8])?(,[0-8](-[0-8])?)*$ ]]; then
    echo "Choose task IDs 0 through 8." >&2
    exit 2
fi

account="${MP_PRED_LOSS_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_PRED_LOSS_TIME:-48:00:00}"
dependency_args=()
if [[ -n "${MP_PRED_LOSS_DEPENDENCY:-}" ]]; then
    dependency_args=(--dependency="$MP_PRED_LOSS_DEPENDENCY")
fi
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/celeba32_prediction_loss_3x3/${batch_tag}"
command=(
    sbatch --account="$account" --time="$walltime" --constraint=thin
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    "${dependency_args[@]}"
    --array="$array_spec"
    --output="${batch_root}/slurm/%x_%A_%a.out"
    --error="${batch_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch" "$repo_root" "$batch_root"
)

printf 'CelebA gray 32 prediction/loss 3x3. N=256, effective batch=256.\n'
printf 'Each task targets 1M updates and pauses at 200k.\n'
printf 'Batch directory: %s\n' "$batch_root"
cat "${script_dir}/variants.tsv"
printf 'Submission command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/slurm" "${batch_root}/commands" "${batch_root}/source"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "${repo_root}/configs/runners/celeba32_pred_loss_grid_1m.py" "${batch_root}/run_config.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src configs/runners > "${batch_root}/source.diff"
"${command[@]}"
