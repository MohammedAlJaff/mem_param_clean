#!/bin/bash
# The researcher launches this script. All six matched tasks run concurrently.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then preview=true; shift; fi
if (( $# )); then
    echo "Usage: bash submit.sh [--dry-run]" >&2
    exit 2
fi
account="${MP_EPS_1M_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS_1M_TIME:-48:00:00}"
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/eps_recovery_multiseed/${tag}"
command=(
    sbatch --account="$account" --time="$walltime" --array=0-5
    --chdir="$repo_root"
    --output="${batch_root}/slurm/%x_%A_%a.out"
    --error="${batch_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch" "$repo_root" "$batch_root"
)
printf 'Six concurrent 1M-update tasks: seeds 1/2 x eps/eps, v/v, x/x.\n'
printf 'Batch root: %s\n' "$batch_root"
cat "${script_dir}/variants.tsv"
printf 'Submission command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then exit 0; fi
mkdir -p "${batch_root}/slurm" "${batch_root}/commands"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "${repo_root}/configs/runners/celeba32_eps_recovery_1m.py" \
    "${batch_root}/run_config.py"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
cp "${script_dir}/run_array.sbatch" "${batch_root}/run_array.sbatch"
cp "${script_dir}/submit.sh" "${batch_root}/submit.sh"
cp "${script_dir}/README.md" "${batch_root}/README.md"
git -C "$repo_root" diff HEAD -- mem_param_src configs/runners/celeba32_eps_recovery_1m.py \
    > "${batch_root}/source.diff"
"${command[@]}"
