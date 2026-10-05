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
    echo "Usage: bash submit.sh [--dry-run] [array, e.g. 0-2]" >&2
    exit 2
fi
array_spec="${1:-0-2}"
if [[ ! "$array_spec" =~ ^[0-2](-[0-2])?(,[0-2](-[0-2])?)*$ ]]; then
    echo "Choose task IDs 0 through 2." >&2
    exit 2
fi

account="${MP_DIT_DIAG_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_DIT_DIAG_TIME:-48:00:00}"
dependency_args=()
if [[ -n "${MP_DIT_DIAG_DEPENDENCY:-}" ]]; then
    dependency_args=(--dependency="$MP_DIT_DIAG_DEPENDENCY")
fi
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/celeba32_dit_diagonal_100k_viax/${batch_tag}"
command=(
    sbatch --account="$account" --time="$walltime" --constraint=thin
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    "${dependency_args[@]}"
    --array="$array_spec"
    --output="${batch_root}/slurm/%x_%A_%a.out"
    --error="${batch_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch" "$repo_root" "$batch_root"
)

printf 'CelebA gray 32 Small-DiT diagonal matrix. N=256, physical batch=64, effective batch=256.\n'
printf 'Each task trains for 100000 optimizer updates; seed=0; sampling eps_to_v=via_x.\n'
printf 'Batch directory: %s\n' "$batch_root"
cat "${script_dir}/variants.tsv"
printf 'Submission command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/slurm" "${batch_root}/commands" \
    "${batch_root}/source" "${batch_root}/job_spec"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "${repo_root}/configs/runners/celeba32_dit_diagonal_100k_viax.py" \
    "${batch_root}/run_config.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
cp "${script_dir}/README.md" "${script_dir}/submit.sh" \
    "${script_dir}/run_array.sbatch" "${script_dir}/resume.sh" \
    "${script_dir}/resume_array.sbatch" "${batch_root}/job_spec/"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src \
    configs/runners/celeba32_dit_diagonal_100k_viax.py \
    > "${batch_root}/source.diff"
(cd "$batch_root" && find source job_spec -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256)
"${command[@]}"
