#!/bin/bash
# Freeze and submit the unthrottled 36-task production matrix.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# > 1 )); then
    echo "Usage: bash submit.sh [--dry-run] [ARRAY_SPEC]" >&2
    exit 2
fi
array_spec="${1:-0-35}"
python "${script_dir}/validate_array_spec.py" "$array_spec" >/dev/null

config_path="${repo_root}/configs/runners/celeba32_dit_3x3_nscale_1m_viax.py"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
PYTHONPATH="$repo_root" python "${script_dir}/validate_suite.py" \
    "${script_dir}/variants.tsv" "$config_path"

data_root="${MP_DIT_3X3_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}; refusing GPU submission." >&2
    exit 2
fi

account="${MP_DIT_3X3_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_DIT_3X3_TIME:-24:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/celeba32_dit_3x3_nscale_1m_viax/${batch_tag}"
command=(
    sbatch --parsable --account="$account" --time="$walltime"
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --mem=32G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --array="$array_spec"
    --output="${batch_root}/slurm/initial_%A_%a.out"
    --error="${batch_root}/slurm/initial_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root"
)

printf 'Small-DiT CelebA32-gray production matrix: 36 tasks, seed 0.\n'
printf 'N={512,1024,2048,4096}; full prediction/loss 3x3; target=1M.\n'
printf 'Each process receives 24 hours and saves exact state on USR1.\n'
printf 'Batch directory: %s\n' "$batch_root"
printf 'Submission command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/slurm" "${batch_root}/commands" \
    "${batch_root}/source" "${batch_root}/job"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "$config_path" "${batch_root}/run_config.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
cp "${script_dir}/README.md" "${script_dir}/validate_array_spec.py" \
    "${script_dir}/validate_suite.py" "${script_dir}/assert_frozen_import.py" \
    "${script_dir}/submit.sh" "${script_dir}/run_array.sbatch" \
    "${script_dir}/resume.sh" "${script_dir}/resume_array.sbatch" \
    "${batch_root}/job/"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" status --short > "${batch_root}/source_status.txt"
git -C "$repo_root" diff HEAD -- mem_param_src "$config_path" \
    > "${batch_root}/source.diff"
(cd "$batch_root" && find source job -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256 && \
    sha256sum run_config.py variants.tsv source_commit.txt source_status.txt \
    source.diff >> snapshot.sha256)

job_id=$("${command[@]}")
job_id="${job_id%%;*}"
printf 'initial\t%s\t%s\t%s\n' "$job_id" "$array_spec" "$walltime" \
    > "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
printf 'Submitted batch job %s\n' "$job_id"
printf 'Frozen batch: %s\n' "$batch_root"
