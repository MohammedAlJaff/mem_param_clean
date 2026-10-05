#!/bin/bash
# Freeze and submit nine fresh N=2048 native-v/v weighting controls.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: bash submit.sh [--dry-run]" >&2
    exit 2
fi

config_path="${repo_root}/configs/runners/celeba32_unet_vv_weighting_n2048_1m.py"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
PYTHONPATH="$repo_root" python "${script_dir}/validate_suite.py" \
    "${script_dir}/variants.tsv" "$config_path"
python "${script_dir}/validate_array_spec.py" 0-8 --require-full >/dev/null

data_root="${MP_CELEBA32_VV_WEIGHT_N2048_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}; refusing GPU submission." >&2
    exit 2
fi

account="${MP_CELEBA32_VV_WEIGHT_N2048_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_CELEBA32_VV_WEIGHT_N2048_TIME:-24:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/celeba32_unet_vv_weighting_n2048_1m/${batch_tag}"
command=(
    sbatch --parsable --account="$account" --time="$walltime"
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=4 --mem=32G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --array=0-8
    --output="${batch_root}/slurm/initial_%A_%a.out"
    --error="${batch_root}/slurm/initial_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" 1000000
)

printf 'Fresh CelebA32 U-Net-S N=2048 native-v/v controls: nine trajectories.\n'
printf 'N=2048; seeds={0,1,2}; weights={x_equivalent,unit,eps_equivalent}.\n'
printf 'Unthrottled 0-8 array: thin A100-40GB, 24 hours, direct target 1M.\n'
printf 'Physical/effective batch 256/256; accumulation one.\n'
printf 'Training exact; sampling safe floor 0.05, via-x, full Heun.\n'
printf 'W&B project: celeba32-unet-vv-weighting-1m\n'
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
cp "${script_dir}/README.md" "${script_dir}/assert_frozen_import.py" \
    "${script_dir}/validate_array_spec.py" "${script_dir}/validate_suite.py" \
    "${script_dir}/validate_resume_selection.py" \
    "${script_dir}/verify_complete.py" "${script_dir}/verify_complete.sbatch" \
    "${script_dir}/verify.sh" "${script_dir}/submit.sh" \
    "${script_dir}/resume.sh" "${script_dir}/run_array.sbatch" \
    "${script_dir}/resume_array.sbatch" "${batch_root}/job/"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" status --short > "${batch_root}/source_status.txt"
git -C "$repo_root" diff HEAD -- mem_param_src "$config_path" \
    > "${batch_root}/source.diff"
mamba list --explicit > "${batch_root}/environment_explicit.txt"
(cd "$batch_root" && find source job -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256 && \
    sha256sum run_config.py variants.tsv source_commit.txt source_status.txt \
    source.diff environment_explicit.txt >> snapshot.sha256)

job_id=$("${command[@]}")
job_id="${job_id%%;*}"
printf 'initial\t%s\t0-8\t1000000\t%s\n' "$job_id" "$walltime" \
    > "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
cat "${batch_root}/submission.tsv"
printf 'Frozen batch: %s\n' "$batch_root"
