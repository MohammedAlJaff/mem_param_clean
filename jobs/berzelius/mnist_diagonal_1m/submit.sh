#!/bin/bash
# Freeze and submit 18 fresh MNIST32 and 18 fresh MNIST64 trajectories.
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

config_path="${repo_root}/configs/runners/mnist_diagonal_1m.py"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
PYTHONPATH="$repo_root" python "${script_dir}/validate_suite.py" \
    "${script_dir}/variants.tsv" "$config_path"
python "${script_dir}/validate_array_spec.py" 0-17 thin >/dev/null
python "${script_dir}/validate_array_spec.py" 18-35 fat >/dev/null

data_root="${MP_MNIST_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/MNIST/raw" ]]; then
    echo "Missing MNIST cache under ${data_root}; refusing GPU submission." >&2
    exit 2
fi

account="${MP_MNIST_ACCOUNT:-berzelius-2026-269}"
thin_time="${MP_MNIST_THIN_TIME:-36:00:00}"
fat_time="${MP_MNIST_FAT_TIME:-30:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/mnist_diagonal_1m/${batch_tag}"

thin=(
    sbatch --parsable --account="$account" --time="$thin_time"
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=4 --mem=32G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --export=ALL,MP_MNIST_RUN_CLASS=thin --array=0-17
    --output="${batch_root}/slurm/thin_initial_%A_%a.out"
    --error="${batch_root}/slurm/thin_initial_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" 1000000
)
fat=(
    sbatch --parsable --account="$account" --time="$fat_time"
    --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=128G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --export=ALL,MP_MNIST_RUN_CLASS=fat --array=18-35
    --output="${batch_root}/slurm/fat_stage1_%A_%a.out"
    --error="${batch_root}/slurm/fat_stage1_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" 200000
)

printf 'Fresh MNIST diagonal suite: 36 trajectories, no reuse.\n'
printf 'MNIST32: 18 thin A100-40GB tasks, 36 hours, direct target 1M.\n'
printf 'MNIST64: 18 fat A100-80GB tasks, five 200k stages, 30 hours each.\n'
printf 'Physical/effective batch is 256/256; accumulation is one.\n'
printf 'W&B project: mnist-unet-diagonal-1m\n'
printf 'Batch directory: %s\n' "$batch_root"
printf 'Thin command: '; printf '%q ' "${thin[@]}"; printf '\n'
printf 'Fat stage-1 command: '; printf '%q ' "${fat[@]}"; printf '\n'
printf 'Fat stages 2-5 use matching aftercorr dependencies and stop at 400k, 600k, 800k, and 1M.\n'
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
    "${script_dir}/verify_complete.py" "${script_dir}/verify_complete.sbatch" \
    "${script_dir}/submit.sh" "${script_dir}/resume.sh" \
    "${script_dir}/run_array.sbatch" "${script_dir}/resume_array.sbatch" \
    "${batch_root}/job/"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" status --short > "${batch_root}/source_status.txt"
git -C "$repo_root" diff HEAD -- mem_param_src "$config_path" \
    > "${batch_root}/source.diff"
mamba list --explicit > "${batch_root}/environment_explicit.txt"
(cd "$batch_root" && find source job -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256 && \
    sha256sum run_config.py variants.tsv source_commit.txt source_status.txt \
    source.diff environment_explicit.txt >> snapshot.sha256)

thin_job=$("${thin[@]}")
thin_job="${thin_job%%;*}"
printf 'thin_initial\t%s\t0-17\t1000000\t%s\n' "$thin_job" "$thin_time" \
    > "${batch_root}/submission.tsv"

fat_job=$("${fat[@]}")
fat_job="${fat_job%%;*}"
printf 'fat_stage1\t%s\t18-35\t200000\t%s\n' "$fat_job" "$fat_time" \
    >> "${batch_root}/submission.tsv"

previous_job="$fat_job"
stage=2
for stop_step in 400000 600000 800000 1000000; do
    resume_job=$(sbatch --parsable --account="$account" --time="$fat_time" \
        --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=128G \
        --no-requeue --signal=B:USR1@300 --chdir="$repo_root" \
        --dependency="aftercorr:${previous_job}" \
        --export=ALL,MP_MNIST_RUN_CLASS=fat --array=18-35 \
        --output="${batch_root}/slurm/fat_stage${stage}_%A_%a.out" \
        --error="${batch_root}/slurm/fat_stage${stage}_%A_%a.err" \
        "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step")
    resume_job="${resume_job%%;*}"
    printf 'fat_stage%s\t%s\t18-35\t%s\t%s\n' \
        "$stage" "$resume_job" "$stop_step" "$fat_time" \
        >> "${batch_root}/submission.tsv"
    previous_job="$resume_job"
    stage=$((stage + 1))
done

verify_job=$(sbatch --parsable --account="$account" --time=00:20:00 \
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=1 --mem=4G \
    --no-requeue --chdir="$repo_root" \
    --dependency="afterok:${thin_job}:${previous_job}" \
    --output="${batch_root}/slurm/verify_%j.out" \
    --error="${batch_root}/slurm/verify_%j.err" \
    "${batch_root}/job/verify_complete.sbatch" "$batch_root")
verify_job="${verify_job%%;*}"
printf 'completion_verifier\t%s\t-\t1000000\t00:20:00\n' "$verify_job" \
    >> "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
cat "${batch_root}/submission.tsv"
printf 'Frozen batch: %s\n' "$batch_root"
