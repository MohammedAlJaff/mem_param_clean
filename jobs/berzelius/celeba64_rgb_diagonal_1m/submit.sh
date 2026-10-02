#!/bin/bash
# Freeze and submit 18 fresh CelebA64 RGB diagonal trajectories.
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

config_path="${repo_root}/configs/runners/celeba64_rgb_diagonal_1m.py"
stage_stops=(200000 400000 600000 800000 1000000)
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
PYTHONPATH="$repo_root" python "${script_dir}/validate_suite.py" \
    "${script_dir}/variants.tsv" "$config_path"
python "${script_dir}/validate_array_spec.py" 0-17 --require-full >/dev/null
python "${script_dir}/validate_stage_plan.py" "${stage_stops[@]}" >/dev/null

data_root="${MP_CELEBA64_RGB_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}; refusing GPU submission." >&2
    exit 2
fi

account="${MP_CELEBA64_RGB_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_CELEBA64_RGB_TIME:-30:00:00}"
batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/celeba64_rgb_diagonal_1m/${batch_tag}"
initial=(
    sbatch --parsable --account="$account" --time="$walltime"
    --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=128G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --array=0-17
    --output="${batch_root}/slurm/stage1_%A_%a.out"
    --error="${batch_root}/slurm/stage1_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" "${stage_stops[0]}"
)

printf 'Fresh CelebA64 RGB diagonal suite: 18 trajectories, no reuse.\n'
printf 'N={512,2048}; seeds={0,1,2}; targets={x/x,eps/eps,v/v}.\n'
printf 'All tasks: fat A100-80GB, five 200k stages, 30 hours each.\n'
printf 'Physical/effective batch 256/256; accumulation one.\n'
printf 'Training exact; 72 Bonnaire checkpoints; one safe via-x full-Heun monitor.\n'
printf 'W&B project: celeba64-rgb-unet-diagonal-1m\n'
printf 'Batch directory: %s\n' "$batch_root"
printf 'Stage-1 command: '; printf '%q ' "${initial[@]}"; printf '\n'
printf 'Stages 2-5 use matching aftercorr dependencies and stop at 400k, 600k, 800k, and 1M.\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/slurm" "${batch_root}/commands" \
    "${batch_root}/source" "${batch_root}/job"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "$config_path" "${batch_root}/run_config.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
cp "${script_dir}/README.md" "${script_dir}/assert_frozen_import.py" \
    "${script_dir}/validate_array_spec.py" "${script_dir}/validate_stage_plan.py" \
    "${script_dir}/validate_suite.py" "${script_dir}/verify_complete.py" \
    "${script_dir}/verify_complete.sbatch" "${script_dir}/submit.sh" \
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

job_id=$("${initial[@]}")
job_id="${job_id%%;*}"
printf 'stage1\t%s\t0-17\t%s\t%s\n' \
    "$job_id" "${stage_stops[0]}" "$walltime" > "${batch_root}/submission.tsv"

previous_job="$job_id"
stage=2
for stop_step in "${stage_stops[@]:1}"; do
    resume_job=$(sbatch --parsable --account="$account" --time="$walltime" \
        --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=128G \
        --no-requeue --signal=B:USR1@300 --chdir="$repo_root" \
        --dependency="aftercorr:${previous_job}" --array=0-17 \
        --output="${batch_root}/slurm/stage${stage}_%A_%a.out" \
        --error="${batch_root}/slurm/stage${stage}_%A_%a.err" \
        "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step")
    resume_job="${resume_job%%;*}"
    printf 'stage%s\t%s\t0-17\t%s\t%s\n' \
        "$stage" "$resume_job" "$stop_step" "$walltime" \
        >> "${batch_root}/submission.tsv"
    previous_job="$resume_job"
    stage=$((stage + 1))
done

verify_job=$(sbatch --parsable --account="$account" --time=00:30:00 \
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=1 --mem=4G \
    --no-requeue --chdir="$repo_root" \
    --dependency="afterok:${previous_job}" \
    --output="${batch_root}/slurm/verify_%j.out" \
    --error="${batch_root}/slurm/verify_%j.err" \
    "${batch_root}/job/verify_complete.sbatch" "$batch_root")
verify_job="${verify_job%%;*}"
printf 'completion_verifier\t%s\t-\t1000000\t00:30:00\n' "$verify_job" \
    >> "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
cat "${batch_root}/submission.tsv"
printf 'Frozen batch: %s\n' "$batch_root"
