#!/bin/bash
# Submit the guarded 16-thin plus 8-fat main study, without a throttle.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 1 )); then
    echo "Usage: bash submit.sh [--dry-run] BATCH_DIRECTORY" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
if [[ ! -f "${batch_root}/SMOKE_COMPLETE" ]]; then
    echo "The six W&B smokes and exact-resume comparison must pass first." >&2
    exit 2
fi
if [[ -e "${batch_root}/MAIN_SUBMITTED" ]]; then
    echo "The main study has already been submitted from this frozen batch." >&2
    exit 2
fi
submission_path="${batch_root}/main/submission.tsv"
if [[ -e "$submission_path" ]]; then
    echo "A prior main submission is complete or partial: ${submission_path}" >&2
    echo "Inspect recorded job IDs and Slurm state. Do not resubmit blindly." >&2
    exit 2
fi
launch_mode=$(<"${batch_root}/validation/launch_mode.txt")
if [[ "$launch_mode" != "direct" && "$launch_mode" != "split" ]]; then
    echo "Unknown launch mode: ${launch_mode}" >&2
    exit 2
fi

data_root="${MP_EPS64_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
for required in celeba/img_align_celeba MNIST/raw flowers-102/jpg; do
    if [[ ! -d "${data_root}/${required}" ]]; then
        echo "Missing dataset cache ${data_root}/${required}; refusing GPU submission." >&2
        exit 2
    fi
done

account="${MP_EPS64_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS64_TIME:-48:00:00}"
dependency_args=()
if [[ -n "${MP_EPS64_DEPENDENCY:-}" ]]; then
    dependency_args=(--dependency="$MP_EPS64_DEPENDENCY")
fi
first_stop=400000
if [[ "$launch_mode" == "split" ]]; then
    first_stop=200000
fi
thin=(sbatch --parsable --account="$account" --time="$walltime" --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1 --mem=64G --no-requeue --signal=B:USR1@300
    --chdir="$repo_root" "${dependency_args[@]}" --export=ALL,MP_EPS64_RUN_CLASS=thin
    --array=0-15 --output="${batch_root}/main/slurm/thin_%A_%a.out"
    --error="${batch_root}/main/slurm/thin_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" "$first_stop")
fat=(sbatch --parsable --account="$account" --time="$walltime" --constraint=fat
    --gres=gpu:A100-SXM4-80GB:1 --mem=128G --no-requeue --signal=B:USR1@300
    --chdir="$repo_root" "${dependency_args[@]}" --export=ALL,MP_EPS64_RUN_CLASS=fat
    --array=16-23 --output="${batch_root}/main/slurm/fat_%A_%a.out"
    --error="${batch_root}/main/slurm/fat_%A_%a.err"
    "${batch_root}/job/run_array.sbatch" "$repo_root" "$batch_root" "$first_stop")

printf 'Launch mode: %s; first stop: %s.\n' "$launch_mode" "$first_stop"
printf 'Thin main command: '; printf '%q ' "${thin[@]}"; printf '\n'
printf 'Fat main command: '; printf '%q ' "${fat[@]}"; printf '\n'
if [[ "$launch_mode" == "split" ]]; then
    printf 'Each matching task resumes 200k to 400k through aftercorr.\n'
fi
if [[ "$preview" == true ]]; then
    exit 0
fi

thin_job=$("${thin[@]}")
thin_job="${thin_job%%;*}"
printf 'stage1_thin\t%s\n' "$thin_job" > "$submission_path"
fat_job=$("${fat[@]}")
fat_job="${fat_job%%;*}"
printf 'stage1_fat\t%s\n' "$fat_job" >> "$submission_path"
final_thin_job="$thin_job"
final_fat_job="$fat_job"
if [[ "$launch_mode" == "split" ]]; then
    thin_resume=$(sbatch --parsable --account="$account" --time="$walltime" \
        --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --mem=64G --no-requeue \
        --signal=B:USR1@300 --chdir="$repo_root" --dependency="aftercorr:${thin_job}" \
        --export=ALL,MP_EPS64_RUN_CLASS=thin --array=0-15 \
        --output="${batch_root}/main/slurm/thin_resume_%A_%a.out" \
        --error="${batch_root}/main/slurm/thin_resume_%A_%a.err" \
        "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" 400000)
    thin_resume="${thin_resume%%;*}"
    printf 'stage2_thin\t%s\n' "$thin_resume" >> "$submission_path"
    fat_resume=$(sbatch --parsable --account="$account" --time="$walltime" \
        --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --mem=128G --no-requeue \
        --signal=B:USR1@300 --chdir="$repo_root" --dependency="aftercorr:${fat_job}" \
        --export=ALL,MP_EPS64_RUN_CLASS=fat --array=16-23 \
        --output="${batch_root}/main/slurm/fat_resume_%A_%a.out" \
        --error="${batch_root}/main/slurm/fat_resume_%A_%a.err" \
        "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" 400000)
    fat_resume="${fat_resume%%;*}"
    printf 'stage2_fat\t%s\n' "$fat_resume" >> "$submission_path"
    final_thin_job="$thin_resume"
    final_fat_job="$fat_resume"
fi
gate_job=$(sbatch --parsable --account="$account" --time=00:20:00 \
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --mem=4G --no-requeue \
    --chdir="$repo_root" \
    --dependency="afterok:${final_thin_job}:${final_fat_job}" \
    --output="${batch_root}/main/slurm/gate_verify_%j.out" \
    --error="${batch_root}/main/slurm/gate_verify_%j.err" \
    "${batch_root}/job/verify_gate.sbatch" "$repo_root" "$batch_root")
gate_job="${gate_job%%;*}"
printf 'gate_verify\t%s\n' "$gate_job" >> "$submission_path"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/MAIN_SUBMITTED"
cat "$submission_path"
