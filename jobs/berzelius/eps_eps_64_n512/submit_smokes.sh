#!/bin/bash
# Submit six W&B integration smokes plus one exact-resume comparison.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 1 )); then
    echo "Usage: bash submit_smokes.sh [--dry-run] BATCH_DIRECTORY" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
if [[ ! -f "${batch_root}/PROBE_COMPLETE" || ! -f "${batch_root}/validation/selected_batches.tsv" ]]; then
    echo "Capacity probes and batch selection must pass first." >&2
    exit 2
fi
if [[ -e "${batch_root}/SMOKE_COMPLETE" ]]; then
    echo "This batch already has completed integration smokes." >&2
    exit 2
fi
jobs_path="${batch_root}/validation/smoke_jobs.tsv"
if [[ -e "$jobs_path" ]]; then
    echo "A prior smoke submission is complete or partial: ${jobs_path}" >&2
    echo "Inspect recorded job IDs and Slurm state. Do not resubmit blindly." >&2
    exit 2
fi

data_root="${MP_EPS64_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}; refusing GPU submission." >&2
    exit 2
fi

account="${MP_EPS64_ACCOUNT:-berzelius-2026-269}"
worker="${batch_root}/job/run_smoke_array.sbatch"
thin=(sbatch --parsable --account="$account" --time=04:00:00 --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1 --mem=64G --no-requeue --chdir="$repo_root"
    --export=ALL,MP_EPS64_SMOKE_CLASS=thin --array=0-3
    --output="${batch_root}/validation/slurm/smoke_thin_%A_%a.out"
    --error="${batch_root}/validation/slurm/smoke_thin_%A_%a.err"
    "$worker" "$repo_root" "$batch_root")
fat=(sbatch --parsable --account="$account" --time=04:00:00 --constraint=fat
    --gres=gpu:A100-SXM4-80GB:1 --mem=128G --no-requeue --chdir="$repo_root"
    --export=ALL,MP_EPS64_SMOKE_CLASS=fat --array=4-5
    --output="${batch_root}/validation/slurm/smoke_fat_%A_%a.out"
    --error="${batch_root}/validation/slurm/smoke_fat_%A_%a.err"
    "$worker" "$repo_root" "$batch_root")
resume=(sbatch --parsable --account="$account" --time=04:00:00 --constraint=fat
    --gres=gpu:A100-SXM4-80GB:1 --mem=128G --no-requeue --chdir="$repo_root"
    --output="${batch_root}/validation/slurm/resume_smoke_%j.out"
    --error="${batch_root}/validation/slurm/resume_smoke_%j.err"
    "${batch_root}/job/run_exact_resume.sbatch" "$repo_root" "$batch_root")

printf 'Thin W&B smoke command: '; printf '%q ' "${thin[@]}"; printf '\n'
printf 'Fat W&B smoke command: '; printf '%q ' "${fat[@]}"; printf '\n'
printf 'Exact-resume command: '; printf '%q ' "${resume[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    printf 'Finalize: afterok on both W&B arrays and exact-resume job.\n'
    exit 0
fi

thin_job=$("${thin[@]}")
thin_job="${thin_job%%;*}"
printf 'thin\t%s\n' "$thin_job" > "$jobs_path"
fat_job=$("${fat[@]}")
fat_job="${fat_job%%;*}"
printf 'fat\t%s\n' "$fat_job" >> "$jobs_path"
resume_job=$("${resume[@]}")
resume_job="${resume_job%%;*}"
printf 'exact_resume\t%s\n' "$resume_job" >> "$jobs_path"
final_job=$(sbatch --parsable --account="$account" --time=00:10:00 --constraint=thin \
    --gres=gpu:A100-SXM4-40GB:1 --mem=2G --no-requeue \
    --dependency="afterok:${thin_job}:${fat_job}:${resume_job}" --chdir="$repo_root" \
    --output="${batch_root}/validation/slurm/smoke_finalize_%j.out" \
    --error="${batch_root}/validation/slurm/smoke_finalize_%j.err" \
    "${batch_root}/job/finalize_smokes.sbatch" "$repo_root" "$batch_root")
final_job="${final_job%%;*}"
printf 'finalize\t%s\n' "$final_job" >> "$jobs_path"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SMOKE_SUBMITTED"
printf 'Thin W&B array: %s\nFat W&B array: %s\nExact resume: %s\nFinalize: %s\n' \
    "$thin_job" "$fat_job" "$resume_job" "$final_job"
