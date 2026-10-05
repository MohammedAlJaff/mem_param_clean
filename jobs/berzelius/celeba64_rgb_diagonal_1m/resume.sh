#!/bin/bash
# Manually resume selected CelebA64 RGB tasks from their newest exact states.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 3 )); then
    echo "Usage: bash resume.sh [--dry-run] BATCH_DIRECTORY ARRAY_SPEC STOP_STEP" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
array_spec="$2"
stop_step="$3"
python "${batch_root}/job/validate_array_spec.py" "$array_spec" >/dev/null
if [[ ! "$stop_step" =~ ^[0-9]+$ ]] || (( stop_step < 1 || stop_step > 1000000 )); then
    echo "STOP_STEP must be an integer from 1 through 1000000." >&2
    exit 2
fi
if [[ ! -s "${batch_root}/submission.tsv" ]]; then
    echo "No accepted Slurm submission is recorded for this batch." >&2
    exit 2
fi

account="${MP_CELEBA64_RGB_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_CELEBA64_RGB_TIME:-30:00:00}"
command=(
    sbatch --parsable --account="$account" --time="$walltime"
    --constraint=fat --gres=gpu:A100-SXM4-80GB:1 --cpus-per-task=8 --mem=128G
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --array="$array_spec"
    --output="${batch_root}/slurm/manual_resume_%A_%a.out"
    --error="${batch_root}/slurm/manual_resume_%A_%a.err"
    "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step"
)
printf 'Resume command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

job_id=$("${command[@]}")
job_id="${job_id%%;*}"
printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_id" \
    "$array_spec" "$stop_step" "$walltime" >> "${batch_root}/recovery_jobs.tsv"
printf 'Submitted exact resume job %s for tasks %s to step %s.\n' \
    "$job_id" "$array_spec" "$stop_step"
