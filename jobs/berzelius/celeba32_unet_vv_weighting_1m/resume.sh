#!/bin/bash
# Resume selected interrupted weighted-v/v tasks from exact saved state.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
if (( $# < 2 || $# > 3 )); then
    echo "Usage: bash resume.sh BATCH_DIRECTORY ARRAY_SPEC [STOP_STEP]" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
array_spec="$2"
stop_step="${3:-1000000}"
if [[ ! "$stop_step" =~ ^[0-9]+$ ]] || (( stop_step < 1 || stop_step > 1000000 )); then
    echo "STOP_STEP must be an integer from 1 through 1000000." >&2
    exit 2
fi
python "${batch_root}/job/validate_array_spec.py" "$array_spec" >/dev/null
python "${batch_root}/job/validate_resume_selection.py" "$batch_root" "$array_spec"

account="${MP_CELEBA32_VV_WEIGHT_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_CELEBA32_VV_WEIGHT_TIME:-24:00:00}"
job_id=$(sbatch --parsable --account="$account" --time="$walltime" \
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --cpus-per-task=4 --mem=32G \
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root" \
    --array="$array_spec" \
    --output="${batch_root}/slurm/resume_%A_%a.out" \
    --error="${batch_root}/slurm/resume_%A_%a.err" \
    "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step")
job_id="${job_id%%;*}"
printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$job_id" "$array_spec" "$stop_step" >> "${batch_root}/recovery_jobs.tsv"
printf 'Submitted exact resume array %s for tasks %s to step %s.\n' \
    "$job_id" "$array_spec" "$stop_step"
