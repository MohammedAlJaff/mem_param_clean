#!/bin/bash
# Manually resume selected interrupted tasks from their newest exact states.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 4 )); then
    echo "Usage: bash resume.sh [--dry-run] BATCH_DIRECTORY ARRAY_SPEC STOP_STEP thin|fat" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
array_spec="$2"
stop_step="$3"
run_class="$4"
python "${batch_root}/job/validate_array_spec.py" "$array_spec" "$run_class" >/dev/null
if [[ ! "$stop_step" =~ ^[0-9]+$ ]] || (( stop_step < 1 || stop_step > 1000000 )); then
    echo "STOP_STEP must be an integer from 1 through 1000000." >&2
    exit 2
fi
if [[ ! -s "${batch_root}/submission.tsv" ]]; then
    echo "No accepted Slurm submission is recorded for this batch." >&2
    exit 2
fi

account="${MP_MNIST_ACCOUNT:-berzelius-2026-269}"
if [[ "$run_class" == "thin" ]]; then
    walltime="${MP_MNIST_THIN_TIME:-36:00:00}"
    constraint=thin
    gres=gpu:A100-SXM4-40GB:1
    cpus=4
    memory=32G
else
    walltime="${MP_MNIST_FAT_TIME:-30:00:00}"
    constraint=fat
    gres=gpu:A100-SXM4-80GB:1
    cpus=8
    memory=128G
fi
command=(
    sbatch --parsable --account="$account" --time="$walltime"
    --constraint="$constraint" --gres="$gres" --cpus-per-task="$cpus" --mem="$memory"
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --export="ALL,MP_MNIST_RUN_CLASS=${run_class}" --array="$array_spec"
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
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_id" "$run_class" \
    "$array_spec" "$stop_step" "$walltime" >> "${batch_root}/recovery_jobs.tsv"
printf 'Submitted resume job %s\n' "$job_id"
