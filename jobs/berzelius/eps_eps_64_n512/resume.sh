#!/bin/bash
# Manually resume selected interrupted tasks, still capped at the 400k gate.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# < 3 || $# > 4 )); then
    echo "Usage: bash resume.sh [--dry-run] BATCH_DIRECTORY thin|fat ARRAY_SPEC [200000|400000]" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
resource_class="$2"
array_spec="$3"
stop_step="${4:-400000}"
if [[ "$resource_class" != "thin" && "$resource_class" != "fat" ]]; then
    echo "Resource class must be thin or fat." >&2
    exit 2
fi
if [[ "$stop_step" != "200000" && "$stop_step" != "400000" ]]; then
    echo "This study permits only the 200k recovery point or 400k gate." >&2
    exit 2
fi
python "${batch_root}/job/validate_array_spec.py" "$resource_class" "$array_spec" >/dev/null
if [[ ! -s "${batch_root}/main/submission.tsv" ]]; then
    echo "No accepted main Slurm job is recorded for this batch." >&2
    exit 2
fi

account="${MP_EPS64_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS64_TIME:-48:00:00}"
if [[ "$resource_class" == "thin" ]]; then
    gres="gpu:A100-SXM4-40GB:1"
    memory="64G"
else
    gres="gpu:A100-SXM4-80GB:1"
    memory="128G"
fi
command=(sbatch --parsable --account="$account" --time="$walltime"
    --constraint="$resource_class" --gres="$gres" --mem="$memory"
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root"
    --export="ALL,MP_EPS64_RUN_CLASS=${resource_class}" --array="$array_spec"
    --output="${batch_root}/main/slurm/manual_resume_%A_%a.out"
    --error="${batch_root}/main/slurm/manual_resume_%A_%a.err"
    "${batch_root}/job/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step")
printf 'Resume command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi
resume_job=$("${command[@]}")
resume_job="${resume_job%%;*}"
printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$resource_class" "$array_spec" "$stop_step" "$resume_job" \
    >> "${batch_root}/main/recovery_jobs.tsv"
printf 'Recovery job: %s\n' "$resume_job"
