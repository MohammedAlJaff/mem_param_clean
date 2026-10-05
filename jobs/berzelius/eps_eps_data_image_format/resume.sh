#!/bin/bash
# Resume selected interrupted or paused tasks from their newest recovery states.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
if (( $# < 2 || $# > 3 )); then
    echo "Usage: bash resume.sh BATCH_DIRECTORY ARRAY_SPEC [STOP_STEP]" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
array_spec="$2"
stop_step="${3:-200000}"
if [[ ! "$stop_step" =~ ^[0-9]+$ ]] || (( stop_step < 1 || stop_step > 1000000 )); then
    echo "STOP_STEP must be an integer from 1 through 1000000." >&2
    exit 2
fi
account="${MP_DATA_FORMAT_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_DATA_FORMAT_TIME:-48:00:00}"

sbatch --account="$account" --time="$walltime" --constraint=fat \
    --no-requeue --signal=B:USR1@300 --chdir="$repo_root" \
    --array="$array_spec" \
    --output="${batch_root}/slurm/resume_%x_%A_%a.out" \
    --error="${batch_root}/slurm/resume_%x_%A_%a.err" \
    "${script_dir}/resume_array.sbatch" "$repo_root" "$batch_root" "$stop_step"
