#!/bin/bash
#SBATCH --job-name=celeba32_u3x3_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
repo_root="${1:?submitter must pass the repository root}"
batch_root="${2:?submitter must pass the frozen batch directory}"
stop_step="${3:?submitter must pass the requested stop step}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$batch_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/variants.tsv")
if [[ -z "$row" ]]; then
    echo "No CelebA32 production row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r row_id label n_train seed prediction loss <<< "$row"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"

run_dir="${batch_root}/runs/${task_id}_${label}"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --resume-run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
)
command_file="${batch_root}/commands/resume_${SLURM_ARRAY_JOB_ID}_${task_id}_${label}_to_${stop_step}.sh"
printf '%q ' "${command[@]}" > "$command_file"
printf '\n' >> "$command_file"
exec "${command[@]}"
