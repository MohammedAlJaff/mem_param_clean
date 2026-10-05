#!/bin/bash
#SBATCH --job-name=eps_data_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
repo_root="${1:?resume.sh must pass the repository root}"
batch_root="${2:?resume.sh must pass the batch directory}"
stop_step="${3:?resume.sh must pass the requested stop step}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$repo_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/variants.tsv")
if [[ -z "$row" ]]; then
    echo "No data/image-format row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r row_id label rest <<< "$row"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1

run_dir="${batch_root}/runs/${task_id}_${label}_s0"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --resume-run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
)
printf '%q ' "${command[@]}" > "${batch_root}/commands/resume_${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/resume_${task_id}_${label}.sh"
exec "${command[@]}"
