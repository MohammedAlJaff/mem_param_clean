#!/bin/bash
#SBATCH --job-name=eps64_n512_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
repo_root="${1:?resume.sh must pass the repository root}"
batch_root="${2:?resume.sh must pass the batch directory}"
stop_step="${3:?resume.sh must pass the planned stop step}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$batch_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/variants.tsv")
if [[ -z "$row" ]]; then
    echo "No EPS/EPS 64x64 row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r row_id label format dataset dataset_key train_split reference_split image_size channels capacity profile_id base_channels channel_mult time_emb_dim attention middle_attention attn_resolutions constraint <<< "$row"
if [[ "$constraint" != "${MP_EPS64_RUN_CLASS:?submitter must set MP_EPS64_RUN_CLASS}" ]]; then
    echo "Task ${task_id} requires ${constraint}, not ${MP_EPS64_RUN_CLASS}." >&2
    exit 2
fi

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
run_dir="${batch_root}/main/runs/${task_id}_${label}_s0"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --resume-run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
)
printf '%q ' "${command[@]}" > "${batch_root}/commands/resume_${task_id}_${label}_to_${stop_step}.sh"
printf '\n' >> "${batch_root}/commands/resume_${task_id}_${label}_to_${stop_step}.sh"
exec "${command[@]}"
