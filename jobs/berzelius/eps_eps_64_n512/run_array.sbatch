#!/bin/bash
#SBATCH --job-name=eps64_n512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
stop_step="${3:?submit.sh must pass the planned stop step}"
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
selected=$(awk -F '\t' -v id="$profile_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/validation/selected_batches.tsv")
if [[ -z "$selected" ]]; then
    echo "No selected physical batch for profile ${profile_id}." >&2
    exit 2
fi
IFS=$'\t' read -r selected_id selected_capacity selected_attention selected_constraint physical_batch effective_batch accumulation_steps peak_reserved seconds_per_step estimated_hours <<< "$selected"
if [[ "$selected_capacity" != "$capacity" || "$selected_attention" != "$attention" || "$selected_constraint" != "$constraint" ]]; then
    echo "Selected batch metadata does not match task ${task_id}." >&2
    exit 2
fi

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
data_root="${MP_EPS64_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
run_dir="${batch_root}/main/runs/${task_id}_${label}_s0"
run_name="${label}-N512-B${physical_batch}-Beff256-eps_eps-gate400k-target1m-s0"
group="eps-eps-64-n512-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
    --batch-size "$physical_batch" --effective-batch-size "$effective_batch"
    --set "data.root=${data_root}"
    --set "data.dataset=${dataset}"
    --set "study.dataset_key=${dataset_key}"
    --set "data.train_split=${train_split}"
    --set "data.reference_split=${reference_split}"
    --set "data.image_size=${image_size}"
    --set "data.resize_size=${image_size}"
    --set "data.image_channels=${channels}"
    --set "network.base_channels=${base_channels}"
    --set "network.channel_mult=${channel_mult}"
    --set "network.time_emb_dim=${time_emb_dim}"
    --set "network.middle_attention=${middle_attention}"
    --set "network.attn_resolutions=${attn_resolutions}"
    --set 'tracking.wandb.project=eps-eps-64-n512'
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf 'Task %s: %s; B=%s, Beff=%s, accumulation=%s, stop=%s.\n' \
    "$task_id" "$label" "$physical_batch" "$effective_batch" "$accumulation_steps" "$stop_step"
printf '%q ' "${command[@]}" > "${batch_root}/commands/main_${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/main_${task_id}_${label}.sh"
exec "${command[@]}"
