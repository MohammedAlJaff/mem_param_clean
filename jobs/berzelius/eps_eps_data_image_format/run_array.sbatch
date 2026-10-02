#!/bin/bash
#SBATCH --job-name=eps_data_format
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$repo_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/variants.tsv")
if [[ -z "$row" ]]; then
    echo "No data/image-format row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r row_id label dataset dataset_key train_split reference_split image_size channels channel_mult attn_resolutions <<< "$row"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1

data_root="${MP_DATA_FORMAT_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
run_dir="${batch_root}/runs/${task_id}_${label}_s0"
run_name="${label}-N256-B128-Beff256-eps_eps-gate200k-target1m-s0"
group="eps-eps-data-image-format-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --stop-after-step 200000
    --set "data.root=${data_root}"
    --set "data.dataset=${dataset}"
    --set "study.dataset_key=${dataset_key}"
    --set "data.train_split=${train_split}"
    --set "data.reference_split=${reference_split}"
    --set "data.image_size=${image_size}"
    --set "data.resize_size=${image_size}"
    --set "data.image_channels=${channels}"
    --set "network.channel_mult=${channel_mult}"
    --set "network.attn_resolutions=${attn_resolutions}"
    --set "tracking.wandb.project=eps-eps-data-image-format"
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf 'Task %s: %s\n' "$task_id" "$label"
printf '%q ' "${command[@]}" > "${batch_root}/commands/${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/${task_id}_${label}.sh"
exec "${command[@]}"
