#!/bin/bash
#SBATCH --job-name=mnist_diag_1m
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the frozen batch directory}"
stop_step="${3:?submit.sh must pass the planned stop step}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$batch_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/variants.tsv")
if [[ -z "$row" ]]; then
    echo "No MNIST production row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r row_id label image_size model_label n_train seed prediction loss base_channels channel_mult middle_attention attn_resolutions constraint <<< "$row"
if [[ "$constraint" != "${MP_MNIST_RUN_CLASS:?submitter must set MP_MNIST_RUN_CLASS}" ]]; then
    echo "Task ${task_id} requires ${constraint}, not ${MP_MNIST_RUN_CLASS}." >&2
    exit 2
fi

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"

data_root="${MP_MNIST_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/MNIST/raw" ]]; then
    echo "Missing MNIST cache under ${data_root}; refusing training." >&2
    exit 2
fi

run_dir="${batch_root}/runs/${task_id}_${label}"
run_name="${label}-B256-Beff256-${prediction}_${loss}-1m"
group="mnist${image_size}-${model_label}-1m-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
    --batch-size 256 --effective-batch-size 256
    --set "data.root=${data_root}"
    --set "data.n_train=${n_train}"
    --set "data.image_size=${image_size}"
    --set "data.resize_size=${image_size}"
    --set "training.seed=${seed}"
    --set "framework.prediction_space=${prediction}"
    --set "framework.loss_space=${loss}"
    --set "network.base_channels=${base_channels}"
    --set "network.channel_mult=${channel_mult}"
    --set "network.middle_attention=${middle_attention}"
    --set "network.attn_resolutions=${attn_resolutions}"
    --set 'sampling.eps_to_v=via_x'
    --set 'tracking.wandb.project=mnist-unet-diagonal-1m'
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf 'Task %s: %s; physical/effective batch 256/256; accumulation 1; stop %s.\n' \
    "$task_id" "$label" "$stop_step"
printf '%q ' "${command[@]}" > "${batch_root}/commands/initial_${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/initial_${task_id}_${label}.sh"
exec "${command[@]}"
