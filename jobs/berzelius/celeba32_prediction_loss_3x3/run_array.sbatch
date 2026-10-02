#!/bin/bash
#SBATCH --job-name=pred_loss_3x3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$repo_root"

found=false
while IFS=$'\t' read -r row_id label prediction loss; do
    if [[ "$row_id" == "$task_id" ]]; then
        found=true
        break
    fi
done < "${batch_root}/variants.tsv"
if [[ "$found" != true ]]; then
    echo "No 3x3 row for task ${task_id}." >&2
    exit 2
fi

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1

data_root="${MP_PRED_LOSS_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}." >&2
    exit 2
fi

run_dir="${batch_root}/runs/${task_id}_${label}_s0"
run_name="celeba32-gray-N256-Beff256-${prediction}_${loss}-gate200k-target1m-s0"
group="celeba32-pred-loss-3x3-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --stop-after-step 200000
    --set "data.root=${data_root}"
    --set "framework.prediction_space=${prediction}"
    --set "framework.loss_space=${loss}"
    --set "tracking.wandb.project=celeba32-pred-loss-grid"
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf 'Task %s: %s\n' "$task_id" "$label"
printf '%q ' "${command[@]}" > "${batch_root}/commands/${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/${task_id}_${label}.sh"
exec "${command[@]}"
