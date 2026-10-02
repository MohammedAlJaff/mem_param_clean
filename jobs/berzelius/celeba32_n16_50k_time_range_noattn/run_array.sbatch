#!/bin/bash
#SBATCH --job-name=celeba16_50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch root}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker through submit.sh}"
cd "$repo_root"

found=false
while IFS=$'\t' read -r row_id variant prediction loss train_start train_end sample_start sample_end use_safe_div safe_div_denominator_floor eps_to_v final_pixel_clamp; do
    if [[ "$row_id" == "$task_id" ]]; then
        found=true
        break
    fi
done < "$batch_root/variants.tsv"
if [[ "$found" != true ]]; then
    echo "No comparison row for task $task_id." >&2
    exit 2
fi

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1

data_root="${MP_TIME_RANGE_DATA_ROOT:-$repo_root/../old_mem_par/data}"
if [[ ! -d "$data_root/celeba/img_align_celeba" ]]; then
    echo "Missing aligned CelebA images under $data_root/celeba." >&2
    exit 2
fi

experiment_set="celeba32_n16_50k_time_range_noattn"
run_dir="$batch_root/runs/${task_id}_${variant}_s0"
run_name="celeba32-N16-B16-base32_m124-noattn-${variant}-s0"
group="celeba32-n16-b16-base32_m124-noattn-lr5e-4-heun50-final-euler-50k-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "$batch_root/time_range_config.py"
    --run-dir "$run_dir"
    --device cuda
    --steps 50000
    --lr 0.0005
    --sampler heun
    --sampling-steps 50
    --heun-final-euler
    --wandb
    --wandb-mode online
    --set "data.root=$data_root"
    --set "framework.prediction_space=$prediction"
    --set "framework.loss_space=$loss"
    --set "training.start_time=$train_start"
    --set "training.end_time=$train_end"
    --set "sampling.start_time=$sample_start"
    --set "sampling.end_time=$sample_end"
    --set "sampling.use_safe_div=$use_safe_div"
    --set "sampling.safe_div_denominator_floor=$safe_div_denominator_floor"
    --set "sampling.eps_to_v=$eps_to_v"
    --set "sampling.final_pixel_clamp=$final_pixel_clamp"
    --set "outputs.run_name=$run_name"
    --set "tracking.wandb.name=$run_name"
    --set "tracking.wandb.group=$group"
    --set "tracking.wandb.subexperiment=$experiment_set"
)
printf 'Task %s: %s (subexperiment: %s)\n' "$task_id" "$variant" "$experiment_set"
printf 'Training command: '
printf '%q ' "${command[@]}"
printf '\n'
printf '%q ' "${command[@]}" > "$batch_root/commands/${task_id}_${variant}.sh"
printf '\n' >> "$batch_root/commands/${task_id}_${variant}.sh"
exec "${command[@]}"
