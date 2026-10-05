#!/bin/bash
#SBATCH --job-name=celeba16_10k
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
while IFS=$'\t' read -r row_id label prediction loss use_safe_div eps_to_v final_pixel_clamp; do
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

data_root="${MP_SMOKE_DATA_ROOT:-$repo_root/../old_mem_par/data}"
if [[ ! -d "$data_root/celeba/img_align_celeba" ]]; then
    echo "Missing aligned CelebA images under $data_root/celeba." >&2
    exit 2
fi

if [[ "$use_safe_div" == true ]]; then
    division_label="divsafe0p05"
else
    division_label="divexact"
fi
if [[ "$final_pixel_clamp" == true ]]; then
    pixel_label="pixclip"
else
    pixel_label="pixraw"
fi
if [[ "$prediction" == v ]]; then
    variant_label="v_v-native-$pixel_label"
else
    variant_label="${prediction}_${loss}-${eps_to_v}-${division_label}-${pixel_label}"
fi

run_dir="$batch_root/runs/${task_id}_${label}_s0"
run_name="celeba32-N16-B16-base32_m124-${variant_label}-s0"
group="celeba32-n16-b16-base32_m124-lr5e-4-heun50-final-euler-10k-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "$batch_root/smoke_config.py"
    --run-dir "$run_dir"
    --device cuda
    --steps 10000
    --lr 0.0005
    --sampler heun
    --sampling-steps 50
    --heun-final-euler
    --wandb
    --wandb-mode online
    --set "data.root=$data_root"
    --set "framework.prediction_space=$prediction"
    --set "framework.loss_space=$loss"
    --set "sampling.use_safe_div=$use_safe_div"
    --set "sampling.eps_to_v=$eps_to_v"
    --set "sampling.final_pixel_clamp=$final_pixel_clamp"
    --set "outputs.run_name=$run_name"
    --set "tracking.wandb.name=$run_name"
    --set "tracking.wandb.group=$group"
)
printf 'Task %s: %s\n' "$task_id" "$label"
printf 'Training command: '
printf '%q ' "${command[@]}"
printf '\n'
printf '%q ' "${command[@]}" > "$batch_root/commands/${task_id}_${label}.sh"
printf '\n' >> "$batch_root/commands/${task_id}_${label}.sh"
exec "${command[@]}"
