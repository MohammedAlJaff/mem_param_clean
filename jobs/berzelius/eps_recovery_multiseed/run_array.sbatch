#!/bin/bash
#SBATCH --job-name=eps_multiseed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
repo_root=${1:?submit.sh must pass the repository root}
batch_root=${2:?submit.sh must pass the batch root}
task_id=${SLURM_ARRAY_TASK_ID:?run this worker through submit.sh}
found=false
while IFS=$'\t' read -r row_id label seed prediction loss; do
    if [[ "$row_id" == "$task_id" ]]; then found=true; break; fi
done < "${batch_root}/variants.tsv"
if [[ "$found" != true ]]; then
    echo "No multi-seed variant for task ${task_id}." >&2
    exit 2
fi
cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1
data_root="${MP_EPS_1M_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache below ${data_root}; set MP_EPS_1M_DATA_ROOT." >&2
    exit 2
fi
run_dir="${batch_root}/runs/${task_id}_${label}_s${seed}"
run_name="celeba32-unet-N256-B256-${prediction}_${loss}-steps1000000-s${seed}-${label}"
group="eps-recovery-multiseed-$(basename -- "$batch_root")"
command=(
    conda run --no-capture-output -n mem_param_env
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --steps 1000000 --seed "$seed" --batch-size 256 --lr 0.0001
    --wandb --wandb-mode online
    --set "data.root=${data_root}"
    --set "framework.prediction_space=${prediction}"
    --set "framework.loss_space=${loss}"
    --set "framework.loss_weighting=unit"
    --set "sampling.method=heun"
    --set "sampling.n_steps=50"
    --set "sampling.start_time=0.001"
    --set "sampling.end_time=0.999"
    --set "sampling.use_safe_div=true"
    --set "sampling.safe_div_denominator_floor=0.05"
    --set "sampling.eps_to_v=direct"
    --set "sampling.heun_final_euler=false"
    --set "sampling.output=x_t_max"
    --set "sampling.final_pixel_clamp=false"
    --set "outputs.run_name=${run_name}"
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
)
printf 'Task %s: seed=%s %s/%s\n' "$task_id" "$seed" "$prediction" "$loss"
printf '%q ' "${command[@]}" > "${batch_root}/commands/${task_id}_${label}_s${seed}.sh"
printf '\n' >> "${batch_root}/commands/${task_id}_${label}_s${seed}.sh"
exec "${command[@]}"
