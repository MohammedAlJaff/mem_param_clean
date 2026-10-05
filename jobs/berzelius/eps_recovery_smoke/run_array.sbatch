#!/bin/bash
#SBATCH --job-name=eps_recovery_2k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# This worker is submitted by submit.sh; each array task launches one runner.
set -euo pipefail
repo_root="${1:?submit.sh must pass the clean repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker through the submission script}"
cd "$repo_root"

found=false
while IFS=$'\t' read -r row_id label prediction loss lr use_safe_div eps_to_v final_euler; do
    if [[ "$row_id" == "$task_id" ]]; then
        found=true
        break
    fi
done < "${batch_root}/variants.tsv"
if [[ "$found" != true ]]; then
    echo "No comparison row for task ${task_id}." >&2
    exit 2
fi

# Mamba activation can read unset shell variables on this cluster.
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1

# Reuse the already downloaded cache. Do not download data inside the job.
data_root="${MP_SMOKE_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing aligned CelebA images under ${data_root}/celeba." >&2
    echo "Set MP_SMOKE_DATA_ROOT to a cache containing the celeba directory." >&2
    exit 2
fi
run_dir="${batch_root}/runs/${task_id}_${label}_s0"
run_name="celeba32-unet-N256-B256-${prediction}_${loss}-steps2000-s0-${label}"
group="eps-recovery-2k-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/smoke_config.py"
    --run-dir "$run_dir" --device cuda
    --steps 2000 --checkpoint-every 250
    --wandb --wandb-mode online
    --set "data.root=${data_root}"
    --set "framework.prediction_space=${prediction}"
    --set "framework.loss_space=${loss}"
    --lr "$lr"
    --set "sampling.use_safe_div=${use_safe_div}"
    --set "sampling.eps_to_v=${eps_to_v}"
    --set "sampling.heun_final_euler=${final_euler}"
    --set "outputs.run_name=${run_name}"
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
)
printf 'Task %s: %s
' "$task_id" "$label"
printf 'Training command: '
printf '%q ' "${command[@]}"
printf '
'
printf '%q ' "${command[@]}" > "${batch_root}/commands/${task_id}_${label}.sh"
printf '
' >> "${batch_root}/commands/${task_id}_${label}.sh"
exec "${command[@]}"
