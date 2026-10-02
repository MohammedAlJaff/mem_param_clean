#!/bin/bash
#SBATCH --job-name=celeba64_rgb_diag
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
    echo "No CelebA64 RGB production row for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r _row_id label n_train seed prediction loss <<< "$row"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"

data_root="${MP_CELEBA64_RGB_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}; refusing training." >&2
    exit 2
fi

run_dir="${batch_root}/runs/${task_id}_${label}"
run_name="celeba64-rgb-wide-deep-attn-${label}-B256-Beff256-1m"
group="celeba64-rgb-unet-diagonal-1m-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --stop-after-step "$stop_step"
    --batch-size 256 --effective-batch-size 256
    --set "data.root=${data_root}"
    --set "data.n_train=${n_train}"
    --set "training.seed=${seed}"
    --set "framework.prediction_space=${prediction}"
    --set "framework.loss_space=${loss}"
    --set 'sampling.eps_to_v=via_x'
    --set 'tracking.wandb.project=celeba64-rgb-unet-diagonal-1m'
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf 'Task %s: %s; physical/effective batch 256/256; accumulation 1; stop %s.\n' \
    "$task_id" "$label" "$stop_step"
printf '%q ' "${command[@]}" > "${batch_root}/commands/initial_${task_id}_${label}.sh"
printf '\n' >> "${batch_root}/commands/initial_${task_id}_${label}.sh"
exec "${command[@]}"
