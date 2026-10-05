#!/bin/bash
#SBATCH --job-name=eps64_wandb_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
repo_root="${1:?submit_smokes.sh must pass the repository root}"
batch_root="${2:?submit_smokes.sh must pass the batch directory}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$batch_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/profiles.tsv")
if [[ -z "$row" ]]; then
    echo "No smoke profile for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r profile_id capacity base_channels channel_mult time_emb_dim attention middle_attention attn_resolutions constraint <<< "$row"
if [[ "$constraint" != "${MP_EPS64_SMOKE_CLASS:?submitter must set MP_EPS64_SMOKE_CLASS}" ]]; then
    echo "Profile ${profile_id} requires ${constraint}, not ${MP_EPS64_SMOKE_CLASS}." >&2
    exit 2
fi
selected=$(awk -F '\t' -v id="$profile_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/validation/selected_batches.tsv")
if [[ -z "$selected" ]]; then
    echo "No selected batch for profile ${profile_id}." >&2
    exit 2
fi
IFS=$'\t' read -r _ _ _ _ physical_batch effective_batch _ _ _ _ <<< "$selected"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
data_root="${MP_EPS64_DATA_ROOT:-${repo_root}/../old_mem_par/data}"
run_dir="${batch_root}/validation/smokes/${profile_id}_${capacity}_${attention}"
run_name="eps64-N512-${capacity}-${attention}-B${physical_batch}-wandb-smoke-s0"
group="eps64-n512-validation-$(basename -- "$batch_root")"
command=(
    python -u -B -m mem_param_src.runners.image_train
    --config "${batch_root}/run_config.py"
    --run-dir "$run_dir" --device cuda
    --steps 250 --checkpoint-every 250 --recovery-every 250
    --batch-size "$physical_batch" --effective-batch-size "$effective_batch"
    --set "data.root=${data_root}"
    --set 'data.dataset=celeba32_grey'
    --set 'study.dataset_key=celeba64-rgb-smoke'
    --set 'data.train_split=train'
    --set 'data.reference_split=valid'
    --set 'data.image_size=64'
    --set 'data.resize_size=64'
    --set 'data.image_channels=3'
    --set "network.base_channels=${base_channels}"
    --set "network.channel_mult=${channel_mult}"
    --set "network.time_emb_dim=${time_emb_dim}"
    --set "network.middle_attention=${middle_attention}"
    --set "network.attn_resolutions=${attn_resolutions}"
    --set 'tracking.wandb.project=eps-eps-64-n512-smoke'
    --set "tracking.wandb.name=${run_name}"
    --set "tracking.wandb.group=${group}"
    --set "outputs.run_name=${run_name}"
)
printf '%q ' "${command[@]}" > "${batch_root}/commands/smoke_${profile_id}_${capacity}_${attention}.sh"
printf '\n' >> "${batch_root}/commands/smoke_${profile_id}_${capacity}_${attention}.sh"
start_epoch=$(date +%s)
"${command[@]}"
end_epoch=$(date +%s)
printf '%s\n' "$((end_epoch - start_epoch))" > "${run_dir}/smoke_wall_seconds.txt"
