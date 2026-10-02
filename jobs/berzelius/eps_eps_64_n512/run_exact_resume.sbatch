#!/bin/bash
#SBATCH --job-name=eps64_resume_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
repo_root="${1:?submit_smokes.sh must pass the repository root}"
batch_root="${2:?submit_smokes.sh must pass the batch directory}"
cd "$batch_root"

profile_id=5
row=$(awk -F '\t' -v id="$profile_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/profiles.tsv")
IFS=$'\t' read -r _ capacity base_channels channel_mult time_emb_dim attention middle_attention attn_resolutions constraint <<< "$row"
selected=$(awk -F '\t' -v id="$profile_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/validation/selected_batches.tsv")
IFS=$'\t' read -r _ _ _ _ physical_batch effective_batch _ _ _ _ <<< "$selected"
if [[ "$capacity" != "wide_deep" || "$attention" != "on" || "$constraint" != "fat" ]]; then
    echo "Profile 5 is not the expected wide_deep attention-on fat profile." >&2
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
root="${batch_root}/validation/exact_resume"
common=(
    --config "${batch_root}/run_config.py" --device cuda
    --steps 4 --checkpoint-every 2 --recovery-every 1
    --set 'training.log_every=1'
    --batch-size "$physical_batch" --effective-batch-size "$effective_batch"
    --set "data.root=${data_root}"
    --set 'data.dataset=celeba32_grey'
    --set 'study.dataset_key=celeba64-rgb-resume-smoke'
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
    --set 'generation_monitor.n_samples=8'
    --set 'generation_monitor.n_train_samples=8'
    --set 'random_projected_f_mem.n_samples=8'
    --set 'evaluation.n_samples=8'
    --set 'evaluation.sample_bank_size=8'
    --set 'tracking.wandb.project=eps-eps-64-n512-resume-smoke'
    --set "tracking.wandb.group=eps64-n512-resume-$(basename -- "$batch_root")"
)

python -u -B -m mem_param_src.runners.image_train \
    "${common[@]}" --run-dir "${root}/baseline" \
    --set 'tracking.wandb.name=wide-deep-attn-uninterrupted'
python -u -B -m mem_param_src.runners.image_train \
    "${common[@]}" --run-dir "${root}/resumed" --stop-after-step 2 \
    --set 'tracking.wandb.name=wide-deep-attn-resumed'
cp "${root}/resumed/run.json" "${root}/resumed_paused_run.json"
python -u -B -m mem_param_src.runners.image_train \
    --resume-run-dir "${root}/resumed" --device cuda
python -u -B "${batch_root}/job/compare_resume.py" \
    "$root" "${batch_root}/validation/EXACT_RESUME_COMPLETE"
