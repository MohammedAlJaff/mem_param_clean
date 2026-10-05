#!/bin/bash
#SBATCH --job-name=eps64_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail
repo_root="${1:?submit_probes.sh must pass the repository root}"
batch_root="${2:?submit_probes.sh must pass the batch directory}"
task_id="${SLURM_ARRAY_TASK_ID:?Run this worker as an array}"
cd "$batch_root"

row=$(awk -F '\t' -v id="$task_id" 'NR > 1 && $1 == id {print $0}' "${batch_root}/profiles.tsv")
if [[ -z "$row" ]]; then
    echo "No capacity profile for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r profile_id capacity base_channels channel_mult time_emb_dim attention middle_attention attn_resolutions constraint <<< "$row"
if [[ "$constraint" != "${MP_EPS64_PROBE_CLASS:?submitter must set MP_EPS64_PROBE_CLASS}" ]]; then
    echo "Profile ${profile_id} requires ${constraint}, not ${MP_EPS64_PROBE_CLASS}." >&2
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
if [[ ! -d "${data_root}/celeba/img_align_celeba" ]]; then
    echo "Missing CelebA cache under ${data_root}." >&2
    exit 2
fi

for physical_batch in 512 256 128 64; do
    output="${batch_root}/validation/probes/results/profile_${profile_id}_batch_${physical_batch}.json"
    python -u -B "${batch_root}/job/probe_one.py" \
        --config "${batch_root}/run_config.py" \
        --output "$output" \
        --data-root "$data_root" \
        --profile-id "$profile_id" \
        --capacity "$capacity" \
        --constraint "$constraint" \
        --base-channels "$base_channels" \
        --channel-mult "$channel_mult" \
        --time-emb-dim "$time_emb_dim" \
        --attention "$attention" \
        --middle-attention "$middle_attention" \
        --attn-resolutions "$attn_resolutions" \
        --physical-batch "$physical_batch"
done
