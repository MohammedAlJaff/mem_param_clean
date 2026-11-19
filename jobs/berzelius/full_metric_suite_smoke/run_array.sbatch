#!/bin/bash
#SBATCH --job-name=full_metric_suite_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

batch_root=${1:?submit.sh must pass the frozen batch directory}
task_id=${SLURM_ARRAY_TASK_ID:?This worker requires an array task ID}
job_root="${batch_root}/_job"
repo_root=$(<"${job_root}/repo_root.txt")
profile_id=trajectory512
sample_count=512
loss_count=512
checkpoint_selector=all
expected_override=0
if [[ -f "${job_root}/evaluation_profile.sh" ]]; then
    source "${job_root}/evaluation_profile.sh"
fi
variant_file="${job_root}/variants.tsv"
row=$(sed -n "$((task_id + 2))p" "$variant_file")
if [[ -z "$row" ]]; then
    echo "No frozen variant exists for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r task_key run_path expected_checkpoints reference_path <<< "$row"
if (( expected_override > 0 )); then
    expected_checkpoints=$expected_override
fi
run_dir="${repo_root}/${run_path}"
output_dir="${batch_root}/${task_key}"
noise_artifact="${repo_root}/outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt"
assets_manifest="${repo_root}/outputs/model_assets/feature_f_mem/manifest.json"
task_root="${job_root}/tasks/${task_id}_${task_key}"
mkdir -p "$task_root"

status=failed
finish() {
    code=$?
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'task_id\t%s\ntask_key\t%s\nstatus\t%s\nexit_code\t%s\nfinished_at\t%s\n' \
        "$task_id" "$task_key" "$status" "$code" "$finished_at" \
        > "${task_root}/status.tsv"
}
trap finish EXIT

cd "$repo_root"
sha256sum --check --quiet "${job_root}/source_inputs_${task_key}.sha256"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
frozen_torch_fidelity="${job_root}/source/torch_fidelity"
if [[ ! -f "${frozen_torch_fidelity}/__init__.py" ]]; then
    echo "Missing frozen torch-fidelity source: ${frozen_torch_fidelity}" >&2
    exit 1
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="${task_root}/matplotlib"
export PYTHONPATH="${job_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

first_incomplete=complete
if [[ "$checkpoint_selector" == all ]]; then
    mapfile -t selected_checkpoints < <(
        find "${run_dir}/checkpoints" -maxdepth 1 -name 'step_*.pt' | sort -V
    )
else
    selected_checkpoints=(
        "${run_dir}/checkpoints/step_$(printf '%06d' "$checkpoint_selector").pt"
    )
fi
for checkpoint_path in "${selected_checkpoints[@]}"; do
    step_name=$(basename "$checkpoint_path" .pt)
    if [[ ! -f "${output_dir}/checkpoints/${step_name}/COMPLETE.json" ]]; then
        first_incomplete="$step_name"
        break
    fi
done
printf '%s\n' "$first_incomplete" > "${task_root}/first_incomplete.txt"

command=(
    python -u -B -m mem_param_src.runners.image_eval
    --run-dir "$run_dir"
    --checkpoints "$checkpoint_selector"
    --metrics full
    --sample-count "$sample_count"
    --loss-count "$loss_count"
    --noise-artifact "$noise_artifact"
    --assets-manifest "$assets_manifest"
    --output-dir "$output_dir"
    --feature-batch-size 32
    --nn-batch-size 512
    --write-plots
    --device cuda
)
if [[ "$reference_path" != "-" ]]; then
    command+=(--reference-artifact "${repo_root}/${reference_path}")
fi
printf '%q ' "${command[@]}" > "${task_root}/evaluator_command.sh"
printf '\n' >> "${task_root}/evaluator_command.sh"

"${command[@]}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/first_pass_complete.txt"
"${command[@]}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/reuse_pass_complete.txt"

python -u -B -m mem_param_src.runners.full_suite_validate \
    --evaluation-root "$output_dir" \
    --expected-checkpoints "$expected_checkpoints" \
    --sample-count "$sample_count"
sha256sum --check --quiet "${job_root}/source_inputs_${task_key}.sha256"
status=complete
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
