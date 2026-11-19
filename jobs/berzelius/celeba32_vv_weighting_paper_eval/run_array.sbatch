#!/bin/bash
#SBATCH --job-name=vv_weighting_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

batch_root=${1:?submit.sh must pass the frozen batch directory}
task_id=${SLURM_ARRAY_TASK_ID:?This worker requires an array task ID}
job_root="${batch_root}/_job"
repo_root=$(<"${job_root}/repo_root.txt")
variant_file="${job_root}/variants.tsv"
row=$(sed -n "$((task_id + 2))p" "$variant_file")
if [[ -z "$row" ]]; then
    echo "No frozen variant exists for task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r task_key run_path expected_checkpoints reference_path \
    n_train seed weighting <<< "$row"
run_dir="${repo_root}/${run_path}"
output_dir="${batch_root}/${task_key}"
noise_artifact="${repo_root}/outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt"
assets_manifest="${repo_root}/outputs/model_assets/feature_f_mem/manifest.json"
task_root="${job_root}/tasks/${task_id}_${task_key}"
mkdir -p "$task_root"

status=failed
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
finish() {
    code=$?
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'task_id\t%s\ntask_key\t%s\nn_train\t%s\nseed\t%s\nweighting\t%s\nstatus\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\n' \
        "$task_id" "$task_key" "$n_train" "$seed" "$weighting" "$status" \
        "$code" "$started_at" "$finished_at" > "${task_root}/status.tsv"
}
trap finish EXIT

cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
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
mapfile -t selected_checkpoints < <(
    find "${run_dir}/checkpoints" -maxdepth 1 -name 'step_*.pt' | sort -V
)
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
    --checkpoints all
    --metrics full
    --sample-count 10000
    --loss-count 512
    --noise-artifact "$noise_artifact"
    --reference-artifact "${repo_root}/${reference_path}"
    --assets-manifest "$assets_manifest"
    --output-dir "$output_dir"
    --feature-batch-size 32
    --nn-batch-size 512
    --write-plots
    --publication-evaluation
    --device cuda
)
printf '%q ' "${command[@]}" > "${task_root}/evaluator_command.sh"
printf '\n' >> "${task_root}/evaluator_command.sh"

cd "$repo_root"
"${command[@]}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/first_pass_complete.txt"
"${command[@]}"
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/reuse_pass_complete.txt"

python -u -B -m mem_param_src.runners.full_suite_validate \
    --evaluation-root "$output_dir" \
    --expected-checkpoints "$expected_checkpoints" \
    --sample-count 10000 \
    --evaluation-tier paper
cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
sha256sum --check --quiet "${job_root}/source_inputs_${task_key}.sha256"
status=complete
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
