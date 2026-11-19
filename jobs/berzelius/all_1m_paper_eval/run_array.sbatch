#!/bin/bash
#SBATCH --job-name=all_1m_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

batch_root=${1:?submit.sh must pass the frozen batch directory}
group=${2:?submit.sh must pass the frozen group name}
task_id=${SLURM_ARRAY_TASK_ID:?This worker requires an array task ID}
job_root="${batch_root}/_job"
repo_root=$(<"${job_root}/repo_root.txt")
variant_file="${job_root}/variants.tsv"
row=$(awk -F '\t' -v group="$group" -v task="$task_id" \
    'NR > 1 && $1 == group && $2 == task { print; exit }' "$variant_file")
if [[ -z "$row" ]]; then
    echo "No frozen variant exists for ${group} task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r frozen_group frozen_task task_key run_path \
    expected_checkpoints noise_path reference_path dataset image_channels \
    image_size <<< "$row"
if [[ "$frozen_group" != "$group" || "$frozen_task" != "$task_id" ]]; then
    echo "Frozen task identity differs from the requested task." >&2
    exit 2
fi

run_dir="${repo_root}/${run_path}"
output_dir="${batch_root}/results/${group}/${task_key}"
noise_artifact="${repo_root}/${noise_path}"
assets_manifest="${repo_root}/outputs/model_assets/feature_f_mem/manifest.json"
task_root="${job_root}/tasks/${group}/${task_id}_${task_key}"
retention_manifest="${output_dir}/retention/DELETE_MANIFEST.json"
mkdir -p "$task_root"

status=failed
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
finish() {
    code=$?
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'group\t%s\ntask_id\t%s\ntask_key\t%s\ndataset\t%s\nimage_channels\t%s\nimage_size\t%s\nstatus\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\n' \
        "$group" "$task_id" "$task_key" "$dataset" "$image_channels" \
        "$image_size" "$status" "$code" "$started_at" "$finished_at" \
        > "${task_root}/status.tsv"
}
trap finish EXIT

cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
sha256sum --check --quiet "${job_root}/common_inputs.sha256"
sha256sum --check --quiet \
    "${job_root}/tasks/${group}/${task_id}_${task_key}/source_inputs.sha256"
nscquota > "${task_root}/quota_start.txt" 2>&1 || true

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

if [[ -f "${output_dir}/RETENTION_COMPLETE.json" ]]; then
    python -u -B -m mem_param_src.runners.full_suite_retention \
        --evaluation-root "$output_dir" --validate
    status=complete
    date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
    exit 0
fi

if [[ -f "$retention_manifest" ]]; then
    python -u -B -m mem_param_src.runners.full_suite_retention \
        --evaluation-root "$output_dir" \
        --manifest "$retention_manifest" \
        --apply
    python -u -B -m mem_param_src.runners.full_suite_retention \
        --evaluation-root "$output_dir" --validate
    status=complete
    date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
    exit 0
fi

if [[ ! -f "$retention_manifest" ]]; then
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
        --assets-manifest "$assets_manifest"
        --output-dir "$output_dir"
        --feature-batch-size 32
        --nn-batch-size 512
        --write-plots
        --publication-evaluation
        --device cuda
    )
    if [[ "$reference_path" != "-" ]]; then
        command+=(--reference-artifact "${repo_root}/${reference_path}")
    fi
    printf '%q ' "${command[@]}" > "${task_root}/evaluator_command.sh"
    printf '\n' >> "${task_root}/evaluator_command.sh"

    cd "$repo_root"
    "${command[@]}"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/first_pass_complete.txt"
    "${command[@]}"
    date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/reuse_pass_complete.txt"
fi

python -u -B -m mem_param_src.runners.full_suite_validate \
    --evaluation-root "$output_dir" \
    --expected-checkpoints "$expected_checkpoints" \
    --sample-count 10000 \
    --evaluation-tier paper

if [[ ! -f "$retention_manifest" ]]; then
    python -u -B -m mem_param_src.runners.full_suite_retention \
        --evaluation-root "$output_dir"
fi
python -u -B -m mem_param_src.runners.full_suite_retention \
    --evaluation-root "$output_dir" \
    --manifest "$retention_manifest" \
    --apply
python -u -B -m mem_param_src.runners.full_suite_retention \
    --evaluation-root "$output_dir" --validate

cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
sha256sum --check --quiet "${job_root}/common_inputs.sha256"
sha256sum --check --quiet \
    "${job_root}/tasks/${group}/${task_id}_${task_key}/source_inputs.sha256"
nscquota > "${task_root}/quota_end.txt" 2>&1 || true
status=complete
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
