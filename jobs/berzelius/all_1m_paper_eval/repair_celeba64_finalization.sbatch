#!/bin/bash
# Finalize completed CelebA64 evaluations with their original frozen source.
#SBATCH --job-name=repair_celeba64
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

batch_root=$(realpath "${1:?Pass the frozen all-1M evaluation batch directory}")
group=celeba64_rgb
task_id=${SLURM_ARRAY_TASK_ID:?This worker requires an array task ID}
job_root="${batch_root}/_job"
variant_file="${job_root}/variants.tsv"
row=$(awk -F '\t' -v group="$group" -v task="$task_id" \
    'NR > 1 && $1 == group && $2 == task { print; exit }' "$variant_file")
if [[ -z "$row" ]]; then
    echo "No frozen variant exists for ${group} task ${task_id}." >&2
    exit 2
fi
IFS=$'\t' read -r frozen_group frozen_task task_key _ \
    expected_checkpoints _ _ _ _ _ <<< "$row"
if [[ "$frozen_group" != "$group" || "$frozen_task" != "$task_id" ]]; then
    echo "Frozen task identity differs from the requested task." >&2
    exit 2
fi

output_dir="${batch_root}/results/${group}/${task_key}"
original_task_root="${job_root}/tasks/${group}/${task_id}_${task_key}"
task_root="${job_root}/tasks/${group}_repair/${task_id}_${task_key}"
retention_manifest="${output_dir}/retention/DELETE_MANIFEST.json"
frozen_source="${job_root}/source"
mkdir -p "$task_root"

status=failed
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
finish() {
    code=$?
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf 'group\t%s\ntask_id\t%s\ntask_key\t%s\nstatus\t%s\nexit_code\t%s\nstarted_at\t%s\nfinished_at\t%s\n' \
        "$group" "$task_id" "$task_key" "$status" "$code" \
        "$started_at" "$finished_at" > "${task_root}/status.tsv"
}
trap finish EXIT

cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
sha256sum --check --quiet "${job_root}/common_inputs.sha256"
sha256sum --check --quiet "${original_task_root}/source_inputs.sha256"
nscquota > "${task_root}/quota_start.txt" 2>&1 || true

module --force purge
unset LMOD_FAMILY_MAMBA_ENV LMOD_FAMILY_MAMBA_ENV_VERSION
set +u
module load Mambaforge
mamba activate mem_param_env
set -u

if [[ ! -f "${frozen_source}/mem_param_src/__init__.py" ]]; then
    echo "Missing frozen mem_param source: ${frozen_source}/mem_param_src" >&2
    exit 1
fi
if [[ ! -f "${frozen_source}/torch_fidelity/__init__.py" ]]; then
    echo "Missing frozen torch-fidelity source: ${frozen_source}/torch_fidelity" >&2
    exit 1
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR="${task_root}/matplotlib"
export PYTHONPATH="$frozen_source"
export PYTHONUNBUFFERED=1

python -c \
    'import pathlib, sys, mem_param_src, torch_fidelity; root = pathlib.Path(sys.argv[1]).resolve(); paths = [pathlib.Path(mem_param_src.__file__).resolve(), pathlib.Path(torch_fidelity.__file__).resolve()]; assert all(path.is_relative_to(root) for path in paths), paths; print("frozen imports:", *paths)' \
    "$frozen_source" > "${task_root}/frozen_imports.txt"

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

python -c \
    'import json, pathlib, sys; manifest = json.loads(pathlib.Path(sys.argv[1]).read_text()); assert manifest.get("status") == "complete"; assert "streaming_retention" not in manifest' \
    "${output_dir}/eval_manifest.json"
complete_count=$(find "${output_dir}/checkpoints" -mindepth 2 -maxdepth 2 \
    -name COMPLETE.json -type f | wc -l)
if [[ "$complete_count" -ne "$expected_checkpoints" ]]; then
    echo "Expected ${expected_checkpoints} complete checkpoints, found ${complete_count}." >&2
    exit 1
fi
if [[ ! -f "${original_task_root}/evaluator_command.sh" ]]; then
    echo "Missing original frozen evaluator command." >&2
    exit 1
fi

# Stay outside repo_root so Python imports the original frozen source.
cd "$batch_root"
bash "${original_task_root}/evaluator_command.sh"
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/reuse_pass_complete.txt"

python -u -B -m mem_param_src.runners.full_suite_validate \
    --evaluation-root "$output_dir" \
    --expected-checkpoints "$expected_checkpoints" \
    --sample-count 10000 \
    --evaluation-tier paper
python -u -B -m mem_param_src.runners.full_suite_retention \
    --evaluation-root "$output_dir"
python -u -B -m mem_param_src.runners.full_suite_retention \
    --evaluation-root "$output_dir" \
    --manifest "$retention_manifest" \
    --apply
python -u -B -m mem_param_src.runners.full_suite_retention \
    --evaluation-root "$output_dir" --validate

cd "$batch_root"
sha256sum --check --quiet "${job_root}/snapshot.sha256"
sha256sum --check --quiet "${job_root}/common_inputs.sha256"
sha256sum --check --quiet "${original_task_root}/source_inputs.sha256"
nscquota > "${task_root}/quota_end.txt" 2>&1 || true
status=complete
date -u +%Y-%m-%dT%H:%M:%SZ > "${task_root}/COMPLETE"
