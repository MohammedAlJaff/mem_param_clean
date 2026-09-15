#!/bin/bash
#SBATCH --job-name=eps_1m_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

repo_root=$1
output_root=$2
manifest_path=$3
sample_count=$4
fid_reference_count=$5
loss_count=$6
fid_reference_artifact=$7

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required." >&2
    exit 2
fi

cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="${repo_root}:${repo_root}/../old_mem_par/third_party/torch-fidelity${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-eps-1m-eval-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$MPLCONFIGDIR"

conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/eval/evaluate_checkpoints.py run \
    --manifest "$manifest_path" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --output-root "$output_root" \
    --sample-count "$sample_count" \
    --fid-reference-count "$fid_reference_count" \
    --fid-reference-artifact "$fid_reference_artifact" \
    --loss-count "$loss_count" \
    --sample-batch-size 256 \
    --fid-batch-size 128 \
    --loss-batch-size 256 \
    --nn-chunk-size 256 \
    --fid-feature-extractor inception-v3-compat \
    --device cuda
