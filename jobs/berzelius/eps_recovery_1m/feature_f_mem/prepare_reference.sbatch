#!/bin/bash
#SBATCH --job-name=feature_ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
repo_root=${1:?repository root required}
output_root=${2:?evaluation output root required}
assets_manifest=${3:?encoder asset manifest required}
reference_artifact=${4:?frozen validation artifact required}
cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="${repo_root}:${repo_root}/../old_mem_par/third_party/torch-fidelity${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-feature-ref-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/feature_f_mem/evaluate_feature_f_mem.py \
    prepare-reference \
    --output-root "$output_root" \
    --assets-manifest "$assets_manifest" \
    --reference-artifact "$reference_artifact" \
    --feature-batch-size 64 \
    --fid-batch-size 128 \
    --device cuda
