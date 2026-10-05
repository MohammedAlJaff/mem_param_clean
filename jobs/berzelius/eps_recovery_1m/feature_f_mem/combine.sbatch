#!/bin/bash
#SBATCH --job-name=feature_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
repo_root=${1:?repository root required}
output_root=${2:?evaluation output root required}
manifest=${3:?evaluation manifest required}
cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="$repo_root${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-feature-combine-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/feature_f_mem/combine_feature_f_mem.py \
    --output-root "$output_root" --manifest "$manifest"
