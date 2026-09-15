#!/bin/bash
#SBATCH --job-name=eps_1m_eval_combine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

repo_root=$1
output_root=$2

cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-eps-1m-combine-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/eval/combine_results.py \
    --output-root "$output_root"
