#!/bin/bash
#SBATCH --job-name=eps_fid_reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

repo_root=$1
source_run=$2
output_artifact=$3
reference_count=$4

cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/eval/prepare_fid_reference.py \
    --source-run "$source_run" \
    --output-artifact "$output_artifact" \
    --reference-count "$reference_count"
