#!/bin/bash
#SBATCH --job-name=feature_fmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
repo_root=${1:?repository root required}
output_root=${2:?evaluation output root required}
manifest=${3:?evaluation manifest required}
assets_manifest=${4:?encoder asset manifest required}
reference_artifact=${5:?frozen validation artifact required}
noise_artifact=${6:?common seed-2003 noise artifact required}
task_id=${SLURM_ARRAY_TASK_ID:?run through submit.sh as a Slurm array}
cd "$repo_root"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
export PYTHONPATH="${repo_root}:${repo_root}/../old_mem_par/third_party/torch-fidelity${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-feature-fmem-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
conda run --no-capture-output -n mem_param_env \
    python jobs/berzelius/eps_recovery_1m/feature_f_mem/evaluate_feature_f_mem.py run \
    --manifest "$manifest" \
    --task-index "$task_id" \
    --output-root "$output_root" \
    --assets-manifest "$assets_manifest" \
    --reference-artifact "$reference_artifact" \
    --noise-artifact "$noise_artifact" \
    --sample-count 10000 \
    --sample-batch-size 256 \
    --feature-batch-size 64 \
    --fid-batch-size 128 \
    --nn-chunk-size 512 \
    --device cuda
