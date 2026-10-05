#!/bin/bash
#SBATCH --job-name=resume_cuda_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
cd "$repo_root"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1
data_root="${repo_root}/../old_mem_par/data"

python -u -B -m mem_param_src.runners.image_train \
    --config "${batch_root}/run_config.py" \
    --run-dir "${batch_root}/baseline" --device cuda \
    --set "data.root=${data_root}"

python -u -B -m mem_param_src.runners.image_train \
    --config "${batch_root}/run_config.py" \
    --run-dir "${batch_root}/resumed" --device cuda \
    --stop-after-step 20 \
    --set "data.root=${data_root}"

python -u -B -m mem_param_src.runners.image_train \
    --resume-run-dir "${batch_root}/resumed" --device cuda

python -u -B "${batch_root}/compare.py" "$batch_root"
