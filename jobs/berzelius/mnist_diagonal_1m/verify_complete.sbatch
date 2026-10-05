#!/bin/bash
#SBATCH --job-name=mnist_diag_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
batch_root="${1:?submit.sh must pass the frozen batch directory}"
cd "$batch_root"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
python -u -B "${batch_root}/job/verify_complete.py" "$batch_root"
