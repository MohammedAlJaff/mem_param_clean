#!/bin/bash
#SBATCH --job-name=celeba32_vvw_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
batch_root="${1:?verify job requires the frozen batch directory}"
cd "$batch_root"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
python -u -B "${batch_root}/job/verify_complete.py" "$batch_root"
