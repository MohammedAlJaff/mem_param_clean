#!/bin/bash
#SBATCH --job-name=eps64_gate_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
repo_root="${1:?submit.sh must pass the repository root}"
batch_root="${2:?submit.sh must pass the batch directory}"
cd "$batch_root"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
python -u -B "${batch_root}/job/verify_gate.py" "$batch_root"
