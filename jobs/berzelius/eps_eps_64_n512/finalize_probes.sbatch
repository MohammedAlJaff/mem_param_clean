#!/bin/bash
#SBATCH --job-name=eps64_probe_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
repo_root="${1:?submit_probes.sh must pass the repository root}"
batch_root="${2:?submit_probes.sh must pass the batch directory}"
cd "$batch_root"

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="${batch_root}/source"
python -u -B "${batch_root}/job/assert_frozen_import.py" "$batch_root"
python -u -B "${batch_root}/job/select_batches.py" "$batch_root"
