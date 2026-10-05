#!/bin/bash
#SBATCH --job-name=mnist_jointae32
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

repo_root=${1:?submit.sh must pass the repository root}
batch_root=${2:?submit.sh must pass the frozen batch directory}
data_root=${MP_MNIST_DATA_ROOT:-${repo_root}/../old_mem_par/data}
if [[ ! -d "${data_root}/MNIST/raw" ]]; then
    echo "Missing MNIST cache under ${data_root}; refusing training." >&2
    exit 2
fi

cd "$batch_root"
set +u
module load Mambaforge
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

epochs=${MP_MNIST_ENCODER_EPOCHS:-15}
command=(
    conda run --no-capture-output -n mem_param_env
    python -u -B "${batch_root}/job/train.py"
    --data-root "$data_root"
    --output-dir "${batch_root}/asset"
    --device cuda
    --seed 2003
    --epochs "$epochs"
    --batch-size 256
    --workers "${SLURM_CPUS_PER_TASK:-4}"
    --feature-dim 128
    --classification-weight 0.1
    --min-validation-accuracy 0.99
)
printf '%q ' "${command[@]}" > "${batch_root}/commands/train.sh"
printf '\n' >> "${batch_root}/commands/train.sh"
exec "${command[@]}"
