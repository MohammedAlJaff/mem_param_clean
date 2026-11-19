#!/bin/bash
#SBATCH --job-name=checkpoint_sampling_runtime
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

batch_root=${1:?submit.sh must pass the frozen batch directory}
run_dir=${2:?submit.sh must pass the saved run directory}
checkpoint_step=${3:?submit.sh must pass the checkpoint step}
measure_count=${4:?submit.sh must pass the measured sample count}
target_count=${5:?submit.sh must pass the target sample count}
batch_size=${6:?submit.sh must pass the sampling batch size}
time_budget=${7:?submit.sh must pass the sampling time budget}

cd "$batch_root"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="${batch_root}/source"
export PYTHONUNBUFFERED=1

command=(
    python -u -B -m mem_param_src.runners.sampling_benchmark
    --run-dir "$run_dir"
    --checkpoint-step "$checkpoint_step"
    --measure-count "$measure_count"
    --target-count "$target_count"
    --batch-size "$batch_size"
    --warmup-batches 1
    --time-budget-seconds "$time_budget"
    --output-dir "${batch_root}/results"
    --preview-count 0
    --device cuda
)
printf '%q ' "${command[@]}" > "${batch_root}/commands/worker.sh"
printf '\n' >> "${batch_root}/commands/worker.sh"
exec "${command[@]}"
