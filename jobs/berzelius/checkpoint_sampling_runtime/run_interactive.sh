#!/bin/bash
# Run the CelebA32 sampling benchmark inside an existing GPU allocation.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
run_dir="${repo_root}/outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324/runs/18_n2048_x_x_s0"
measure_count=${1:-10000}
batch_size=${2:-64}
tag="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir=${3:-${repo_root}/outputs/interactive_sampling_runtime/${tag}}

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
export PYTHONPATH="$repo_root"
export PYTHONUNBUFFERED=1

cd "$repo_root"
python -u -B -m mem_param_src.runners.sampling_benchmark \
    --run-dir "$run_dir" \
    --checkpoint-step 1000000 \
    --measure-count "$measure_count" \
    --target-count 10000 \
    --batch-size "$batch_size" \
    --warmup-batches 1 \
    --device cuda \
    --output-dir "$output_dir" \
    --preview-count 0

printf 'Timing summary: %s/summary.json\n' "$output_dir"
