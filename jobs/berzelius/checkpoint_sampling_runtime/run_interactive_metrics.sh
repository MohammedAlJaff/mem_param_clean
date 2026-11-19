#!/bin/bash
# Run one full CelebA32 checkpoint evaluation inside an existing GPU allocation.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
run_dir="${repo_root}/outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324/runs/18_n2048_x_x_s0"
reference_artifact="${repo_root}/outputs/reference_banks/celeba_test_seed1_clean_n10000.pt"
assets_manifest="${repo_root}/outputs/model_assets/feature_f_mem/manifest.json"
checkpoint_step=${1:-1000000}
sample_batch_size=${2:-64}
tag="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir=${3:-${repo_root}/outputs/interactive_checkpoint_metrics/${tag}}
generated_bank=${4:-"${repo_root}/outputs/interactive_checkpoint_metrics/20261003T000727Z/generated_samples.pt"}

set +u
module load Mambaforge
mamba activate mem_param_env
set -u
torch_fidelity_root="${repo_root}/../old_mem_par/third_party/torch-fidelity"
if [[ ! -f "${torch_fidelity_root}/torch_fidelity/__init__.py" ]]; then
    printf 'Missing pinned torch-fidelity source: %s\n' "$torch_fidelity_root" >&2
    exit 1
fi
export PYTHONPATH="${repo_root}:${torch_fidelity_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

cd "$repo_root"
python -u -B -m mem_param_src.runners.checkpoint_metrics_benchmark \
    --run-dir "$run_dir" \
    --checkpoint-step "$checkpoint_step" \
    --reference-artifact "$reference_artifact" \
    --assets-manifest "$assets_manifest" \
    --output-dir "$output_dir" \
    --generated-bank "$generated_bank" \
    --sample-count 10000 \
    --sample-batch-size "$sample_batch_size" \
    --feature-batch-size 32 \
    --fid-batch-size 64 \
    --nn-chunk-size 256 \
    --device cuda

printf 'Complete metric summary: %s/summary.json\n' "$output_dir"
