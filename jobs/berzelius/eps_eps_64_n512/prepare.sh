#!/bin/bash
# Freeze one source/config/matrix snapshot before any validation or training job.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: bash prepare.sh [--dry-run]" >&2
    exit 2
fi

batch_tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/eps_eps_64_n512/${batch_tag}"
printf 'Study batch directory: %s\n' "$batch_root"
printf 'This command freezes source, config, the 24-row matrix, and validation tools.\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p \
    "${batch_root}/commands" \
    "${batch_root}/job" \
    "${batch_root}/main/slurm" \
    "${batch_root}/validation/probes/results" \
    "${batch_root}/validation/slurm" \
    "${batch_root}/source"
cp -a "${script_dir}/." "${batch_root}/job/"
cp "${script_dir}/variants.tsv" "${batch_root}/variants.tsv"
cp "${script_dir}/profiles.tsv" "${batch_root}/profiles.tsv"
cp "${repo_root}/configs/runners/eps_eps_64_n512_400k.py" \
    "${batch_root}/run_config.py"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src configs/runners \
    > "${batch_root}/source.diff"
cd "$batch_root"
set +u
module load Mambaforge
mamba activate mem_param_env
set -u
PYTHONPATH="${batch_root}/source" python -u -B \
    "${batch_root}/job/assert_frozen_import.py" "$batch_root"
PYTHONPATH="${batch_root}/source" python -u -B \
    "${batch_root}/job/validate_suite.py" \
    "${batch_root}/job" "${batch_root}/run_config.py"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/BATCH_PREPARED"
printf 'Prepared: %s\n' "$batch_root"
