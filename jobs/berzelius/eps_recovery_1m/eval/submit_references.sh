#!/bin/bash
# Add the saved v/v and x/x references to the corrected three-point evaluation.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
output_root="${repo_root}/outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point"
source_manifest="${output_root}/manifest.json"
reference_manifest="${output_root}/reference_manifest.json"
fid_reference_artifact="${output_root}/reference/celeba32_valid_seed1_n10000.pt"
preview=false

if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
elif (( $# )); then
    echo "Usage: bash submit_references.sh [--dry-run]" >&2
    exit 2
fi

for path in "$source_manifest" "$fid_reference_artifact"; do
    if [[ ! -f "$path" ]]; then
        echo "Required corrected-evaluation artifact is missing: $path" >&2
        exit 2
    fi
done

account="${MP_EPS_EVAL_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS_EVAL_TIME:-02:00:00}"
manifest_command=(
    python3 "${script_dir}/make_reference_manifest.py"
    --source-manifest "$source_manifest"
)
array_command=(
    sbatch --parsable
    --account="$account"
    --time="$walltime"
    --array=0-1
    --chdir="$repo_root"
    --output="${output_root}/slurm/%x_%A_%a.out"
    --error="${output_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch"
    "$repo_root" "$output_root" "$reference_manifest"
    10000 10000 1024 "$fid_reference_artifact"
)

"${manifest_command[@]}"
printf 'Reference array command: '
printf '%q ' "${array_command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

"${manifest_command[@]}" --output "$reference_manifest"
git -C "$repo_root" diff HEAD -- mem_param_src jobs/berzelius/eps_recovery_1m/eval \
    > "${output_root}/reference_source.diff"
array_job_output=$("${array_command[@]}")
array_job=${array_job_output%%;*}
combine_job_output=$(
    sbatch --parsable \
        --account="$account" \
        --dependency="afterok:${array_job}" \
        --chdir="$repo_root" \
        --output="${output_root}/slurm/%x_%j.out" \
        --error="${output_root}/slurm/%x_%j.err" \
        "${script_dir}/combine_references.sbatch" "$repo_root" "$output_root"
)
combine_job=${combine_job_output%%;*}
printf '{"array_job":"%s","combine_job":"%s"}\n' \
    "$array_job" "$combine_job" > "${output_root}/reference_job_ids.json"
printf 'Submitted reference array %s and dependent combine job %s.\n' \
    "$array_job" "$combine_job"
