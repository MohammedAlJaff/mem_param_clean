#!/bin/bash
# Submit six independent capacity-profile probes on their intended GPU classes.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 1 )); then
    echo "Usage: bash submit_probes.sh [--dry-run] BATCH_DIRECTORY" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
if [[ ! -f "${batch_root}/BATCH_PREPARED" ]]; then
    echo "BATCH_DIRECTORY was not created by prepare.sh." >&2
    exit 2
fi
if [[ -e "${batch_root}/PROBE_COMPLETE" ]]; then
    echo "This batch already has completed capacity probes." >&2
    exit 2
fi
jobs_path="${batch_root}/validation/probe_jobs.tsv"
if [[ -e "$jobs_path" ]]; then
    echo "A prior probe submission is complete or partial: ${jobs_path}" >&2
    echo "Inspect recorded job IDs and Slurm state. Do not resubmit blindly." >&2
    exit 2
fi

account="${MP_EPS64_ACCOUNT:-berzelius-2026-269}"
worker="${batch_root}/job/run_probe_array.sbatch"
thin=(sbatch --parsable --account="$account" --time=02:00:00 --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1 --mem=64G --no-requeue --chdir="$repo_root"
    --export=ALL,MP_EPS64_PROBE_CLASS=thin --array=0-3
    --output="${batch_root}/validation/slurm/probe_thin_%A_%a.out"
    --error="${batch_root}/validation/slurm/probe_thin_%A_%a.err"
    "$worker" "$repo_root" "$batch_root")
fat=(sbatch --parsable --account="$account" --time=02:00:00 --constraint=fat
    --gres=gpu:A100-SXM4-80GB:1 --mem=128G --no-requeue --chdir="$repo_root"
    --export=ALL,MP_EPS64_PROBE_CLASS=fat --array=4-5
    --output="${batch_root}/validation/slurm/probe_fat_%A_%a.out"
    --error="${batch_root}/validation/slurm/probe_fat_%A_%a.err"
    "$worker" "$repo_root" "$batch_root")

printf 'Thin probe command: '; printf '%q ' "${thin[@]}"; printf '\n'
printf 'Fat probe command: '; printf '%q ' "${fat[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    printf 'Finalize: afterok on both probe arrays; writes selected_batches.tsv.\n'
    exit 0
fi

thin_job=$("${thin[@]}")
thin_job="${thin_job%%;*}"
printf 'thin\t%s\n' "$thin_job" > "$jobs_path"
fat_job=$("${fat[@]}")
fat_job="${fat_job%%;*}"
printf 'fat\t%s\n' "$fat_job" >> "$jobs_path"
final_job=$(sbatch --parsable --account="$account" --time=00:10:00 --constraint=thin \
    --gres=gpu:A100-SXM4-40GB:1 --mem=2G --no-requeue \
    --dependency="afterok:${thin_job}:${fat_job}" --chdir="$repo_root" \
    --output="${batch_root}/validation/slurm/probe_finalize_%j.out" \
    --error="${batch_root}/validation/slurm/probe_finalize_%j.err" \
    "${batch_root}/job/finalize_probes.sbatch" "$repo_root" "$batch_root")
final_job="${final_job%%;*}"
printf 'finalize\t%s\n' "$final_job" >> "$jobs_path"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/PROBE_SUBMITTED"
printf 'Thin probe array: %s\nFat probe array: %s\nFinalize job: %s\n' \
    "$thin_job" "$fat_job" "$final_job"
