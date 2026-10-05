#!/bin/bash
# Submit a replacement 400k verifier after repaired tasks have completed.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# < 1 || $# > 2 )); then
    echo "Usage: bash submit_gate.sh [--dry-run] BATCH_DIRECTORY [DEPENDENCY]" >&2
    exit 2
fi
batch_root=$(cd -- "$1" && pwd)
dependency="${2:-}"
if [[ ! -s "${batch_root}/main/submission.tsv" ]]; then
    echo "No accepted main Slurm job is recorded for this batch." >&2
    exit 2
fi
if [[ -e "${batch_root}/GATE_400K_COMPLETE" ]]; then
    echo "The 400k gate is already complete." >&2
    exit 2
fi
gate_log="${batch_root}/main/replacement_gate_jobs.tsv"
if [[ -s "$gate_log" ]]; then
    previous_job=$(tail -n 1 "$gate_log" | awk -F '\t' '{print $3}')
    active_job=$(squeue -h -j "$previous_job" -o '%i')
    if [[ -n "$active_job" ]]; then
        echo "Replacement gate ${previous_job} is still active." >&2
        exit 2
    fi
fi
dependency_args=()
if [[ -n "$dependency" ]]; then
    if [[ ! "$dependency" =~ ^(afterok|afterany):[0-9_:+]+$ ]]; then
        echo "DEPENDENCY must be an explicit afterok:... or afterany:... expression." >&2
        exit 2
    fi
    dependency_args=(--dependency="$dependency")
fi

account="${MP_EPS64_ACCOUNT:-berzelius-2026-269}"
command=(sbatch --parsable --account="$account" --time=00:20:00
    --constraint=thin --gres=gpu:A100-SXM4-40GB:1 --mem=4G --no-requeue
    --chdir="$repo_root"
    "${dependency_args[@]}"
    --output="${batch_root}/main/slurm/replacement_gate_%j.out"
    --error="${batch_root}/main/slurm/replacement_gate_%j.err"
    "${batch_root}/job/verify_gate.sbatch" "$repo_root" "$batch_root")
printf 'Replacement gate command: '; printf '%q ' "${command[@]}"; printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

gate_job=$("${command[@]}")
gate_job="${gate_job%%;*}"
printf '%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${dependency:-none}" "$gate_job" \
    >> "$gate_log"
printf 'Replacement gate job: %s\n' "$gate_job"
