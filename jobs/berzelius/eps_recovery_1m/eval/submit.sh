#!/bin/bash
# Evaluate three nonzero checkpoints for the six existing eps/eps runs.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
source_root="${repo_root}/outputs/job_sets/eps_recovery_1m/20260914T113123Z_2308577/runs"
variants="${script_dir}/../variants.tsv"
profiles="${script_dir}/sampling_profiles.tsv"

preview=false
mode=three_point
max_step=700000
sample_count=10000
fid_reference_count=10000
loss_count=1024
checkpoint_steps="100000,390000,695000"

usage() {
    echo "Usage: bash submit.sh [--dry-run] [--max-step N]" >&2
    echo "                      [--sample-count N] [--fid-reference-count N]" >&2
    echo "                      [--loss-count N] [--checkpoint-steps CSV]" >&2
}

while (( $# )); do
    case "$1" in
        --dry-run)
            preview=true
            shift
            ;;
        --max-step|--sample-count|--fid-reference-count|--loss-count|--checkpoint-steps)
            if (( $# < 2 )); then
                usage
                exit 2
            fi
            case "$1" in
                --max-step) max_step=$2 ;;
                --sample-count) sample_count=$2 ;;
                --fid-reference-count) fid_reference_count=$2 ;;
                --loss-count) loss_count=$2 ;;
                --checkpoint-steps) checkpoint_steps=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

for value in "$max_step" "$sample_count" "$fid_reference_count" "$loss_count"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "Counts and max step must be positive integers." >&2
        exit 2
    fi
done
if (( loss_count > 1024 )); then
    echo "Loss count cannot exceed the saved 1,024 held-out images." >&2
    exit 2
fi
if (( sample_count == 10000 && fid_reference_count < 10000 )); then
    echo "A 10k generated-sample FID requires at least 10k reference images." >&2
    exit 2
fi

account="${MP_EPS_EVAL_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_EPS_EVAL_TIME:-02:00:00}"
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$_${mode}"
output_root="${repo_root}/outputs/evals/eps_recovery_1m/${tag}"
manifest_path="${output_root}/manifest.json"
reference_source_run="${source_root}/0_eps_old_sampling_s0"
reference_artifact="${output_root}/reference/celeba32_valid_seed1_n${fid_reference_count}.pt"

manifest_command=(
    python3 "${script_dir}/evaluate_checkpoints.py" manifest
    --run-root "$source_root"
    --variants "$variants"
    --profiles "$profiles"
    --max-step "$max_step"
)
if [[ -n "$checkpoint_steps" ]]; then
    manifest_command+=(--checkpoint-steps "$checkpoint_steps")
fi

prepare_command=(
    sbatch --parsable
    --account="$account"
    --chdir="$repo_root"
    --output="${output_root}/slurm/%x_%j.out"
    --error="${output_root}/slurm/%x_%j.err"
    "${script_dir}/prepare_fid_reference.sbatch"
    "$repo_root" "$reference_source_run" "$reference_artifact"
    "$fid_reference_count"
)

array_command=(
    sbatch --parsable
    --dependency=afterok:REFERENCE_JOB_ID
    --account="$account"
    --time="$walltime"
    --array=0-5
    --chdir="$repo_root"
    --output="${output_root}/slurm/%x_%A_%a.out"
    --error="${output_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch"
    "$repo_root" "$output_root" "$manifest_path"
    "$sample_count" "$fid_reference_count" "$loss_count"
    "$reference_artifact"
)

printf 'Mode: %s\n' "$mode"
printf 'Source runs: %s\n' "$source_root"
printf 'Output: %s\n' "$output_root"
printf 'Samples/FID reference/loss: %s/%s/%s\n' \
    "$sample_count" "$fid_reference_count" "$loss_count"
"${manifest_command[@]}"
printf 'Reference command: '
printf '%q ' "${prepare_command[@]}"
printf '\n'
printf 'Array command: '
printf '%q ' "${array_command[@]}"
printf '\n'

if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${output_root}/slurm"
"${manifest_command[@]}" --output "$manifest_path"
cp "$profiles" "${output_root}/sampling_profiles.tsv"
cp "$variants" "${output_root}/training_variants.tsv"
git -C "$repo_root" rev-parse HEAD > "${output_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src jobs/berzelius/eps_recovery_1m/eval \
    > "${output_root}/source.diff"

reference_job_output=$("${prepare_command[@]}")
reference_job=${reference_job_output%%;*}
array_command[2]="--dependency=afterok:${reference_job}"
array_job_output=$("${array_command[@]}")
array_job=${array_job_output%%;*}
combine_job_output=$(
    sbatch --parsable \
        --account="$account" \
        --dependency="afterok:${array_job}" \
        --chdir="$repo_root" \
        --output="${output_root}/slurm/%x_%j.out" \
        --error="${output_root}/slurm/%x_%j.err" \
        "${script_dir}/combine.sbatch" "$repo_root" "$output_root"
)
combine_job=${combine_job_output%%;*}
printf '{"reference_job":"%s","array_job":"%s","combine_job":"%s"}\n' \
    "$reference_job" "$array_job" "$combine_job" \
    > "${output_root}/job_ids.json"
printf 'Submitted reference job %s, dependent array job %s, and combine job %s.\n' \
    "$reference_job" "$array_job" "$combine_job"
