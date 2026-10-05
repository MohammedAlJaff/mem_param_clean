#!/bin/bash
# The researcher launches this script. --dry-run prints commands and writes nothing.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
seed0_root="${repo_root}/outputs/job_sets/eps_recovery_1m/20260914T113123Z_2308577/runs"
assets_manifest="${repo_root}/outputs/model_assets/feature_f_mem/manifest.json"
reference_artifact="${repo_root}/outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point/reference/celeba32_valid_seed1_n10000.pt"
noise_artifact="${repo_root}/outputs/evals/eps_recovery_1m/20260915T111553Z_3025872_three_point/cache/noise/seed2003_n10000_1x32x32.pt"
scope=seed0
multiseed_root=""
preview=false

usage() {
    echo "Usage: bash submit.sh [--dry-run] [--scope seed0|all]" >&2
    echo "       [--multiseed-root PATH] [--assets-manifest PATH]" >&2
    echo "       [--reference-artifact PATH] [--noise-artifact PATH]" >&2
}
while (( $# )); do
    case "$1" in
        --dry-run) preview=true; shift ;;
        --scope|--multiseed-root|--assets-manifest|--reference-artifact|--noise-artifact)
            if (( $# < 2 )); then usage; exit 2; fi
            case "$1" in
                --scope) scope=$2 ;;
                --multiseed-root) multiseed_root=$2 ;;
                --assets-manifest) assets_manifest=$2 ;;
                --reference-artifact) reference_artifact=$2 ;;
                --noise-artifact) noise_artifact=$2 ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
if [[ "$scope" != seed0 && "$scope" != all ]]; then
    echo "--scope must be seed0 or all." >&2
    exit 2
fi
if [[ "$scope" == all && -z "$multiseed_root" ]]; then
    echo "--multiseed-root is required for --scope all." >&2
    exit 2
fi

account="${MP_FEATURE_FMEM_ACCOUNT:-berzelius-2026-269}"
walltime="${MP_FEATURE_FMEM_TIME:-12:00:00}"
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$_${scope}"
output_root="${repo_root}/outputs/evals/eps_recovery_feature_f_mem/${tag}"
manifest="${output_root}/manifests/evaluation.json"
last_task=2
if [[ "$scope" == all ]]; then last_task=8; fi
manifest_command=(
    python3 "${script_dir}/evaluate_feature_f_mem.py" manifest
    --seed0-root "$seed0_root" --scope "$scope"
)
if [[ -n "$multiseed_root" ]]; then
    manifest_command+=(--multiseed-root "$multiseed_root")
fi
prepare_command=(
    sbatch --parsable --account="$account" --chdir="$repo_root"
    --output="${output_root}/slurm/%x_%j.out"
    --error="${output_root}/slurm/%x_%j.err"
    "${script_dir}/prepare_reference.sbatch"
    "$repo_root" "$output_root" "$assets_manifest" "$reference_artifact"
)
array_command=(
    sbatch --parsable --account="$account" --time="$walltime"
    --dependency=afterok:REFERENCE_JOB_ID --array="0-${last_task}"
    --chdir="$repo_root"
    --output="${output_root}/slurm/%x_%A_%a.out"
    --error="${output_root}/slurm/%x_%A_%a.err"
    "${script_dir}/run_array.sbatch"
    "$repo_root" "$output_root" "$manifest" "$assets_manifest"
    "$reference_artifact" "$noise_artifact"
)
printf 'Scope: %s (%s trajectories, three checkpoints each)\n' "$scope" "$((last_task + 1))"
printf 'Output: %s\n' "$output_root"
printf 'Manifest command: '
printf '%q ' "${manifest_command[@]}"
printf '\nReference command: '
printf '%q ' "${prepare_command[@]}"
printf '\nArray command: '
printf '%q ' "${array_command[@]}"
printf '\n'
PYTHONPATH="$repo_root" "${manifest_command[@]}" >/dev/null
if [[ "$preview" == true ]]; then
    if [[ ! -f "$assets_manifest" ]]; then
        printf 'Prerequisite not staged yet: %s\n' "$assets_manifest"
        printf 'Run: module load Mambaforge && conda run -n mem_param_env python %q stage --output-root %q\n' \
            "${script_dir}/stage_encoder_assets.py" "$(dirname -- "$assets_manifest")"
    fi
    exit 0
fi
for path in "$assets_manifest" "$reference_artifact" "$noise_artifact"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing prerequisite artifact: $path" >&2
        exit 2
    fi
done
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-}"
module load Mambaforge
conda run -n mem_param_env python "${script_dir}/stage_encoder_assets.py" verify \
    --manifest "$assets_manifest"
mkdir -p "${output_root}/slurm" "${output_root}/manifests/workflow_source"
"${manifest_command[@]}" --output "$manifest"
cp "$assets_manifest" "${output_root}/manifests/encoder_assets.json"
cp "${script_dir}/evaluate_feature_f_mem.py" \
    "${script_dir}/combine_feature_f_mem.py" \
    "${script_dir}/stage_encoder_assets.py" \
    "${script_dir}/prepare_reference.sbatch" \
    "${script_dir}/run_array.sbatch" \
    "${script_dir}/combine.sbatch" \
    "${script_dir}/README.md" "${output_root}/manifests/workflow_source/"
git -C "$repo_root" rev-parse HEAD > "${output_root}/source_commit.txt"
git -C "$repo_root" diff HEAD -- mem_param_src jobs/berzelius/eps_recovery_1m/feature_f_mem \
    > "${output_root}/source.diff"
reference_output=$("${prepare_command[@]}")
reference_job=${reference_output%%;*}
array_command[4]="--dependency=afterok:${reference_job}"
array_output=$("${array_command[@]}")
array_job=${array_output%%;*}
combine_output=$(
    sbatch --parsable --account="$account" --dependency="afterok:${array_job}" \
        --chdir="$repo_root" \
        --output="${output_root}/slurm/%x_%j.out" \
        --error="${output_root}/slurm/%x_%j.err" \
        "${script_dir}/combine.sbatch" "$repo_root" "$output_root" "$manifest"
)
combine_job=${combine_output%%;*}
printf '{"reference_job":"%s","array_job":"%s","combine_job":"%s"}\n' \
    "$reference_job" "$array_job" "$combine_job" > "${output_root}/job_ids.json"
printf 'Submitted reference %s, evaluation array %s, and combine %s.\n' \
    "$reference_job" "$array_job" "$combine_job"
