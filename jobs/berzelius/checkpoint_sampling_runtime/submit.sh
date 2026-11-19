#!/bin/bash
# Freeze and submit one short sampling-only runtime benchmark.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
default_run="${repo_root}/outputs/job_sets/celeba32_unet_3x3_1m/20260927T191224Z_2530324/runs/18_n2048_x_x_s0"

preview=false
run_dir="$default_run"
checkpoint_step=1000000
measure_count=10000
target_count=10000
batch_size=0
while (( $# )); do
    case "$1" in
        --dry-run)
            preview=true
            shift
            ;;
        --run-dir)
            run_dir=${2:?--run-dir requires a path}
            shift 2
            ;;
        --checkpoint-step)
            checkpoint_step=${2:?--checkpoint-step requires an integer}
            shift 2
            ;;
        --measure-count)
            measure_count=${2:?--measure-count requires an integer}
            shift 2
            ;;
        --target-count)
            target_count=${2:?--target-count requires an integer}
            shift 2
            ;;
        --batch-size)
            batch_size=${2:?--batch-size requires an integer}
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

run_dir=$(realpath "$run_dir")
checkpoint_name=$(printf 'step_%06d.pt' "$checkpoint_step")
for required in config.json run.json data_bundle.pt data_manifest.json z_init.pt "checkpoints/${checkpoint_name}"; do
    if [[ ! -f "${run_dir}/${required}" ]]; then
        echo "Missing saved-run artifact: ${run_dir}/${required}" >&2
        exit 2
    fi
done

account=${MP_SAMPLING_BENCHMARK_ACCOUNT:-berzelius-2026-269}
walltime=${MP_SAMPLING_BENCHMARK_TIME:-00:30:00}
qos=${MP_SAMPLING_BENCHMARK_QOS:-devel}
time_budget=${MP_SAMPLING_BENCHMARK_BUDGET_SECONDS:-1440}
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/checkpoint_sampling_runtime/${tag}"
command=(
    sbatch --parsable
    --account="$account"
    --time="$walltime"
    --qos="$qos"
    --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1
    --cpus-per-task=4
    --mem=32G
    --signal=B:USR1@180
    --no-requeue
    --chdir="$batch_root"
    --output="${batch_root}/slurm/%x_%j.out"
    --error="${batch_root}/slurm/%x_%j.err"
    "${batch_root}/job/run.sbatch"
    "$batch_root"
    "$run_dir"
    "$checkpoint_step"
    "$measure_count"
    "$target_count"
    "$batch_size"
    "$time_budget"
)

printf 'Checkpoint sampling runtime benchmark\n'
printf 'Run: %s\n' "$run_dir"
printf 'Checkpoint: %s\n' "$checkpoint_step"
printf 'Measure/target samples: %s/%s\n' "$measure_count" "$target_count"
printf 'Wall time: %s; sampling budget: %s seconds\n' "$walltime" "$time_budget"
printf 'Output: %s\n' "$batch_root"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/job" "${batch_root}/slurm" \
    "${batch_root}/commands" "${batch_root}/results" "${batch_root}/source"
cp "${script_dir}/run.sbatch" "${script_dir}/submit.sh" \
    "${script_dir}/README.md" "${batch_root}/job/"
cp -a "${repo_root}/mem_param_src" "${batch_root}/source/mem_param_src"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" status --short > "${batch_root}/source_status.txt"
git -C "$repo_root" diff HEAD -- mem_param_src jobs/berzelius/checkpoint_sampling_runtime \
    > "${batch_root}/source.diff"
set +u
module load Mambaforge
conda list -n mem_param_env --explicit > "${batch_root}/environment_explicit.txt"
set -u
printf 'run_dir\t%s\ncheckpoint_step\t%s\nmeasure_count\t%s\ntarget_count\t%s\nbatch_size\t%s\ntime_budget_seconds\t%s\n' \
    "$run_dir" "$checkpoint_step" "$measure_count" "$target_count" \
    "$batch_size" "$time_budget" > "${batch_root}/request.tsv"
sha256sum "${run_dir}/config.json" "${run_dir}/run.json" \
    "${run_dir}/data_bundle.pt" "${run_dir}/data_manifest.json" \
    "${run_dir}/z_init.pt" "${run_dir}/checkpoints/${checkpoint_name}" \
    > "${batch_root}/input.sha256"
(cd "$batch_root" && find job source -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256 && \
    sha256sum source_commit.txt source_status.txt source.diff \
    environment_explicit.txt request.tsv input.sha256 >> snapshot.sha256)
printf '%q ' "${command[@]}" > "${batch_root}/commands/submit.sh"
printf '\n' >> "${batch_root}/commands/submit.sh"

submission=$("${command[@]}")
job_id=${submission%%;*}
printf 'checkpoint_sampling_runtime\t%s\t%s\t%s\n' \
    "$job_id" "$walltime" "$batch_root" > "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
printf 'Submitted job %s.\n' "$job_id"
printf 'Frozen batch: %s\n' "$batch_root"
