#!/bin/bash
# Freeze and submit one MNIST JointAE-32 encoder training job.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
if [[ "${1:-}" == "--dry-run" ]]; then
    preview=true
    shift
fi
if (( $# != 0 )); then
    echo "Usage: bash submit.sh [--dry-run]" >&2
    exit 2
fi

data_root=${MP_MNIST_DATA_ROOT:-${repo_root}/../old_mem_par/data}
if [[ ! -d "${data_root}/MNIST/raw" ]]; then
    echo "Missing MNIST cache under ${data_root}; refusing submission." >&2
    exit 2
fi

account=${MP_MNIST_ENCODER_ACCOUNT:-berzelius-2026-269}
walltime=${MP_MNIST_ENCODER_TIME:-01:00:00}
tag="$(date -u +%Y%m%dT%H%M%SZ)_$$"
batch_root="${repo_root}/outputs/job_sets/mnist_jointae32/${tag}"
command=(
    sbatch --parsable
    --account="$account"
    --time="$walltime"
    --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1
    --cpus-per-task=4
    --mem=32G
    --no-requeue
    --chdir="$batch_root"
    --output="${batch_root}/slurm/%x_%j.out"
    --error="${batch_root}/slurm/%x_%j.err"
    "${batch_root}/job/train.sbatch"
    "$repo_root"
    "$batch_root"
)

printf 'MNIST JointAE-32 training job\n'
printf 'Split: 55k train / 5k validation; official 10k test is reference only.\n'
printf 'Output: %s\n' "$batch_root"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

mkdir -p "${batch_root}/job" "${batch_root}/slurm" \
    "${batch_root}/commands" "${batch_root}/source"
cp "${script_dir}/train.py" "${script_dir}/train.sbatch" \
    "${script_dir}/submit.sh" "${script_dir}/README.md" \
    "${batch_root}/job/"
cp "${repo_root}/notebooks/sanity/mnist_jointae_fd_sanity.ipynb" \
    "${batch_root}/source/"
git -C "$repo_root" rev-parse HEAD > "${batch_root}/source_commit.txt"
git -C "$repo_root" status --short > "${batch_root}/source_status.txt"
git -C "$repo_root" diff HEAD -- \
    jobs/berzelius/mnist_jointae32 \
    notebooks/sanity/mnist_jointae_fd_sanity.ipynb \
    > "${batch_root}/source.diff"
set +u
module load Mambaforge
set -u
conda list -n mem_param_env --explicit > "${batch_root}/environment_explicit.txt"
(cd "$batch_root" && find job source -type f -print0 | sort -z | \
    xargs -0 sha256sum > snapshot.sha256 && \
    sha256sum source_commit.txt source_status.txt source.diff \
    environment_explicit.txt >> snapshot.sha256)

submission=$("${command[@]}")
job_id=${submission%%;*}
printf 'mnist_jointae32\t%s\t%s\t%s\n' "$job_id" "$walltime" "$batch_root" \
    > "${batch_root}/submission.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/SUBMITTED"
printf 'Submitted job %s.\n' "$job_id"
printf 'Frozen batch: %s\n' "$batch_root"
