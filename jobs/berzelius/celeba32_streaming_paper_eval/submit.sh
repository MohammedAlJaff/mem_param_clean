#!/bin/bash
# Freeze and submit checkpoint-compacted paper evaluation for all CelebA32 runs.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
dit_seed1=false
resume_root=
resume_group=
resume_tasks=
while (( $# )); do
    case "$1" in
        --dry-run)
            preview=true
            shift
            ;;
        --dit-seed1)
            dit_seed1=true
            shift
            ;;
        --resume-root)
            resume_root=${2:?--resume-root requires a batch path}
            shift 2
            ;;
        --group)
            resume_group=${2:?--group requires a group name}
            shift 2
            ;;
        --tasks)
            resume_tasks=${2:?--tasks requires a Slurm array selection}
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

groups=(
    dit_seed0 celeba32_unet_a celeba32_unet_b celeba32_unet_c
)
run_count=144
output_family=celeba32_streaming_full_10000
prepare_args=()
if [[ "$dit_seed1" == true ]]; then
    groups=(dit_seed1)
    run_count=18
    output_family=celeba32_dit_seed1_streaming_full_10000
    prepare_args+=(--dit-seed1)
fi
declare -A task_ranges=(
    [dit_seed1]=0-17
    [dit_seed0]=0-35
    [celeba32_unet_a]=0-35
    [celeba32_unet_b]=0-35
    [celeba32_unet_c]=0-35
)
declare -A walltimes=(
    [dit_seed1]=12:00:00
    [dit_seed0]=12:00:00
    [celeba32_unet_a]=12:00:00
    [celeba32_unet_b]=12:00:00
    [celeba32_unet_c]=12:00:00
)
declare -A constraints=(
    [dit_seed1]=thin
    [dit_seed0]=thin
    [celeba32_unet_a]=thin
    [celeba32_unet_b]=thin
    [celeba32_unet_c]=thin
)
declare -A gres=(
    [dit_seed1]=gpu:A100-SXM4-40GB:1
    [dit_seed0]=gpu:A100-SXM4-40GB:1
    [celeba32_unet_a]=gpu:A100-SXM4-40GB:1
    [celeba32_unet_b]=gpu:A100-SXM4-40GB:1
    [celeba32_unet_c]=gpu:A100-SXM4-40GB:1
)

if [[ -n "$resume_root" ]]; then
    batch_root=$(realpath "$resume_root")
    if [[ -z "$resume_group" || -z "$resume_tasks" ]]; then
        echo "A restart requires --group and --tasks." >&2
        exit 2
    fi
    if [[ -z "${task_ranges[$resume_group]+x}" ]]; then
        echo "Unknown restart group: $resume_group" >&2
        exit 2
    fi
    for required in _job/run_array.sbatch _job/variants.tsv \
        _job/snapshot.sha256 _job/common_inputs.sha256; do
        if [[ ! -f "${batch_root}/${required}" ]]; then
            echo "Missing frozen restart input: ${batch_root}/${required}" >&2
            exit 2
        fi
    done
else
    if [[ -n "$resume_group" || -n "$resume_tasks" ]]; then
        echo "--group and --tasks are valid only with --resume-root." >&2
        exit 2
    fi
    tag="${MP_CELEBA32_STREAM_EVAL_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    batch_root="${repo_root}/outputs/evals/${output_family}/${tag}"
fi

account=${MP_CELEBA32_STREAM_EVAL_ACCOUNT:-berzelius-2026-269}
qos=${MP_CELEBA32_STREAM_EVAL_QOS:-normal}

printf 'CelebA32 streaming paper evaluation\n'
printf 'Protocol: full suite, 10,000 samples, 512 held-out losses\n'
printf 'Completed runs: %s; independent arrays: %s\n' "$run_count" "${#groups[@]}"
printf 'Retention: compact after every checkpoint; one evaluator pass\n'
printf 'Account: %s; QoS: %s\n' "$account" "$qos"
printf 'Output: %s\n' "$batch_root"
if [[ -n "$resume_root" ]]; then
    printf 'Restart: group=%s tasks=%s\n' "$resume_group" "$resume_tasks"
else
    printf 'Scheduling: all selected arrays submit without dependencies or throttles\n'
fi
if [[ "$preview" == true ]]; then
    exit 0
fi

if [[ -z "$resume_root" ]]; then
    torch_fidelity_root="${repo_root}/../old_mem_par/third_party/torch-fidelity"
    if [[ ! -f "${torch_fidelity_root}/torch_fidelity/__init__.py" ]]; then
        echo "Missing torch-fidelity source: ${torch_fidelity_root}" >&2
        exit 2
    fi
    mkdir -p "${batch_root}/_job/source" "${batch_root}/_job/slurm" \
        "${batch_root}/_job/tasks" "${batch_root}/_job/submissions"
    cp "$script_dir/run_array.sbatch" "$script_dir/submit.sh" \
        "$script_dir/prepare_variants.py" "$script_dir/freeze_inputs.py" \
        "$script_dir/README.md" "${batch_root}/_job/"
    cp -a "${repo_root}/mem_param_src" "${batch_root}/_job/source/mem_param_src"
    cp -a "${torch_fidelity_root}/torch_fidelity" \
        "${batch_root}/_job/source/torch_fidelity"
    cp "${torch_fidelity_root}/LICENSE.md" \
        "${batch_root}/_job/source/torch_fidelity_LICENSE.md"
    printf '%s\n' "$repo_root" > "${batch_root}/_job/repo_root.txt"

    set +u
    module --force purge
    unset LMOD_FAMILY_MAMBA_ENV LMOD_FAMILY_MAMBA_ENV_VERSION
    module load Mambaforge
    conda run -n mem_param_env python "${batch_root}/_job/prepare_variants.py" \
        --repo-root "$repo_root" \
        --output "${batch_root}/_job/variants.tsv" "${prepare_args[@]}"
    conda run -n mem_param_env python "${batch_root}/_job/freeze_inputs.py" \
        --repo-root "$repo_root" \
        --job-root "${batch_root}/_job" \
        --variants "${batch_root}/_job/variants.tsv" \
        --workers 8 --expected-tasks "$run_count"
    conda list -n mem_param_env --explicit \
        > "${batch_root}/_job/environment_explicit.txt"

    git -C "$repo_root" rev-parse HEAD > "${batch_root}/_job/source_commit.txt"
    git -C "$repo_root" status --short > "${batch_root}/_job/source_status.txt"
    git -C "$repo_root" diff HEAD -- mem_param_src \
        jobs/berzelius/celeba32_streaming_paper_eval \
        > "${batch_root}/_job/source.diff"
    git -C "${repo_root}/outputs/model_assets/feature_f_mem/dinov2" rev-parse HEAD \
        > "${batch_root}/_job/dinov2_commit.txt"
    git -C "$torch_fidelity_root" rev-parse HEAD \
        > "${batch_root}/_job/torch_fidelity_commit.txt"
    git -C "$torch_fidelity_root" status --short --untracked-files=all -- \
        torch_fidelity > "${batch_root}/_job/torch_fidelity_status.txt"
    set -u
    git -C "$torch_fidelity_root" diff HEAD -- torch_fidelity \
        > "${batch_root}/_job/torch_fidelity.diff"
    printf 'evaluation_tier\tpaper\nsample_count\t10000\nloss_count\t512\ncheckpoint_selector\tall\nrun_count\t%s\nretention\tcheckpoint_identity_after_completion\nevaluator_passes\t1\ndependencies\tnone\n' "$run_count" \
        > "${batch_root}/_job/evaluation_contract.tsv"
    (
        cd "$batch_root"
        find _job -type f ! -path '_job/snapshot.sha256' -print0 \
            | sort -z | xargs -0 sha256sum > _job/snapshot.sha256
    )
fi

submit_group() {
    group=$1
    tasks=$2
    dependency=${3:-}
    command=(
        sbatch --parsable
        --array="$tasks"
        --account="$account"
        --time="${walltimes[$group]}"
        --qos="$qos"
        --constraint="${constraints[$group]}"
        --gres="${gres[$group]}"
        --cpus-per-task=8
        --mem=64G
        --no-requeue
        --job-name="eval_${group}"
        --chdir="$batch_root"
        --output="${batch_root}/_job/slurm/${group}_%A_%a.out"
        --error="${batch_root}/_job/slurm/${group}_%A_%a.err"
    )
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="afterok:${dependency}")
    fi
    command+=("${batch_root}/_job/run_array.sbatch" "$batch_root" "$group")
    submission=$("${command[@]}")
    job_id=${submission%%;*}
    submitted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$submitted_at" "$group" "$job_id" "$tasks" \
        "${walltimes[$group]}" "${dependency:--}" \
        >> "${batch_root}/_job/submissions/submission.tsv"
    printf '%s\n' "$job_id"
}

if [[ -n "$resume_root" ]]; then
    job_id=$(submit_group "$resume_group" "$resume_tasks")
    printf 'Submitted restart %s as job %s.\n' "$resume_group" "$job_id"
    exit 0
fi

for group in "${groups[@]}"; do
    job_id=$(submit_group "$group" "${task_ranges[$group]}")
    printf 'Submitted independent group %s as job %s.\n' "$group" "$job_id"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "${batch_root}/_job/SUBMITTED"
printf 'Frozen batch: %s\n' "$batch_root"
