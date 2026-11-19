#!/bin/bash
# Freeze and submit the complete v/v loss-weighting paper evaluation.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
preview=false
resume_root=
tasks=0-17
while (( $# )); do
    case "$1" in
        --dry-run)
            preview=true
            shift
            ;;
        --resume-root)
            resume_root=${2:?--resume-root requires a batch path}
            shift 2
            ;;
        --tasks)
            tasks=${2:?--tasks requires a Slurm array selection}
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -n "$resume_root" ]]; then
    batch_root=$(realpath "$resume_root")
    for required in _job/run_array.sbatch _job/variants.tsv \
        _job/snapshot.sha256; do
        if [[ ! -f "${batch_root}/${required}" ]]; then
            echo "Missing frozen restart input: ${batch_root}/${required}" >&2
            exit 2
        fi
    done
else
    tag="${MP_VV_WEIGHTING_EVAL_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    batch_root="${repo_root}/outputs/evals/celeba32_vv_weighting_full_10000/${tag}"
fi

account=${MP_VV_WEIGHTING_EVAL_ACCOUNT:-berzelius-2026-269}
walltime=${MP_VV_WEIGHTING_EVAL_TIME:-12:00:00}
qos=${MP_VV_WEIGHTING_EVAL_QOS:-normal}
command=(
    sbatch --parsable
    --array="$tasks"
    --account="$account"
    --time="$walltime"
    --qos="$qos"
    --constraint=thin
    --gres=gpu:A100-SXM4-40GB:1
    --cpus-per-task=8
    --mem=64G
    --no-requeue
    --chdir="$batch_root"
    --output="${batch_root}/_job/slurm/%A_%a.out"
    --error="${batch_root}/_job/slurm/%A_%a.err"
    "${batch_root}/_job/run_array.sbatch"
    "$batch_root"
)

printf 'CelebA32 v/v loss-weighting paper evaluation\n'
printf 'Tasks: %s\n' "$tasks"
printf 'Protocol: 10,000 samples x 72 checkpoints x 18 runs\n'
printf 'Weightings: x_equivalent, unit, and eps_equivalent\n'
printf 'Wall time: %s; QoS: %s; account: %s\n' \
    "$walltime" "$qos" "$account"
printf 'Output: %s\n' "$batch_root"
printf 'Command: '
printf '%q ' "${command[@]}"
printf '\n'
if [[ "$preview" == true ]]; then
    exit 0
fi

if [[ -z "$resume_root" ]]; then
    noise="${repo_root}/outputs/reference_banks/evaluation_noise/canonical_gaussian_seed2003_n10000.pt"
    reference="${repo_root}/outputs/reference_banks/celeba_test_seed1_clean_n10000.pt"
    assets="${repo_root}/outputs/model_assets/feature_f_mem"
    for required in "$noise" "${noise%.pt}.json" "$reference" \
        "${reference%.pt}.json" "${assets}/manifest.json"; do
        if [[ ! -f "$required" ]]; then
            echo "Missing frozen evaluation input: $required" >&2
            exit 2
        fi
    done
    torch_fidelity_root="${repo_root}/../old_mem_par/third_party/torch-fidelity"
    if [[ ! -f "${torch_fidelity_root}/torch_fidelity/__init__.py" ]]; then
        echo "Missing torch-fidelity source: ${torch_fidelity_root}" >&2
        exit 2
    fi


    mkdir -p "${batch_root}/_job/source" "${batch_root}/_job/slurm" \
        "${batch_root}/_job/tasks" "${batch_root}/_job/submissions"
    cp "$script_dir/run_array.sbatch" "$script_dir/submit.sh" \
        "$script_dir/variants.tsv" "$script_dir/README.md" \
        "${batch_root}/_job/"
    cp -a "${repo_root}/mem_param_src" "${batch_root}/_job/source/mem_param_src"
    cp -a "${torch_fidelity_root}/torch_fidelity" \
        "${batch_root}/_job/source/torch_fidelity"
    cp "${torch_fidelity_root}/LICENSE.md" \
        "${batch_root}/_job/source/torch_fidelity_LICENSE.md"
    printf '%s\n' "$repo_root" > "${batch_root}/_job/repo_root.txt"
    git -C "$repo_root" rev-parse HEAD > "${batch_root}/_job/source_commit.txt"
    git -C "$repo_root" status --short > "${batch_root}/_job/source_status.txt"
    git -C "$repo_root" diff HEAD -- mem_param_src \
        jobs/berzelius/celeba32_vv_weighting_paper_eval \
        > "${batch_root}/_job/source.diff"
    set +u
    module load Mambaforge
    conda list -n mem_param_env --explicit \
        > "${batch_root}/_job/environment_explicit.txt"
    set -u
    printf 'evaluation_tier\tpaper\nsample_count\t10000\nloss_count\t512\ncheckpoint_selector\tall\nrun_count\t18\nweightings\tx_equivalent,unit,eps_equivalent\n' \
        > "${batch_root}/_job/evaluation_contract.tsv"

    common_list="${batch_root}/_job/common_input_paths.txt"
    printf '%s\n' "$noise" "${noise%.pt}.json" "${assets}/manifest.json" \
        "${assets}/weights/inception_v3_compat.pth" \
        "${assets}/weights/sscd_disc_mixup.torchscript.pt" \
        "${assets}/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth" \
        "${assets}/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth" \
        "${assets}/lpips/alex_v0.1.pth" "${assets}/dinov2/hubconf.py" \
        > "$common_list"

    while IFS=$'\t' read -r task_key run_path expected_checkpoints \
        reference_path n_train seed weighting; do
        if [[ "$task_key" == "task_key" ]]; then
            continue
        fi
        run_dir="${repo_root}/${run_path}"
        config_path="${run_dir}/config.json"
        run_path_json="${run_dir}/run.json"
        if ! jq -e --argjson n_train "$n_train" --argjson seed "$seed" \
            --arg weighting "$weighting" '
            .data.dataset == "celeba32_grey" and
            .data.image_size == 32 and
            .data.image_channels == 1 and
            .data.n_train == $n_train and
            .framework.prediction_space == "v" and
            .framework.loss_space == "v" and
            .framework.loss_weighting == $weighting and
            .network.arch == "unet" and
            .training.seed == $seed
        ' "$config_path" >/dev/null; then
            echo "Run configuration violates the paper contract: $run_dir" >&2
            exit 2
        fi
        if ! jq -e --argjson z_seed "$((2003 + seed))" '
            .state == "trained" and
            .final_step == 1000000 and
            .error == null and
            .z_init_seed == $z_seed
        ' "$run_path_json" >/dev/null; then
            echo "Run record is incomplete or inconsistent: $run_dir" >&2
            exit 2
        fi
        checkpoint_count=$(find "${run_dir}/checkpoints" -maxdepth 1 \
            -name 'step_*.pt' | wc -l)
        if [[ "$checkpoint_count" -ne "$expected_checkpoints" ]]; then
            echo "Expected ${expected_checkpoints} checkpoints in $run_dir; found ${checkpoint_count}." >&2
            exit 2
        fi
        input_list="${batch_root}/_job/source_input_paths_${task_key}.txt"
        printf '%s\n' "$config_path" "$run_path_json" \
            "${run_dir}/data_bundle.pt" "${run_dir}/data_manifest.json" \
            "${run_dir}/z_init.pt" > "$input_list"
        find "${run_dir}/checkpoints" -maxdepth 1 -name 'step_*.pt' -print \
            | sort -V >> "$input_list"
        printf '%s\n' "${repo_root}/${reference_path}" \
            "${repo_root}/${reference_path%.pt}.json" >> "$input_list"
        cat "$common_list" >> "$input_list"
        sort -u "$input_list" -o "$input_list"
        while read -r path; do
            if [[ ! -f "$path" ]]; then
                echo "Missing source input: $path" >&2
                exit 2
            fi
        done < "$input_list"
        xargs sha256sum < "$input_list" \
            > "${batch_root}/_job/source_inputs_${task_key}.sha256"
    done < "$script_dir/variants.tsv"

    git -C "${assets}/dinov2" rev-parse HEAD \
        > "${batch_root}/_job/dinov2_commit.txt"
    torch_fidelity_root="${repo_root}/../old_mem_par/third_party/torch-fidelity"
    git -C "$torch_fidelity_root" rev-parse HEAD \
        > "${batch_root}/_job/torch_fidelity_commit.txt"
    git -C "$torch_fidelity_root" status --short --untracked-files=all -- \
        torch_fidelity > "${batch_root}/_job/torch_fidelity_status.txt"
    git -C "$torch_fidelity_root" diff HEAD -- torch_fidelity \
        > "${batch_root}/_job/torch_fidelity.diff"
    (cd "$batch_root" && find _job -type f \
        ! -path '_job/snapshot.sha256' -print0 \
        | sort -z | xargs -0 sha256sum > _job/snapshot.sha256)
fi

submission=$("${command[@]}")
job_id=${submission%%;*}
submitted_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '%s\t%s\t%s\t%s\n' "$submitted_at" "$job_id" "$tasks" "$walltime" \
    >> "${batch_root}/_job/submissions/submission.tsv"
printf '%s\n' "$submitted_at" > "${batch_root}/_job/SUBMITTED"
printf 'Submitted array job %s.\n' "$job_id"
printf 'Frozen batch: %s\n' "$batch_root"
