#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VLMEVAL_DIR="$REPO_ROOT/Benchmark/VLMEvalKit"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_single_test.py"
RUNTIME_SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_tracer_runtime.py"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for required_name in MODEL_NAME MODEL_CONFIG CHECKPOINT USE_DLLM WORK_ROOT; do
    [[ -n "${!required_name:-}" ]] || fail "$required_name is required"
done

case "$USE_DLLM" in
    true|false) ;;
    *) fail "USE_DLLM must be true or false, got: $USE_DLLM" ;;
esac

for tracer_flag in TRACER_ROUTER_ENABLED TRACER_POLICY_ENABLED; do
    if [[ -n "${!tracer_flag:-}" ]]; then
        case "${!tracer_flag}" in
            true|false) ;;
            *) fail "$tracer_flag must be true or false, got: ${!tracer_flag}" ;;
        esac
    fi
done

LMU_ROOT="${LMU_ROOT:-/nvmedata/xiexu/data/LMUData}"
if [[ -n "${LMUData:-}" && "$LMUData" != "$LMU_ROOT" ]]; then
    fail "LMUData must match LMU_ROOT: LMUData=$LMUData LMU_ROOT=$LMU_ROOT"
fi
LMUData="$LMU_ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
EVAL_PREFLIGHT_SAMPLES="${EVAL_PREFLIGHT_SAMPLES:-0}"
EVAL_PREFLIGHT_REQUIRE_EOS="${EVAL_PREFLIGHT_REQUIRE_EOS:-true}"
EVAL_PREFLIGHT_REQUIRE_NONEMPTY="${EVAL_PREFLIGHT_REQUIRE_NONEMPTY:-true}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_NPROC="${EVAL_NPROC:-1}"
EVAL_DIST_BACKEND="${EVAL_DIST_BACKEND:-nccl}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-$0}"

for parameter_name in BLOCK_SIZE DENOISING_STEPS MAX_NEW_TOKENS; do
    parameter_value="${!parameter_name}"
    [[ "$parameter_value" =~ ^[1-9][0-9]*$ ]] ||
        fail "$parameter_name must be a positive integer, got: $parameter_value"
done
[[ "$EVAL_PREFLIGHT_SAMPLES" =~ ^[0-9]+$ ]] ||
    fail "EVAL_PREFLIGHT_SAMPLES must be a non-negative integer, got: $EVAL_PREFLIGHT_SAMPLES"
[[ "$EVAL_NPROC" =~ ^[1-9][0-9]*$ ]] ||
    fail "EVAL_NPROC must be a positive integer, got: $EVAL_NPROC"
for preflight_flag in EVAL_PREFLIGHT_REQUIRE_EOS EVAL_PREFLIGHT_REQUIRE_NONEMPTY; do
    case "${!preflight_flag}" in
        true|false) ;;
        *) fail "$preflight_flag must be true or false, got: ${!preflight_flag}" ;;
    esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] ||
    fail "RUN_ID contains unsupported characters: $RUN_ID"

RUN_DIR="$WORK_ROOT/runs/$RUN_ID"
VLMEVAL_WORK_DIR="$RUN_DIR/vlmeval"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    python - "$RUN_DIR/status.json" "$state" "$MODEL_NAME" "$RUN_ID" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, model_name, run_id, exit_code = sys.argv[1:]
status_path = Path(path)
payload = {}
if status_path.exists():
    payload = json.loads(status_path.read_text(encoding='utf-8'))
payload.update({
    'schema_version': 1,
    'state': state,
    'model_name': model_name,
    'run_id': run_id,
})
now = datetime.now(timezone.utc).isoformat()
if 'created_at_utc' not in payload:
    payload['created_at_utc'] = now
if state == 'running':
    payload['started_at_utc'] = now
if state in {'completed', 'failed', 'dry_run'}:
    payload['finished_at_utc'] = now
if exit_code:
    payload['exit_code'] = int(exit_code)
status_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY

python - \
    "$RUN_DIR/runtime_contract.json" \
    "${TRACER_CELL:-legacy}" \
    "${TRACER_ROUTER_ENABLED:-unset}" \
    "${TRACER_POLICY_ENABLED:-unset}" \
    "$EVAL_PREFLIGHT_SAMPLES" \
    "$EVAL_PREFLIGHT_REQUIRE_EOS" \
    "$EVAL_PREFLIGHT_REQUIRE_NONEMPTY" \
    "$USE_DLLM" <<'PY'
import json
import sys
from pathlib import Path

(
    output, cell, router, policy, samples, require_eos,
    require_nonempty, use_dllm,
) = sys.argv[1:]
payload = {
    'schema_version': 1,
    'evaluation_protocol': 'canonical_single_test',
    'judge': 'exact_matching',
    'mmbench_parser': 'official_can_infer_vanilla_all',
    'tracer_cell': cell,
    'tracer_router_enabled': router,
    'tracer_policy_enabled': policy,
    'preflight_samples': int(samples),
    'preflight_require_eos': require_eos == 'true',
    'preflight_require_nonempty': require_nonempty == 'true',
    'runtime_diagnostics': {
        'enabled': use_dllm == 'true',
        'schema_version': 1,
        'samples_file': 'tracer_runtime_samples.jsonl',
        'summary_file': 'tracer_runtime_stats.json',
        'completeness_check': 'single_test_results.sample_count',
    },
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY
}

write_status prepared
finalized=0
on_exit() {
    local code=$?
    if [[ "$finalized" != "1" && "$DRY_RUN" != "1" ]]; then
        write_status failed "$code" || true
    fi
}
trap on_exit EXIT

python - \
    "$RUN_DIR/config.json" \
    "$MODEL_NAME" \
    "$MODEL_CONFIG" \
    "$CHECKPOINT" \
    "$USE_DLLM" \
    "$BLOCK_SIZE" \
    "$DENOISING_STEPS" \
    "$MAX_NEW_TOKENS" \
    "$EVAL_PREFLIGHT_SAMPLES" \
    "$EVAL_PREFLIGHT_REQUIRE_EOS" \
    "$EVAL_PREFLIGHT_REQUIRE_NONEMPTY" <<'PY'
import json
import sys

(
    output_path,
    model_name,
    model_config,
    checkpoint,
    use_dllm,
    block_size,
    denoising_steps,
    max_new_tokens,
    preflight_samples,
    preflight_require_eos,
    preflight_require_nonempty,
) = sys.argv[1:]
model = {
    'class': 'Harmon',
    'model_path': model_config,
    'checkpoint_path': checkpoint,
    'use_dllm': use_dllm == 'true',
}
if model['use_dllm']:
    model.update({
        'block_size': int(block_size),
        'denoising_steps': int(denoising_steps),
        'max_new_tokens': int(max_new_tokens),
        'preflight_samples': int(preflight_samples),
        'preflight_require_eos': preflight_require_eos == 'true',
        'preflight_require_nonempty': preflight_require_nonempty == 'true',
    })
config = {
    'model': {model_name: model},
    'data': {
        'MMBench_DEV_EN': {
            'class': 'ImageMCQDataset',
            'dataset': 'MMBench_DEV_EN',
        },
        'RealWorldQA': {
            'class': 'ImageMCQDataset',
            'dataset': 'RealWorldQA',
        },
        'MMVP': {
            'class': 'ImageMCQDataset',
            'dataset': 'MMVP',
        },
        'HallusionBench': {
            'class': 'ImageYORNDataset',
            'dataset': 'HallusionBench',
        },
        'VSR-zeroshot': {
            'class': 'ImageYORNDataset',
            'dataset': 'VSR-zeroshot',
        },
    },
}
with open(output_path, 'w', encoding='utf-8') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
PY

snapshot_file() {
    local source_path="$1"
    local destination_path="$2"
    if [[ -f "$source_path" ]]; then
        cp -- "$source_path" "$destination_path"
    else
        printf 'MISSING SOURCE: %s\n' "$source_path" > "$destination_path"
    fi
}

snapshot_file "$MODEL_CONFIG" "$RUN_DIR/model_config.py"
snapshot_file "$ENTRY_SCRIPT" "$RUN_DIR/entry_script.sh"
snapshot_file "$SCRIPT_DIR/eval_unimrg_common.sh" "$RUN_DIR/common_runner.sh"
snapshot_file "$SUMMARY_SCRIPT" "$RUN_DIR/summarize_single_test.py"
snapshot_file "$RUNTIME_SUMMARY_SCRIPT" \
    "$RUN_DIR/summarize_tracer_runtime.py"
snapshot_file "$VLMEVAL_DIR/vlmeval/utils/matching_util.py" \
    "$RUN_DIR/official_matching_util.py"

if [[ "$EVAL_NPROC" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q python run.py --config %q --work-dir %q --judge exact_matching --verbose\n' \
        "$CUDA_VISIBLE_DEVICES" "$RUN_DIR/config.json" "$VLMEVAL_WORK_DIR" \
        > "$RUN_DIR/command.txt"
else
    printf 'CUDA_VISIBLE_DEVICES=%q python -m torch.distributed.run --standalone --nproc-per-node=%q run.py --config %q --work-dir %q --judge exact_matching --verbose\n' \
        "$CUDA_VISIBLE_DEVICES" "$EVAL_NPROC" "$RUN_DIR/config.json" \
        "$VLMEVAL_WORK_DIR" > "$RUN_DIR/command.txt"
fi

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_head.txt" 2>/dev/null ||
    printf '%s\n' 'UNAVAILABLE' > "$RUN_DIR/git_head.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt" 2>/dev/null || true
git -C "$REPO_ROOT" diff --binary --no-ext-diff > "$RUN_DIR/git_diff.patch" 2>/dev/null || true

{
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'requested_conda_env=%s\n' "${CONDA_ENV:-harmon}"
    printf 'launcher_python='; python --version 2>&1
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'eval_nproc=%s\n' "$EVAL_NPROC"
    printf 'eval_dist_backend=%s\n' "$EVAL_DIST_BACKEND"
    printf 'lmu_root=%s\n' "$LMU_ROOT"
    printf 'lmu_data=%s\n' "$LMUData"
    printf 'dry_run=%s\n' "$DRY_RUN"
    printf 'evaluation_protocol=canonical_single_test\n'
    printf 'judge=exact_matching\n'
    printf 'tracer_binding_mode=%s\n' "${TRACER_BINDING_MODE:-default}"
    printf 'tracer_cell=%s\n' "${TRACER_CELL:-legacy}"
    printf 'tracer_router_enabled=%s\n' "${TRACER_ROUTER_ENABLED:-unset}"
    printf 'tracer_policy_enabled=%s\n' "${TRACER_POLICY_ENABLED:-unset}"
    printf 'eval_preflight_samples=%s\n' "$EVAL_PREFLIGHT_SAMPLES"
    printf 'eval_preflight_require_eos=%s\n' "$EVAL_PREFLIGHT_REQUIRE_EOS"
    printf 'eval_preflight_require_nonempty=%s\n' "$EVAL_PREFLIGHT_REQUIRE_NONEMPTY"
    printf 'runtime_diagnostics=%s\n' "$USE_DLLM"
    printf 'official_eval_frozen=%s\n' "${OFFICIAL_EVAL_FROZEN:-unset}"
    printf 'tracer_strict_reproducibility=%s\n' "${TRACER_STRICT_REPRODUCIBILITY:-unset}"
    printf 'eval_seed=%s\n' "${EVAL_SEED:-unset}"
    printf 'pythonhashseed=%s\n' "${PYTHONHASHSEED:-unset}"
    printf 'cublas_workspace_config=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-unset}"
    if [[ "$USE_DLLM" == "true" ]]; then
        printf 'runtime_diagnostics_samples=%s\n' \
            "$RUN_DIR/tracer_runtime_samples.jsonl"
        printf 'runtime_diagnostics_summary=%s\n' \
            "$RUN_DIR/tracer_runtime_stats.json"
    fi
    python -m pip freeze 2>/dev/null || true
    nvidia-smi 2>/dev/null || true
} > "$RUN_DIR/environment.txt"

hash_path() {
    local target="$1"
    local label="$2"
    local output="$3"
    if [[ -f "$target" ]]; then
        sha256sum "$target" | sed "s#  $target\$#  $label#" >> "$output"
    elif [[ -d "$target" ]]; then
        while IFS= read -r -d '' file_path; do
            relative_path="${file_path#"$target"/}"
            sha256sum "$file_path" | sed "s#  $file_path\$#  $label/$relative_path#" >> "$output"
        done < <(
            find "$target" -type f \
                ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z
        )
    else
        printf 'MISSING  %s\n' "$label" >> "$output"
    fi
}

: > "$RUN_DIR/code.sha256"
hash_path "$ENTRY_SCRIPT" "entry_script:$ENTRY_SCRIPT" "$RUN_DIR/code.sha256"
hash_path "$SCRIPT_DIR/eval_unimrg_common.sh" "common_runner:$SCRIPT_DIR/eval_unimrg_common.sh" "$RUN_DIR/code.sha256"
hash_path "$SUMMARY_SCRIPT" "summarizer:$SUMMARY_SCRIPT" "$RUN_DIR/code.sha256"
hash_path "$RUNTIME_SUMMARY_SCRIPT" \
    "runtime_summarizer:$RUNTIME_SUMMARY_SCRIPT" "$RUN_DIR/code.sha256"
hash_path "$MODEL_CONFIG" "model_config:$MODEL_CONFIG" "$RUN_DIR/code.sha256"
hash_path "$REPO_ROOT/Harmon/src/models/harmon_dev.py" \
    "source:Harmon/src/models/harmon_dev.py" "$RUN_DIR/code.sha256"
hash_path "$REPO_ROOT/Harmon/src/models/dllm/guard" \
    "source:Harmon/src/models/dllm/guard" "$RUN_DIR/code.sha256"
hash_path "$VLMEVAL_DIR/vlmeval/vlm/harmon.py" \
    "source:VLMEvalKit/vlmeval/vlm/harmon.py" "$RUN_DIR/code.sha256"
hash_path "$VLMEVAL_DIR/run.py" "vlmeval_runner:$VLMEVAL_DIR/run.py" "$RUN_DIR/code.sha256"
hash_path "$VLMEVAL_DIR/vlmeval/utils/matching_util.py" \
    "official_parser:$VLMEVAL_DIR/vlmeval/utils/matching_util.py" \
    "$RUN_DIR/code.sha256"

: > "$RUN_DIR/checkpoint.sha256"
hash_path "$CHECKPOINT" "checkpoint:$CHECKPOINT" "$RUN_DIR/checkpoint.sha256"
if [[ -n "${HARMON_BASE_CHECKPOINT:-}" ]]; then
    hash_path "$HARMON_BASE_CHECKPOINT" \
        "base_checkpoint:$HARMON_BASE_CHECKPOINT" \
        "$RUN_DIR/checkpoint.sha256"
fi
if [[ -n "${TRACER_RISK_HEAD_PATH:-}" ]]; then
    hash_path "$TRACER_RISK_HEAD_PATH" \
        "tracer_risk_head:$TRACER_RISK_HEAD_PATH" \
        "$RUN_DIR/checkpoint.sha256"
fi
if [[ -n "${TRACER_RISK_AUDIT_PATH:-}" ]]; then
    hash_path "$TRACER_RISK_AUDIT_PATH" \
        "tracer_risk_audit:$TRACER_RISK_AUDIT_PATH" \
        "$RUN_DIR/checkpoint.sha256"
fi

: > "$RUN_DIR/datasets.sha256"
for dataset in MMBench_DEV_EN RealWorldQA MMVP HallusionBench VSR-zeroshot; do
    hash_path "$LMU_ROOT/$dataset.tsv" \
        "dataset:$LMU_ROOT/$dataset.tsv" \
        "$RUN_DIR/datasets.sha256"
done

printf 'Run directory: %s\n' "$RUN_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
    write_status dry_run 0
    finalized=1
    cat "$RUN_DIR/config.json"
    cat "$RUN_DIR/command.txt"
    exit 0
fi

[[ -f "$MODEL_CONFIG" ]] || fail "model config not found: $MODEL_CONFIG"
[[ -e "$CHECKPOINT" ]] || fail "checkpoint not found: $CHECKPOINT"
[[ -f "$VLMEVAL_DIR/run.py" ]] || fail "VLMEvalKit runner not found: $VLMEVAL_DIR/run.py"
[[ -f "$SUMMARY_SCRIPT" ]] || fail "summary script not found: $SUMMARY_SCRIPT"
for dataset in MMBench_DEV_EN RealWorldQA MMVP HallusionBench VSR-zeroshot; do
    [[ -f "$LMU_ROOT/$dataset.tsv" ]] ||
        fail "dataset file not found: $LMU_ROOT/$dataset.tsv"
done
if [[ -n "${HARMON_BASE_CHECKPOINT:-}" ]]; then
    [[ -f "$HARMON_BASE_CHECKPOINT" ]] ||
        fail "base checkpoint not found: $HARMON_BASE_CHECKPOINT"
fi

resolved_conda_sh="${CONDA_SH:-}"
if [[ -z "$resolved_conda_sh" ]]; then
    for candidate in \
        "${HOME:-/home/xiexu}/anaconda3/etc/profile.d/conda.sh" \
        "${HOME:-/home/xiexu}/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
            resolved_conda_sh="$candidate"
            break
        fi
    done
fi
[[ -f "$resolved_conda_sh" ]] ||
    fail "conda initialization script not found; set CONDA_SH explicitly"
PS1="${PS1:-}"
# shellcheck source=/dev/null
source "$resolved_conda_sh"
conda activate "${CONDA_ENV:-harmon}"

{
    printf '\n[activated_environment]\n'
    printf 'conda_default_env=%s\n' "${CONDA_DEFAULT_ENV:-}"
    printf 'python_executable='
    python -c 'import sys; print(sys.executable)'
    printf 'python='; python --version 2>&1
    python - <<'PY'
import torch

print('torch={}'.format(torch.__version__))
print('torch_cuda={}'.format(torch.version.cuda))
PY
    python -m pip freeze 2>/dev/null || true
} >> "$RUN_DIR/environment.txt"

export PYTHONPATH="$REPO_ROOT/Harmon${PYTHONPATH:+:$PYTHONPATH}"
export LMU_ROOT LMUData CUDA_VISIBLE_DEVICES
export VLMEVAL_DIST_BACKEND="$EVAL_DIST_BACKEND"
export TRACER_CELL TRACER_ROUTER_ENABLED TRACER_POLICY_ENABLED
if [[ "$USE_DLLM" == "true" ]]; then
    HARMON_RUNTIME_STATS_PATH="$RUN_DIR/tracer_runtime_samples.jsonl"
    export HARMON_RUNTIME_STATS_PATH
fi
if [[ -n "${HARMON_BASE_CHECKPOINT:-}" ]]; then
    export HARMON_BASE_CHECKPOINT
fi

write_status running
mkdir -p "$VLMEVAL_WORK_DIR"
set +e
if [[ "$EVAL_NPROC" == "1" ]]; then
    (
        cd "$VLMEVAL_DIR"
        python run.py \
            --config "$RUN_DIR/config.json" \
            --work-dir "$VLMEVAL_WORK_DIR" \
            --judge exact_matching \
            --verbose
    ) 2>&1 | tee "$RUN_DIR/eval.log"
else
    (
        cd "$VLMEVAL_DIR"
        python -m torch.distributed.run \
            --standalone --nproc-per-node="$EVAL_NPROC" run.py \
            --config "$RUN_DIR/config.json" \
            --work-dir "$VLMEVAL_WORK_DIR" \
            --judge exact_matching \
            --verbose
    ) 2>&1 | tee "$RUN_DIR/eval.log"
fi
run_code=${PIPESTATUS[0]}
printf '{"stage":"inference","exit_code":%s}\n' "$run_code" >> "$RUN_DIR/stage_exit_codes.jsonl"
set -e

if [[ "$run_code" == "0" ]]; then
    set +e
    python "$SUMMARY_SCRIPT" \
        --run-dir "$VLMEVAL_WORK_DIR" \
        --model-name "$MODEL_NAME" \
        --output-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/postprocess.log"
    summary_code=${PIPESTATUS[0]}
    printf '{"stage":"score_summary","exit_code":%s}\n' "$summary_code" >> "$RUN_DIR/stage_exit_codes.jsonl"
    set -e
    if [[ "$summary_code" != "0" ]]; then
        run_code="$summary_code"
    fi
fi

if [[ "$run_code" == "0" && "$USE_DLLM" == "true" ]]; then
    if [[ ! -s "$RUN_DIR/tracer_runtime_samples.jsonl" ]]; then
        printf 'ERROR: dLLM runtime diagnostics are missing or empty: %s\n' \
            "$RUN_DIR/tracer_runtime_samples.jsonl" >&2
        run_code=1
    else
        set +e
        python "$RUNTIME_SUMMARY_SCRIPT" \
            --input "$RUN_DIR/tracer_runtime_samples.jsonl" \
            --output "$RUN_DIR/tracer_runtime_stats.json" \
            --expected-results "$RUN_DIR/single_test_results.json" 2>&1 | tee -a "$RUN_DIR/postprocess.log"
        runtime_summary_code=${PIPESTATUS[0]}
        printf '{"stage":"runtime_summary","exit_code":%s}\n' "$runtime_summary_code" >> "$RUN_DIR/stage_exit_codes.jsonl"
        set -e
        if [[ "$runtime_summary_code" != "0" ]]; then
            run_code="$runtime_summary_code"
        fi
    fi
fi

if [[ "$run_code" == "0" ]]; then
    find "$VLMEVAL_WORK_DIR" -type f -printf '%s  %p\n' | sort \
        > "$RUN_DIR/result_files.txt"
    write_status completed 0
    finalized=1
    exit 0
fi

write_status failed "$run_code"
finalized=1
exit "$run_code"
