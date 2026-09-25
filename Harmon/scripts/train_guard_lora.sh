#!/usr/bin/env bash
# Stable/GUARD/TRACER LoRA training launcher (conda env `harmon`).
#
# Mirrors the eval runner's DRY_RUN manifest discipline: every real run gets a
# non-overwritable RUN_ID directory with config/env/git/hash snapshots and a
# status.json; DRY_RUN=1 writes the manifest and validates the config loads
# without launching torchrun.
#
# Usage:
#   DRY_RUN=1 bash scripts/train_guard_lora.sh                 # manifest only
#   STAGE=stable bash scripts/train_guard_lora.sh              # 20k S1 stable run
#   STAGE=guard_trajectory bash scripts/train_guard_lora.sh     # P5 §5.6-5.7 (Path C)
#   STAGE=tracer_full SEED=41 bash scripts/train_guard_lora.sh  # TRACER 20k run
#
# Environment:
#   STAGE                       stable (default) | guard_trajectory | tracer_full
#   CUDA_VISIBLE_DEVICES        default 2,3
#   NPROC_PER_NODE              default 2
#   SEED                         default 42
#   DRY_RUN                      default 0
#   RUN_ID                       default UTC timestamp + PID
#   HARMON_GUARD_EXCLUSION_MANIFEST  default points to Task 3 artifact
#   WORK_ROOT                    default /nvmedata/xiexu/uni/work_dirs/guard_training
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
HARMON_ROOT="$REPO_ROOT/Harmon"
TRAIN_SCRIPT="$HARMON_ROOT/scripts/train.py"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

STAGE="${STAGE:-stable}"
case "$STAGE" in
    stable)
        MODEL_CONFIG="$HARMON_ROOT/configs/examples/UniMRG_dllm_lora_stable_opt.py"
        ;;
    guard_trajectory)
        # P5 (spec §5.6-5.7, Path C main line): lagged conditional rank
        # field + risk-aware commit/revision policy.  Hot-starts from the
        # S1 stable checkpoint (set via ``load_from`` in the config) and
        # loads the frozen RiskHead produced by ``run_p4_risk.sh``.
        MODEL_CONFIG="$HARMON_ROOT/configs/examples/UniMRG_dllm_lora_guard_trajectory.py"
        ;;
    tracer_full)
        MODEL_CONFIG="$HARMON_ROOT/configs/examples/UniMRG_dllm_lora_tracer_full.py"
        ;;
    *)
        fail "STAGE must be stable|guard_trajectory|tracer_full, got: $STAGE"
        ;;
esac

if [[ "$STAGE" == "tracer_full" ]]; then
    # Keep the TRACER defaults beside the real launcher so a second wrapper
    # script is not needed for the exp7 training entry point.
    export TRACER_RISK_HEAD_PATH="${TRACER_RISK_HEAD_PATH:-/nvmedata/xiexu/data/uni/tracer_risk_v2/runs/tracer-v2-exp7-deterministic-risk-s42-v2/risk/risk_head.pt}"
    export TRACER_RISK_AUDIT_PATH="${TRACER_RISK_AUDIT_PATH:-/nvmedata/xiexu/data/uni/tracer_risk_v2/runs/tracer-v2-exp7-deterministic-risk-s42-v2/audit/risk_audit_report.json}"
    export TRACER_BASE_CHECKPOINT="${TRACER_BASE_CHECKPOINT:-/nvmedata/xiexu/uni/work_dirs/guard_training/runs/stable-seed42-20260809-v1/work/iter_20000.pth}"
    default_work_root=/nvmedata/xiexu/uni/work_dirs/tracer_lora_training
    default_strict_reproducibility=1
else
    default_work_root=/nvmedata/xiexu/uni/work_dirs/guard_training
    default_strict_reproducibility=0
fi

EXCLUSION_MANIFEST="${HARMON_GUARD_EXCLUSION_MANIFEST:-/nvmedata/xiexu/data/uni/guard_exclusions/train_exclusions.json}"
# MUST export so the python -m torch.distributed.run subprocess inherits the
# GPU selection.  Plain assignment (without export) leaves the child process
# unaware and it falls back to the default device order (physical cards 0,1).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
WORK_ROOT="${WORK_ROOT:-$default_work_root}"
CONDA_ENV="${CONDA_ENV:-harmon}"
MASTER_PORT="${MASTER_PORT:-29501}"
ENTRY_SCRIPT="${ENTRY_SCRIPT:-${BASH_SOURCE[0]}}"
TRACER_STRICT_REPRODUCIBILITY="${TRACER_STRICT_REPRODUCIBILITY:-$default_strict_reproducibility}"
TRACER_VERIFY_DATA_MANIFEST="${TRACER_VERIFY_DATA_MANIFEST:-1}"
TRACER_DATA_MANIFEST="${TRACER_DATA_MANIFEST:-}"
HARMON_DATA_ROOT="${HARMON_DATA_ROOT:-/nvmedata/xiexu/data/LLaVA-Instruct-150K-UniMRG}"

[[ -f "$MODEL_CONFIG" ]] || fail "model config not found: $MODEL_CONFIG"
[[ -f "$TRAIN_SCRIPT" ]] || fail "train script not found: $TRAIN_SCRIPT"
[[ -f "$EXCLUSION_MANIFEST" ]] || fail "exclusion manifest not found: $EXCLUSION_MANIFEST"
[[ -f "${EXCLUSION_MANIFEST}.sha256" ]] || fail "exclusion manifest SHA256 sidecar missing"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "RUN_ID contains unsupported characters: $RUN_ID"
[[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || fail "NPROC_PER_NODE must be a positive integer"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer"
case "$DRY_RUN" in 0|1) ;; *) fail "DRY_RUN must be 0 or 1" ;; esac
case "$TRACER_STRICT_REPRODUCIBILITY" in
    0|1) ;;
    *) fail "TRACER_STRICT_REPRODUCIBILITY must be 0 or 1" ;;
esac
case "$TRACER_VERIFY_DATA_MANIFEST" in
    0|1) ;;
    *) fail "TRACER_VERIFY_DATA_MANIFEST must be 0 or 1" ;;
esac

if [[ "$STAGE" == "tracer_full" && "$DRY_RUN" != "1" ]]; then
    [[ -f "${TRACER_RISK_HEAD_PATH:-}" ]] ||
        fail "TRACER_RISK_HEAD_PATH is missing: ${TRACER_RISK_HEAD_PATH:-unset}"
    [[ -f "${TRACER_RISK_AUDIT_PATH:-}" ]] ||
        fail "TRACER_RISK_AUDIT_PATH is missing: ${TRACER_RISK_AUDIT_PATH:-unset}"
    [[ -f "${TRACER_BASE_CHECKPOINT:-}" ]] ||
        fail "TRACER_BASE_CHECKPOINT is missing: ${TRACER_BASE_CHECKPOINT:-unset}"
    python - "${TRACER_RISK_AUDIT_PATH}" "${TRACER_RISK_HEAD_PATH}" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    report = json.load(handle)
if report.get('go_no_go') != 'PASS':
    raise SystemExit('TRACER risk audit is not PASS: {}'.format(
        report.get('go_no_go')))
digest = hashlib.sha256()
with open(sys.argv[2], 'rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
expected = report.get('provenance', {}).get('risk_head_sha256')
if not expected or digest.hexdigest() != expected:
    raise SystemExit('TRACER RiskHead SHA256 does not match risk audit')
PY
    if [[ "$TRACER_STRICT_REPRODUCIBILITY" == "1" ]]; then
        [[ -f "$TRACER_DATA_MANIFEST" ]] ||
            fail "strict TRACER training requires TRACER_DATA_MANIFEST"
        [[ -d "$HARMON_DATA_ROOT" ]] ||
            fail "HARMON_DATA_ROOT is missing: $HARMON_DATA_ROOT"
    fi
fi

if [[ "$STAGE" == "tracer_full" && \
      "$TRACER_STRICT_REPRODUCIBILITY" == "1" ]]; then
    # Force a stable execution topology. PyTorch's deterministic mode in the
    # config rejects unsupported nondeterministic CUDA kernels.
    export PYTHONHASHSEED="$SEED"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_ALGO=Ring
    export NCCL_PROTO=Simple
    export TOKENIZERS_PARALLELISM=false
fi

RUN_DIR="$WORK_ROOT/runs/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

write_status() {
    local state="$1"
    local exit_code="${2:-}"
    python - "$RUN_DIR/status.json" "$state" "$RUN_ID" "$STAGE" "$SEED" "$exit_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, run_id, stage, seed, exit_code = sys.argv[1:]
status_path = Path(path)
payload = {}
if status_path.exists():
    payload = json.loads(status_path.read_text(encoding='utf-8'))
payload.update({
    'schema_version': 1,
    'state': state,
    'run_id': run_id,
    'stage': stage,
    'seed': int(seed),
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
    json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8'
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

# Resolve conda activation script.
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
[[ -f "$resolved_conda_sh" ]] || fail "conda init script not found; set CONDA_SH explicitly"
PS1="${PS1:-}"
# shellcheck source=/dev/null
source "$resolved_conda_sh"
conda activate "$CONDA_ENV"

if [[ "$STAGE" == "tracer_full" && "$DRY_RUN" != "1" && \
      "$TRACER_STRICT_REPRODUCIBILITY" == "1" && \
      "$TRACER_VERIFY_DATA_MANIFEST" == "1" ]]; then
    python "$SCRIPT_DIR/tracer_data_manifest.py" verify \
        --data-root "$HARMON_DATA_ROOT" \
        --manifest "$TRACER_DATA_MANIFEST" \
        > "$RUN_DIR/data_manifest_verification.json"
fi

# Snapshot config + entry script + git state.
snapshot_file() {
    local src="$1" dst="$2"
    if [[ -f "$src" ]]; then
        cp -- "$src" "$dst"
    else
        printf 'MISSING SOURCE: %s\n' "$src" > "$dst"
    fi
}
snapshot_file "$MODEL_CONFIG" "$RUN_DIR/model_config.py"
snapshot_file "$TRAIN_SCRIPT" "$RUN_DIR/train.py"
snapshot_file "$ENTRY_SCRIPT" "$RUN_DIR/entry_script.sh"
snapshot_file "${BASH_SOURCE[0]}" "$RUN_DIR/common_train_runner.sh"

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_head.txt" 2>/dev/null \
    || printf '%s\n' 'UNAVAILABLE' > "$RUN_DIR/git_head.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt" 2>/dev/null || true
git -C "$REPO_ROOT" diff --binary --no-ext-diff > "$RUN_DIR/git_diff.patch" 2>/dev/null || true

# Environment snapshot.
{
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'conda_default_env=%s\n' "${CONDA_DEFAULT_ENV:-}"
    printf 'python_executable='
    python -c 'import sys; print(sys.executable)'
    printf 'python='; python --version 2>&1
    python - <<'PY'
import torch
print('torch={}'.format(torch.__version__))
print('torch_cuda={}'.format(torch.version.cuda))
print('cuda_available={}'.format(torch.cuda.is_available()))
print('device_count={}'.format(torch.cuda.device_count()))
PY
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'nproc_per_node=%s\n' "$NPROC_PER_NODE"
    printf 'master_port=%s\n' "$MASTER_PORT"
    printf 'stage=%s\n' "$STAGE"
    printf 'seed=%s\n' "$SEED"
    printf 'dry_run=%s\n' "$DRY_RUN"
    printf 'exclusion_manifest=%s\n' "$EXCLUSION_MANIFEST"
    printf 'tracer_risk_head_path=%s\n' "${TRACER_RISK_HEAD_PATH:-unset}"
    printf 'tracer_risk_audit_path=%s\n' "${TRACER_RISK_AUDIT_PATH:-unset}"
    printf 'tracer_base_checkpoint=%s\n' "${TRACER_BASE_CHECKPOINT:-unset}"
    printf 'tracer_strict_reproducibility=%s\n' "$TRACER_STRICT_REPRODUCIBILITY"
    printf 'tracer_verify_data_manifest=%s\n' "$TRACER_VERIFY_DATA_MANIFEST"
    printf 'tracer_data_manifest=%s\n' "${TRACER_DATA_MANIFEST:-unset}"
    printf 'harmon_data_root=%s\n' "$HARMON_DATA_ROOT"
    printf 'pythonhashseed=%s\n' "${PYTHONHASHSEED:-unset}"
    printf 'cublas_workspace_config=%s\n' "${CUBLAS_WORKSPACE_CONFIG:-unset}"
    printf 'cuda_device_max_connections=%s\n' "${CUDA_DEVICE_MAX_CONNECTIONS:-unset}"
    printf 'nccl_algo=%s\n' "${NCCL_ALGO:-unset}"
    printf 'nccl_proto=%s\n' "${NCCL_PROTO:-unset}"
    python -m pip freeze 2>/dev/null || true
    nvidia-smi 2>/dev/null || true
} > "$RUN_DIR/environment.txt"

# Hash code + manifest artifacts.
hash_path() {
    local target="$1" label="$2" output="$3"
    if [[ -f "$target" ]]; then
        sha256sum "$target" | sed "s#  $target\$#  $label#" >> "$output"
    else
        printf 'MISSING  %s\n' "$label" >> "$output"
    fi
}
: > "$RUN_DIR/code.sha256"
hash_path "$MODEL_CONFIG" "model_config:$MODEL_CONFIG" "$RUN_DIR/code.sha256"
hash_path "$TRAIN_SCRIPT" "train_script:$TRAIN_SCRIPT" "$RUN_DIR/code.sha256"
hash_path "$ENTRY_SCRIPT" "entry_script:$ENTRY_SCRIPT" "$RUN_DIR/code.sha256"
hash_path "${BASH_SOURCE[0]}" "common_train_runner:${BASH_SOURCE[0]}" "$RUN_DIR/code.sha256"
hash_path "$HARMON_ROOT/src/models/harmon_dev.py" \
    "source:src/models/harmon_dev.py" "$RUN_DIR/code.sha256"
for tracer_source in "$HARMON_ROOT"/src/models/dllm/guard/*.py; do
    hash_path "$tracer_source" "source:${tracer_source#"$HARMON_ROOT"/}" \
        "$RUN_DIR/code.sha256"
done

: > "$RUN_DIR/data.sha256"
hash_path "$EXCLUSION_MANIFEST" "exclusion_manifest:$EXCLUSION_MANIFEST" "$RUN_DIR/data.sha256"
hash_path "${EXCLUSION_MANIFEST}.sha256" "exclusion_manifest_sidecar" "$RUN_DIR/data.sha256"
hash_path "$HARMON_DATA_ROOT/llava_v1_5_mix665k.json" \
    "training_source:llava_v1_5_mix665k.json" "$RUN_DIR/data.sha256"
hash_path "/nvmedata/xiexu/data/uni/harmon_1.5b.pth" \
    "base_checkpoint:harmon_1.5b.pth" "$RUN_DIR/data.sha256"
if [[ "$STAGE" == "tracer_full" ]]; then
    hash_path "$TRACER_DATA_MANIFEST" \
        "tracer_data_manifest" "$RUN_DIR/data.sha256"
    hash_path "${TRACER_RISK_HEAD_PATH:-}" \
        "tracer_risk_head" "$RUN_DIR/data.sha256"
    hash_path "${TRACER_RISK_AUDIT_PATH:-}" \
        "tracer_risk_audit" "$RUN_DIR/data.sha256"
    hash_path "${TRACER_BASE_CHECKPOINT:-}" \
        "tracer_base_checkpoint" "$RUN_DIR/data.sha256"
fi

# Validate the config loads (and the exclusion manifest binds) before launch.
export HARMON_GUARD_EXCLUSION_MANIFEST="$EXCLUSION_MANIFEST"
export TRACER_RISK_HEAD_PATH TRACER_RISK_AUDIT_PATH TRACER_BASE_CHECKPOINT
export HARMON_DATA_ROOT
export PYTHONPATH="$HARMON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - "$MODEL_CONFIG" "$RUN_ID" "$STAGE" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

from mmengine.config import Config

config_path, run_id, stage, cli_seed = sys.argv[1:]
cfg = Config.fromfile(config_path)
summary = {
    'schema_version': 1,
    'run_id': run_id,
    'stage': stage,
    'max_iters': cfg.train_cfg.get('max_iters'),
    'guard_exclusion_manifest': cfg.get('guard_exclusion_manifest'),
    'mask_token_strategy': cfg.model.get('mask_token_strategy'),
    'trainable_mask_delta': cfg.model.get('trainable_mask_delta'),
    'enforce_nonempty_dllm_targets': cfg.model.get('enforce_nonempty_dllm_targets'),
    'safe_dllm_loss': cfg.model.get('safe_dllm_loss'),
    'mask_invalid_dllm_logits': cfg.model.get('mask_invalid_dllm_logits'),
    'state_conditioner': cfg.model.get('state_conditioner'),
    'clean_anchor_weight': cfg.model.get('clean_anchor_weight'),
    'guard_enabled': cfg.model.get('guard_enabled', False),
    'guard_submodel': cfg.model.get('guard_submodel'),
    'guard_risk_head_path': cfg.model.get('guard_risk_head_path'),
    'tracer_router_enabled': cfg.model.get('tracer_router_enabled'),
    'tracer_policy_enabled': cfg.model.get('tracer_policy_enabled'),
    'tracer_method_contract': cfg.get('tracer_method_contract'),
    'lora_target_modules': list(cfg.model.lora.target_modules),
    'load_from': cfg.get('load_from'),
    'resume': cfg.get('resume'),
    # ``train.py --seed`` merges this value into cfg.randomness immediately
    # before building the runner.  Record the effective fresh-run seed rather
    # than the config file's default (42), so multi-seed manifests are honest.
    'randomness_seed': int(cli_seed),
    'randomness_deterministic': bool(cfg.randomness.get('deterministic')),
    'data_root': cfg.get('data_root'),
    'data_path': cfg.get('data_path'),
    'image_folder': cfg.get('image_folder'),
    'depth_folder': cfg.get('depth_folder'),
    'mask_folder': cfg.get('mask_folder'),
    'n_datasets': len(cfg.train_dataloader.dataset.datasets),
    'dataset_exclusion_manifests': [
        ds.get('exclusion_manifest') for ds in cfg.train_dataloader.dataset.datasets
    ],
}
Path(run_id if False else sys.argv[2] if False else 'config_summary.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
PY
mv config_summary.json "$RUN_DIR/config_summary.json" 2>/dev/null || cp config_summary.json "$RUN_DIR/config_summary.json" 2>/dev/null || true

# Write the exact command that will be executed.
work_dir="$RUN_DIR/work"
mkdir -p "$work_dir"
# Use ``python -m torch.distributed.run`` instead of the ``torchrun`` wrapper:
# the user-site ``~/.local/bin/torchrun`` shebang points at a different conda
# env, so it would import the wrong Python and miss ``xtuner``.  Invoking the
# module through the activated env's ``python`` keeps torch + xtuner aligned.
printf 'CUDA_VISIBLE_DEVICES=%s python -m torch.distributed.run --nproc_per_node=%s --master_port=%s %s %s --work-dir %s --launcher pytorch --seed %s\n' \
    "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$MASTER_PORT" \
    "$TRAIN_SCRIPT" "$MODEL_CONFIG" "$work_dir" "$SEED" > "$RUN_DIR/command.txt"

printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Stage: %s  |  GPUs: %s  |  nproc: %s  |  seed: %s  |  dry_run: %s\n' \
    "$STAGE" "$CUDA_VISIBLE_DEVICES" "$NPROC_PER_NODE" "$SEED" "$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
    write_status dry_run 0
    finalized=1
    cat "$RUN_DIR/config_summary.json"
    cat "$RUN_DIR/command.txt"
    exit 0
fi

# Real launch.
write_status running
set +e
(
    cd "$HARMON_ROOT"
    python -m torch.distributed.run \
        --nproc_per_node="$NPROC_PER_NODE" \
        --master_port="$MASTER_PORT" \
        "$TRAIN_SCRIPT" \
        "$MODEL_CONFIG" \
        --work-dir "$work_dir" \
        --launcher pytorch \
        --seed "$SEED"
) 2>&1 | tee "$RUN_DIR/train.log"
run_code=${PIPESTATUS[0]}
set -e

if [[ "$run_code" == "0" ]]; then
    write_status completed 0
    finalized=1
    exit 0
fi

write_status failed "$run_code"
finalized=1
exit "$run_code"
