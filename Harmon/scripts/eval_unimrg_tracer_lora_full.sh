#!/usr/bin/env bash
# Canonical five-dataset single-test entry for the four TRACER-LoRA cells.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

TRACER_CELL="${TRACER_CELL:-coupled}"
case "$TRACER_CELL" in
    plain)
        TRACER_ROUTER_ENABLED=false
        TRACER_POLICY_ENABLED=false
        ;;
    router)
        TRACER_ROUTER_ENABLED=true
        TRACER_POLICY_ENABLED=false
        ;;
    policy)
        TRACER_ROUTER_ENABLED=false
        TRACER_POLICY_ENABLED=true
        ;;
    coupled)
        TRACER_ROUTER_ENABLED=true
        TRACER_POLICY_ENABLED=true
        ;;
    *)
        printf 'ERROR: TRACER_CELL must be plain|router|policy|coupled, got: %s\n' \
            "$TRACER_CELL" >&2
        exit 1
        ;;
esac

CHECKPOINT="${CHECKPOINT:-/nvmedata/xiexu/uni/work_dirs/tracer_lora_training/runs/tracer-v2-exp7-deterministic-train-v2-s41/work/iter_20000.pth}"
MODEL_NAME="${MODEL_NAME:-HarmonTRACERLoRAFull_${TRACER_CELL}_BD}"
MODEL_CONFIG="${MODEL_CONFIG:-$SCRIPT_DIR/../configs/examples/UniMRG_dllm_lora_tracer_full_infer.py}"
WORK_ROOT="${WORK_ROOT:-/nvmedata/xiexu/uni/work_dirs/eval_unimrg_tracer_lora_full}"
RUN_ID="${RUN_ID:-tracer-full-${TRACER_CELL}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
USE_DLLM="${USE_DLLM:-true}"
TRACER_RISK_HEAD_PATH="${TRACER_RISK_HEAD_PATH:-/nvmedata/xiexu/data/uni/tracer_risk_v2/runs/tracer-v2-exp7-deterministic-risk-s42-v2/risk/risk_head.pt}"
TRACER_RISK_AUDIT_PATH="${TRACER_RISK_AUDIT_PATH:-/nvmedata/xiexu/data/uni/tracer_risk_v2/runs/tracer-v2-exp7-deterministic-risk-s42-v2/audit/risk_audit_report.json}"
EVAL_PREFLIGHT_SAMPLES="${EVAL_PREFLIGHT_SAMPLES:-1}"
EVAL_PREFLIGHT_REQUIRE_EOS="${EVAL_PREFLIGHT_REQUIRE_EOS:-true}"
EVAL_PREFLIGHT_REQUIRE_NONEMPTY="${EVAL_PREFLIGHT_REQUIRE_NONEMPTY:-true}"
ENTRY_SCRIPT="${BASH_SOURCE[0]}"
OFFICIAL_EVAL_FROZEN="${OFFICIAL_EVAL_FROZEN:-0}"
TRACER_STRICT_REPRODUCIBILITY="${TRACER_STRICT_REPRODUCIBILITY:-1}"
EVAL_SEED="${EVAL_SEED:-${PYTHONHASHSEED:-42}}"

case "$OFFICIAL_EVAL_FROZEN" in 0|1) ;;
    *) printf 'ERROR: OFFICIAL_EVAL_FROZEN must be 0 or 1\n' >&2; exit 1 ;;
esac
case "$TRACER_STRICT_REPRODUCIBILITY" in 0|1) ;;
    *) printf 'ERROR: TRACER_STRICT_REPRODUCIBILITY must be 0 or 1\n' >&2; exit 1 ;;
esac
[[ "$EVAL_SEED" =~ ^[0-9]+$ ]] || {
    printf 'ERROR: EVAL_SEED must be a non-negative integer\n' >&2
    exit 1
}
if [[ "${DRY_RUN:-0}" != "1" && "$OFFICIAL_EVAL_FROZEN" != "1" ]]; then
    printf 'ERROR: formal TRACER evaluation requires OFFICIAL_EVAL_FROZEN=1\n' >&2
    exit 1
fi
if [[ "$TRACER_STRICT_REPRODUCIBILITY" == "1" ]]; then
    export PYTHONHASHSEED="$EVAL_SEED"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

# An official run is invalid until the independent risk audit passes. DRY_RUN
# still produces a complete manifest so the command can be reviewed first.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    [[ -f "$TRACER_RISK_HEAD_PATH" ]] || {
        printf 'ERROR: missing TRACER RiskHead: %s\n' "$TRACER_RISK_HEAD_PATH" >&2
        exit 1
    }
    [[ -f "$TRACER_RISK_AUDIT_PATH" ]] || {
        printf 'ERROR: missing TRACER risk audit: %s\n' "$TRACER_RISK_AUDIT_PATH" >&2
        exit 1
    }
    python - "$TRACER_RISK_AUDIT_PATH" "$TRACER_RISK_HEAD_PATH" <<'PY'
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
fi

export CHECKPOINT MODEL_NAME MODEL_CONFIG WORK_ROOT RUN_ID USE_DLLM
export TRACER_CELL TRACER_ROUTER_ENABLED TRACER_POLICY_ENABLED
export TRACER_RISK_HEAD_PATH TRACER_RISK_AUDIT_PATH ENTRY_SCRIPT
export EVAL_PREFLIGHT_SAMPLES EVAL_PREFLIGHT_REQUIRE_EOS
export EVAL_PREFLIGHT_REQUIRE_NONEMPTY
export OFFICIAL_EVAL_FROZEN TRACER_STRICT_REPRODUCIBILITY EVAL_SEED

exec bash "$SCRIPT_DIR/eval_unimrg_common.sh"
