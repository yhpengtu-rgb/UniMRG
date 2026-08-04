#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export MODEL_NAME="${MODEL_NAME:-HarmonLoRAOpt20k_BD}"
export MODEL_CONFIG="${MODEL_CONFIG:-$REPO_ROOT/Harmon/configs/examples/UniMRG_dllm_lora_infer.py}"
export CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/Harmon/work_dirs/UniMRG_dllm_lora_opt/iter_20000.pth}"
export USE_DLLM="${USE_DLLM:-true}"
export WORK_ROOT="${WORK_ROOT:-/nvmedata/xiexu/uni/work_dirs/eval_unimrg_lora_opt20k}"
export HARMON_BASE_CHECKPOINT="${HARMON_BASE_CHECKPOINT:-/nvmedata/xiexu/data/uni/harmon_1.5b.pth}"
export ENTRY_SCRIPT="${BASH_SOURCE[0]}"

COMMON_RUNNER="${COMMON_RUNNER:-$SCRIPT_DIR/eval_unimrg_common.sh}"
exec "$COMMON_RUNNER"
