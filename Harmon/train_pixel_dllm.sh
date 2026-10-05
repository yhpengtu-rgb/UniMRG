#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TRAIN_PYTHON="${TRAIN_PYTHON:-/home/xiexu/anaconda3/envs/harmon/bin/python}"
WORK_DIR="${WORK_DIR:-/nvmedata/xiexu/data/uni/UniMRG_pixel_dllm_20k}"
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
mkdir -p "$WORK_DIR"
# The lock survives through torchrun and prevents two writers to this experiment.
exec 9>"$WORK_DIR/train.lock"
flock -n 9 || { echo "Training already owns $WORK_DIR" >&2; exit 1; }
if [[ -f "$WORK_DIR/last_checkpoint" && " $* " != *" --resume "* ]]; then
    echo "Checkpoint exists. Pass --resume CHECKPOINT or use another WORK_DIR." >&2
    exit 1
fi
printf '%s\n' "$$" > "$WORK_DIR/train.pid"
printf 'running\n' > "$WORK_DIR/status.txt"
trap 'code=$?; printf "exit_code=%s\n" "$code" > "$WORK_DIR/status.txt"' EXIT
"$TRAIN_PYTHON" -u -m torch.distributed.run --standalone \
    --nproc_per_node="${#DEVICES[@]}" \
    scripts/train.py configs/examples/UniMRG_pixel_dllm.py \
    --launcher pytorch --deepspeed deepspeed_zero2 \
    --work-dir "$WORK_DIR" "$@"
