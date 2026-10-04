#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${SCORE_GPU:-0}"
PYTHON=/home/xiexu/anaconda3/envs/harmon/bin/python
RESULTS=/nvmedata/xiexu/data/Benchmark/pixel_results

"$PYTHON" Harmon/scripts/score_pixel.py --benchmark geneval \
  --image-dir "$RESULTS/harmon_1.5b_geneval" --experiment base-geneval --allow-incomplete "$@"
"$PYTHON" Harmon/scripts/score_pixel.py --benchmark dpgbench \
  --image-dir "$RESULTS/harmon_1.5b_dpgbench" --experiment base-dpg-b1 --allow-incomplete "$@"
"$PYTHON" Harmon/scripts/score_pixel.py --benchmark dpgbench \
  --image-dir "$RESULTS/dpg_batch4" --experiment base-dpg-b4 "$@"
