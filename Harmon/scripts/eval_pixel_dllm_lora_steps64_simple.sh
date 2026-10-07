#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

declare -A DATA_PATHS=(
    [geneval]="/nvmedata/xiexu/data/Benchmark/geneval/evaluation_metadata.jsonl"
    [dpgbench]="/nvmedata/xiexu/data/Benchmark/dpg_bench/prompts.json"
)
MODEL_CONFIG="configs/examples/UniMRG_pixel_dllm_lora.py"
CHECKPOINT="/nvmedata/xiexu/data/uni/UniMRG_pixel_dllm_lora_20k/iter_20000.pth"
OUTPUT_DIR="/nvmedata/xiexu/data/Benchmark/pixel_results/pixel_dllm_lora_iter_20000_t2i_steps64_cfg3_b4"

for benchmark in geneval dpgbench; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 /home/xiexu/anaconda3/envs/harmon/bin/python \
        -m torch.distributed.run --standalone --nproc_per_node=4 scripts/eval_pixel.py \
        --benchmark "$benchmark" --prompts "${DATA_PATHS[$benchmark]}" \
        --config "$MODEL_CONFIG" --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR/$benchmark" \
        --sampler pixel_dllm --num-iter 64 --cfg 3 --batch-size 4 "$@" || exit
done
