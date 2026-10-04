CUDA_VISIBLE_DEVICES=0,1 /home/xiexu/anaconda3/envs/harmon/bin/torchrun \
  --standalone --nproc_per_node=2 Harmon/scripts/eval_pixel.py --benchmark geneval

# Full benchmark generation: 1065 DPGBench prompts × 4 images.
CUDA_VISIBLE_DEVICES=0,1 /home/xiexu/anaconda3/envs/harmon/bin/torchrun \
  --standalone --nproc_per_node=2 Harmon/scripts/eval_pixel.py --benchmark dpgbench

# Resume with new batch size: 1065 DPGBench prompts × 4 images.
# CUDA_VISIBLE_DEVICES=2 /home/xiexu/anaconda3/envs/harmon/bin/torchrun \
#   --standalone --nproc_per_node=1 Harmon/scripts/eval_pixel.py \
#   --benchmark dpgbench --batch-size 4 \
#   --output-dir /nvmedata/xiexu/data/Benchmark/pixel_results/dpg_batch4