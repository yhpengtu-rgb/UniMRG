# DPGBench evaluation
export PYTHONPATH=.
torchrun \
  --nnodes 1 \
  --node_rank 0 \
  --nproc-per-node 4 \
  --master_addr 127.0.0.1 \
  --master-port 12346 \
  scripts/parallel_geneval.py \
  --checkpoint work_dirs/UniMRG_dllm/iter_4000.pth \
  --batch_size 4 \
  --outdir "results/UniMRG_dllm_dpg_iter_4000" \
  --mode dpgbench \
  --image_size 512 \
  --prompts_file ../Benchmark/dpg_bench/prompts.json