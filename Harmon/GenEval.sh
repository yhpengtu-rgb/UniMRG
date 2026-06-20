# GenEval evaluation
export PYTHONPATH=.
torchrun \
  --nnodes 1 \
  --node_rank 0 \
  --nproc-per-node 4 \
  --master_addr 127.0.0.1 \
  --master-port 12345 \
  scripts/parallel_geneval.py \
  --checkpoint work_dirs/UniMRG_dllm/iter_4000.pth \
  --batch_size 12 \
  --outdir "results/UniMRG_dllm_gen_bs16_iter4000" \
  --mode geneval \
  --image_size 512 \
  --validation_prompts_file ../Benchmark/geneval/evaluation_metadata.jsonl