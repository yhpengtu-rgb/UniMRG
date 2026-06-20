export LITELLM_API_KEY="sk-m7Ar2F1lQVBv9FHGomFugQ"

python run.py \
  --data VSR-zeroshot MMBench_DEV_EN MMVP HallusionBench RealWorldQA \
  --model Harmon \
  --work-dir ./outputs_start