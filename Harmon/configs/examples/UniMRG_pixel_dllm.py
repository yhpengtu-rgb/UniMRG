"""Stage 1: pixel DLLM only, with the original four-task training schedule."""
from mmengine.config import read_base
from src.models.harmon_pixel_dllm import HarmonPixelDLLM

with read_base():
    from .UniMRG import *

model_root = '/nvmedata/xiexu/data/uni'
data_root = '/nvmedata/xiexu/data/LLaVA-Instruct-150K-UniMRG'
llm_path = model_root + '/Qwen2.5-1.5B-Instruct'
model.update(type=HarmonPixelDLLM, dllm=False, pixel_dllm=True,
             pretrained_pth=model_root + '/harmon_1.5b.pth')
model['vae']['ckpt_path'] = model_root + '/kl16.ckpt'
model['llm'].update(pretrained_model_name_or_path=llm_path, attn_implementation='sdpa')
model['tokenizer']['pretrained_model_name_or_path'] = llm_path

for dataset_config in train_dataloader['dataset']['datasets']:
    dataset_config['data_path'] = data_root + '/llava_v1_5_mix665k.json'
    dataset_config['image_folder'] = data_root + '/tuning_data'
    dataset_config['tokenizer']['pretrained_model_name_or_path'] = llm_path
    if 'depth_folder' in dataset_config:
        dataset_config['depth_folder'] = data_root + '/tuning_data_depth'
    if 'mask_folder' in dataset_config:
        dataset_config['mask_folder'] = data_root + '/tuning_data_mask'
train_dataloader['dataset']['datasets'][0]['cache_folder'] = (
    '/nvmedata/xiexu/data/LLaVA-Instruct-150K/cache_harmon_understanding')

batch_size = 8
train_dataloader['batch_size'] = batch_size
train_dataloader['sampler']['batch_size'] = batch_size
max_iters = 20000
train_cfg['max_iters'] = max_iters
param_scheduler[0]['end'] = warmup_ratio * max_iters
param_scheduler[1].update(begin=warmup_ratio * max_iters, end=max_iters)
# A resumable checkpoint includes optimizer shards; keep disk usage bounded.
default_hooks['checkpoint'].update(interval=1000, max_keep_ckpts=1)
