"""MMBench-focused dLLM LoRA training from the Harmon 1.5B base."""

from copy import deepcopy

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm_lora import *


# FixedBatchMultiSourceSampler source order:
# image2text, depth, recon, mask.
repeat = [4, 1, 1, 1]
accumulative_counts = sum(repeat)
max_iters = 20000
lr = 1e-5
warmup_ratio = 0.01
save_steps = 5000
save_total_limit = 4

train_dataloader = deepcopy(train_dataloader)
train_dataloader['sampler'] = deepcopy(train_dataloader['sampler'])
train_dataloader['sampler']['repeat'] = repeat

optim_wrapper = deepcopy(optim_wrapper)
optim_wrapper['optimizer'] = deepcopy(optim_wrapper['optimizer'])
optim_wrapper['optimizer']['lr'] = lr
optim_wrapper['accumulative_counts'] = accumulative_counts

param_scheduler = deepcopy(param_scheduler)
param_scheduler[0]['begin'] = 0
param_scheduler[0]['end'] = warmup_ratio * max_iters
param_scheduler[1]['begin'] = warmup_ratio * max_iters
param_scheduler[1]['end'] = max_iters
param_scheduler[1]['eta_min'] = 0.0

train_cfg = deepcopy(train_cfg)
train_cfg['max_iters'] = max_iters

default_hooks = deepcopy(default_hooks)
default_hooks['checkpoint'] = deepcopy(default_hooks['checkpoint'])
default_hooks['checkpoint']['interval'] = save_steps
default_hooks['checkpoint']['max_keep_ckpts'] = save_total_limit

# Start a clean LoRA experiment from model.pretrained_pth.
load_from = None
resume = False
