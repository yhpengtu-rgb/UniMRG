
from copy import deepcopy
from mmengine.config import read_base


with read_base():
    from .UniMRG_pixel_dllm import *
    from .UniMRG_dllm_lora_opt import model as understanding_model
    from .UniMRG_dllm_lora_opt import repeat as understanding_repeat


model = deepcopy(model)
model.update(
    # HarmonPixelDLLM inherits the text DLLM loss and overrides recon_loss.
    dllm=True,
    pixel_dllm=True,
    block_size=understanding_model['block_size'],
    min_mask_rate=understanding_model['min_mask_rate'],
    max_mask_rate=understanding_model['max_mask_rate'],
    freeze_llm=understanding_model['freeze_llm'],
    lora=deepcopy(understanding_model['lora']),
    # Keep the pixel config's trainable visual stack for joint optimization.
    freeze_mar_encoder=False,
    freeze_mar_decoder=False,
    freeze_proj_in=False,
    freeze_proj_out=False,
)

repeat = deepcopy(understanding_repeat)
accumulative_counts = sum(repeat)
train_dataloader = deepcopy(train_dataloader)
train_dataloader['sampler']['repeat'] = repeat
optim_wrapper = deepcopy(optim_wrapper)
optim_wrapper['accumulative_counts'] = accumulative_counts

max_iters = 30000
save_steps = 10000
save_total_limit = 3
train_cfg = deepcopy(train_cfg)
train_cfg['max_iters'] = max_iters
param_scheduler = deepcopy(param_scheduler)
param_scheduler[0].update(begin=0, end=warmup_ratio * max_iters)
param_scheduler[1].update(begin=warmup_ratio * max_iters, end=max_iters)
default_hooks = deepcopy(default_hooks)
default_hooks['checkpoint'].update(
    interval=save_steps, max_keep_ckpts=save_total_limit)

# Inherited pixel settings: b8, lr=1e-5, DiffLoss sampling=100.
# Warmup remains 1% (300 iterations); save at 10k/20k/30k and keep all three.
# Text block_size=4 is independent of the whole-image pixel mask schedule.
# Each task loss draws a random corruption and runs one Qwen forward;
# inference's outer 4/64 rounds are not training unroll counts.
load_from = None
resume = False
