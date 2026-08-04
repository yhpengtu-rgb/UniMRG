"""MMBench-focused dLLM training with Attention and MLP LoRA."""

from copy import deepcopy

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm_lora_opt import *


model = deepcopy(model)
model['lora'] = deepcopy(model['lora'])
model['lora']['target_modules'] = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]
