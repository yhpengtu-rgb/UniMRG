"""Joint LoRA inference with the same model structure and base as training."""

from copy import deepcopy

from mmengine.config import read_base

with read_base():
    from .UniMRG_joint_dllm_lora import model

model = deepcopy(model)
model.update(gradient_checkpointing=False, dllm=False)
model['llm']['attn_implementation'] = 'sdpa'
