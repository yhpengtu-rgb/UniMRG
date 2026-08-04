"""Inference-only config for the dLLM LoRA(attn+mlp) checkpoint.

与 UniMRG_dllm_lora_infer.py 的差别仅在 LoRA target_modules 额外包含
MLP 的 gate/up/down_proj，使 attn+mlp LoRA checkpoint 的 adapter 权重能被加载。
"""

import os

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm_lora import model


model['lora'] = dict(model['lora'])
model['lora']['target_modules'] = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]

model.update(
    pretrained_pth=os.environ.get(
        'HARMON_BASE_CHECKPOINT',
        '/nvmedata/xiexu/data/uni/harmon_1.5b.pth',
    ),
    gradient_checkpointing=False,
    dllm=False,
)
