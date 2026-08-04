"""Inference-only model config for the Harmon dLLM LoRA checkpoint."""

import os

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm_lora import model


model.update(
    pretrained_pth=os.environ.get(
        'HARMON_BASE_CHECKPOINT',
        '/nvmedata/xiexu/data/uni/harmon_1.5b.pth',
    ),
    gradient_checkpointing=False,
    dllm=False,
)
