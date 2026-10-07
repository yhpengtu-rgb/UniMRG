"""Pixel DLLM training with Qwen Attention LoRA and trainable pixel queries."""

from mmengine.config import read_base


with read_base():
    from .UniMRG_pixel_dllm import *


model.update(
    # PEFT re-enables its adapters; proj_in and the new pixel queries stay trainable.
    freeze_llm=True,
    freeze_mar_encoder=True,
    freeze_proj_in=False,
    freeze_mar_decoder=True,
    freeze_proj_out=True,
    lora=dict(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
        ],
    ),
)
