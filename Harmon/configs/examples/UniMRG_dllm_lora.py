"""dLLM training with Qwen Attention LoRA and a trainable proj_in."""

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm import *


model.update(
    # Freeze the pretrained backbones. PEFT re-enables only its adapters.
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
