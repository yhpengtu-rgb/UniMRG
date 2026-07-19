"""
统一推理配置（MMBench / VLMEvalKit 评测专用）
- 不加载 pretrained_pth，权重由 VLMEvalKit 在外部显式加载
- dllm=False：推理走标准 AR 生成，不走 block-diffusion 训练路径
- gradient_checkpointing=False：推理模式不需要节省显存

适用模型:
  1. HarmonBase  —— harmon_1.5b.pth（官方预训练基座）
  2. HarmonDLLM  —— iter_10000.pth（UniMRG dLLM 微调 10k 步）

两者架构完全相同，只是加载不同的权重。
"""
from mmengine.config import read_base
from src.models.harmon_dev import HarmonDev

with read_base():
    from ..models.qwen2_5_1_5b_kl16_mar_h import model

model.update(
    type=HarmonDev,
    # 不在 __init__ 里加载权重；VLMEvalKit 会显式 load_state_dict
    pretrained_pth=None,
    # 推理不需要 dllm 逻辑（generate_inner 直接调 llm.generate）
    dllm=False,
    # 推理不需要 gradient checkpointing
    gradient_checkpointing=False,
    freeze_llm=False,
    freeze_mar_encoder=False,
    freeze_proj_in=False,
    freeze_mar_decoder=False,
    freeze_proj_out=False,
    loss_weights={},
)
# 推理用 sdpa，效率更高
model['llm']['attn_implementation'] = 'sdpa'
