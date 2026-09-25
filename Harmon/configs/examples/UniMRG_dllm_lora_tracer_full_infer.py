"""TRACER-LoRA full factorial inference config.

Runtime cell selection is intentionally outside the checkpoint:
``TRACER_ROUTER_ENABLED`` and ``TRACER_POLICY_ENABLED`` only neutralise the
two consumers. Every cell loads the same LoRA, RankGate and RiskHead keys.
"""

import os
from copy import deepcopy

from mmengine.config import read_base


with read_base():
    from .UniMRG_dllm_lora_infer import model


tracer_risk_head_path = os.environ.get(
    'TRACER_RISK_HEAD_PATH',
    '/nvmedata/xiexu/data/uni/tracer_risk_v2/risk_head.pt',
)
if not tracer_risk_head_path:
    tracer_risk_head_path = None

model = deepcopy(model)
model.update(
    gradient_checkpointing=False,
    dllm=False,
    enforce_nonempty_dllm_targets=True,
    safe_dllm_loss=True,
    state_conditioner=False,
    state_conditioner_fields=('t', 'u', 'h'),
    state_conditioner_mode='additive_zero_init',
    clean_anchor_weight=0.0,
    guard_enabled=True,
    guard_submodel='R6',
    guard_risk_head_path=tracer_risk_head_path,
    guard_risk_head_hidden=None,
    guard_risk_head_proj_hidden=128,
    guard_risk_head_state_dim=3,
    tracer_router_enabled=True,
    tracer_policy_enabled=True,
    mask_token_strategy='canonical',
    trainable_mask_delta=True,
    mask_invalid_dllm_logits=True,
)

tracer_method_contract = dict(
    method='TRACER-LoRA',
    timing='strict_k_to_k_plus_1',
    evaluation_protocol='canonical_single_test',
    factorial_cells=('plain', 'router', 'policy', 'coupled'),
    risk_head_path=tracer_risk_head_path,
)

# VLMEvalKit must fail before generation if any trainable TRACER component
# is absent or only partially loaded from the evaluated checkpoint.
checkpoint_required_key_groups = (
    'proj_in.',
    'lora_A',
    'lora_B',
    'guard_rank_gate.',
    'mask_token_delta.',
)

load_from = None
resume = False
