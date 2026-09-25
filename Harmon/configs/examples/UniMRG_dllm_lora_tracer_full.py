"""TRACER-LoRA strict-lag training config (exp7).

The external method name is TRACER-LoRA. Historical ``guard_*`` model keys
remain unchanged so earlier LoRA/RankGate checkpoints can still be loaded.
This config starts from the matched stable 20k checkpoint and trains fresh
lagged gates against a separately audited RiskHead using adjacent schedule
states. The sampler policy has no trainable main-model parameters and is
evaluated through the pre-registered factorial cells at inference.
"""

from copy import deepcopy

from mmengine.config import read_base
from src.hooks import StableCheckpointHook, StableRunContractHook


with read_base():
    from .UniMRG_dllm_lora_opt import *  # noqa: F401,F403


# The exp7 checkpoint starts from the stable 20k weights, but this config
# carries its own training contract instead of importing the old S1 launcher.
EXCLUSION_ENV = 'HARMON_GUARD_EXCLUSION_MANIFEST'
_manifest_env = __import__('os').environ.get(EXCLUSION_ENV)
if not _manifest_env:
    raise RuntimeError(f'{EXCLUSION_ENV} must be set')
if not __import__('os').path.isfile(_manifest_env):
    raise RuntimeError(f'manifest file not found: {_manifest_env}')
_sidecar = _manifest_env + '.sha256'
if not __import__('os').path.isfile(_sidecar):
    raise RuntimeError(f'SHA256 sidecar missing: {_sidecar}')

train_dataloader = deepcopy(train_dataloader)
_train_dataset = deepcopy(train_dataloader['dataset'])
_datasets = list(_train_dataset['datasets'])
for _ds in _datasets:
    _ds['exclusion_manifest'] = _manifest_env
_train_dataset['datasets'] = _datasets
train_dataloader['dataset'] = _train_dataset
guard_exclusion_manifest = _manifest_env

model = deepcopy(model)
model.update(
    mask_token_strategy='canonical',
    trainable_mask_delta=True,
    enforce_nonempty_dllm_targets=True,
    safe_dllm_loss=True,
    mask_invalid_dllm_logits=True,
    state_conditioner=False,
    state_conditioner_fields=('t', 'u', 'h'),
    state_conditioner_mode='additive_zero_init',
    clean_anchor_weight=0.0,
    gradient_checkpointing=False,
)
model['mar']['grad_checkpointing'] = False
strategy = dict(
    type='DDPStrategy',
    model_wrapper=dict(
        type='MMDistributedDataParallel',
        broadcast_buffers=False,
        find_unused_parameters=True,
    ),
)

default_hooks = deepcopy(default_hooks)
default_hooks['checkpoint'] = dict(
    type=StableCheckpointHook,
    by_epoch=False,
    interval=5000,
    max_keep_ckpts=4,
    save_last=True,
    save_best=None,
)
custom_hooks = [dict(type=StableRunContractHook)]
train_cfg = deepcopy(train_cfg)
train_cfg['max_iters'] = 20000


tracer_risk_head_path = __import__('os').environ.get(
    'TRACER_RISK_HEAD_PATH',
    '/nvmedata/xiexu/data/uni/tracer_risk_v2/risk_head.pt',
)
if not tracer_risk_head_path:
    tracer_risk_head_path = None
tracer_base_checkpoint = __import__('os').environ.get(
    'TRACER_BASE_CHECKPOINT',
    '/nvmedata/xiexu/uni/work_dirs/guard_training/runs/'
    'stable-seed42-20260809-v1/work/iter_20000.pth',
)

model = deepcopy(model)
model.update(
    # Compatibility namespace; public experiment name is TRACER-LoRA.
    guard_enabled=True,
    guard_submodel='R6',
    guard_risk_head_path=tracer_risk_head_path,
    guard_risk_head_hidden=None,
    guard_risk_head_proj_hidden=128,
    guard_risk_head_state_dim=3,
    guard_train_rounds=2,
    guard_gate_regularizer_weight=0.001,
    tracer_router_enabled=True,
    tracer_policy_enabled=True,
    # Fresh RankGate parameters are installed after loading this matched
    # static checkpoint. Do not hot-start the constant-state P2 gates.
    pretrained_pth=tracer_base_checkpoint,
    pretrained_required_key_groups=(
        'proj_in.',
        'lora_A',
        'lora_B',
        'mask_token_delta.',
    ),
    freeze_proj_in=True,
)

# These settings were inherited through the historical GUARD trajectory
# config. Keep them here so exp7 does not depend on that separate experiment.
optim_wrapper = deepcopy(optim_wrapper)
optim_wrapper['optimizer'] = deepcopy(optim_wrapper['optimizer'])
optim_wrapper['optimizer']['guard_lr_multiplier'] = 10.0
optim_wrapper['optimizer']['guard_weight_decay'] = 0.0

max_iters = 20000
save_steps = 2000
save_total_limit = 4
train_cfg = deepcopy(train_cfg)
train_cfg['max_iters'] = max_iters
param_scheduler = deepcopy(param_scheduler)
param_scheduler[0]['end'] = warmup_ratio * max_iters
param_scheduler[1]['begin'] = warmup_ratio * max_iters
param_scheduler[1]['end'] = max_iters
param_scheduler[1]['eta_min'] = 0.0
default_hooks = deepcopy(default_hooks)
default_hooks['checkpoint']['interval'] = save_steps
default_hooks['checkpoint']['max_keep_ckpts'] = save_total_limit

# Fail closed on CUDA operators for which PyTorch cannot provide a
# deterministic implementation.  The launcher additionally fixes Python,
# CuBLAS and NCCL execution settings.  Exact checkpoint equality is only
# promised for the same hardware/software topology; cross-platform runs are
# expected to reproduce the protocol and metrics, not necessarily every bit.
randomness = deepcopy(randomness)
randomness.update(seed=42, deterministic=True)

# Machine-readable method contract saved by the launcher.
tracer_method_contract = dict(
    method='TRACER-LoRA',
    expansion='Trajectory Risk-Aware Coupled Evidence Routing LoRA',
    timing='strict_k_to_k_plus_1',
    state='schedule_aligned_t_u_h',
    commit_budget='cumulative_fixed_nfe',
    max_revision_fraction=0.10,
    rank_gate_initial_beta=0.10,
    factorial_cells=('plain', 'router', 'policy', 'coupled'),
    risk_head_path=tracer_risk_head_path,
    base_checkpoint=tracer_base_checkpoint,
    deterministic_training=True,
    deterministic_scope='same_hardware_software_topology',
)

load_from = None
resume = False
