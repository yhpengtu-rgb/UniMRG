"""GUARD-LoRA core modules.

Submodules:
* :mod:`guard_lora`   - rank-routed LoRA with log-scale gate and submodels.
* :mod:`risk_control` - trajectory-evidence risk head and fixed-NFE
  commit/revision policy (Path C main line).
"""

from .guard_lora import (
    GuardContext,
    GuardedLoRAManager,
    RankGate,
    SUBMODELS,
    scale_peft_lora_output,
)
from .risk_control import (
    CommitRevisionDecision,
    CommitRevisionPolicy,
    IsotonicCalibrator,
    RiskHead,
    TrajectoryCollector,
    TrajectoryFeatures,
    advance_corruption_state,
    advance_masked_tokens,
    apply_commit_revision,
    compute_trajectory_features,
    transfer_schedule_to_mask_probs,
)

__all__ = [
    'CommitRevisionDecision',
    'CommitRevisionPolicy',
    'GuardContext',
    'GuardedLoRAManager',
    'IsotonicCalibrator',
    'RankGate',
    'RiskHead',
    'SUBMODELS',
    'TrajectoryCollector',
    'TrajectoryFeatures',
    'advance_corruption_state',
    'advance_masked_tokens',
    'apply_commit_revision',
    'compute_trajectory_features',
    'scale_peft_lora_output',
    'transfer_schedule_to_mask_probs',
]
