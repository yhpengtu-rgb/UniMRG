"""Stable dLLM foundation package.

Re-exports the canonical mask-token registration, corruption state machinery,
state conditioner and matched causal view from the stable foundation submodules
so that ``harmon_dev`` and the test suite can import them as
``from src.models.dllm import ...``.

Submodules:
* :mod:`corruption`      - ``CorruptionState``, ``MaskedCorruptor``,
                           ``safe_token_cross_entropy``.
* :mod:`mask_token`      - canonical token registration, support sets,
                           trainable mask delta, valid logit mask, tokenizer
                           snapshot integrity and checkpoint key validation.
* :mod:`stable_baseline` - ``CorruptionStateConditioner``, ``MatchedCausalView``,
                           ``matched_clean_causal_kl``,
                           ``pack_corruption_state_features``.
"""

from .corruption import (
    CorruptionState,
    MaskedCorruptor,
    safe_token_cross_entropy,
)
from .mask_token import (
    TOKEN_METADATA_FILE,
    CheckpointKeyMismatchError,
    HarmonTokenSupport,
    TokenIds,
    TrainableMaskDelta,
    ValidLogitMask,
    ensure_embedding_capacity,
    harmon_token_support,
    load_harmon_tokenizer,
    load_harmon_tokenizer_snapshot,
    register_harmon_tokens,
    validate_checkpoint_keys,
    write_harmon_tokenizer_snapshot,
)
from .stable_baseline import (
    CorruptionStateConditioner,
    MatchedCausalView,
    matched_clean_causal_kl,
    pack_corruption_state_features,
)

__all__ = [
    'CheckpointKeyMismatchError',
    'CorruptionState',
    'CorruptionStateConditioner',
    'HarmonTokenSupport',
    'MaskedCorruptor',
    'MatchedCausalView',
    'TOKEN_METADATA_FILE',
    'TokenIds',
    'TrainableMaskDelta',
    'ValidLogitMask',
    'ensure_embedding_capacity',
    'harmon_token_support',
    'load_harmon_tokenizer',
    'load_harmon_tokenizer_snapshot',
    'matched_clean_causal_kl',
    'pack_corruption_state_features',
    'register_harmon_tokens',
    'safe_token_cross_entropy',
    'validate_checkpoint_keys',
    'write_harmon_tokenizer_snapshot',
]
