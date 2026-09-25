import json
from collections.abc import Mapping
from pathlib import Path

import torch
from mmengine.dist import barrier, is_main_process
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from src.models.dllm.mask_token import (
    TOKEN_METADATA_FILE,
    harmon_token_support,
    load_harmon_tokenizer_snapshot,
    write_harmon_tokenizer_snapshot,
)


TOKEN_SUPPORT_NAMES = (
    'source_token_ids',
    'input_token_ids',
    'output_token_ids',
)
TOKEN_METADATA_REQUIRED_FIELDS = {
    'schema_version',
    'image_token_id',
    'mask_token_id',
    'tokenizer_length',
    'tree_sha256',
    *(
        field
        for name in TOKEN_SUPPORT_NAMES
        for field in (f'{name}_sha256', f'{name}_count')
    ),
}
CONTRACT_STATE_KEYS = (
    'harmon_image_token_id',
    'harmon_canonical_mask_token_id',
    'harmon_legacy_mask_token_id',
    'harmon_mask_token_id',
    'mask_token_delta.mask_token_id',
    'dllm_valid_logit_mask.valid_token_mask',
)


def unwrap_model(model):
    while is_model_wrapper(model):
        model = model.module
    return model


def validate_model_token_identity(model, metadata):
    expected = {
        'image_token_id': 'harmon_image_token_id',
        'mask_token_id': 'harmon_canonical_mask_token_id',
    }
    for metadata_key, attribute in expected.items():
        value = getattr(model, attribute, None)
        if value is None:
            raise RuntimeError(f'model is missing {attribute}')
        if hasattr(value, 'numel'):
            if value.numel() != 1:
                raise RuntimeError(f'{attribute} must be scalar')
            value = value.item()
        if int(value) != int(metadata.get(metadata_key, -1)):
            raise RuntimeError(
                f'token identity mismatch for {metadata_key}: '
                f"model={int(value)}, metadata={metadata.get(metadata_key)}"
            )

    canonical_id = int(metadata['mask_token_id'])
    active_id = getattr(model, 'harmon_mask_token_id', None)
    if active_id is None or int(active_id.item()) != canonical_id:
        raise RuntimeError(
            'active mask token identity mismatch: '
            f"expected {canonical_id}, got {active_id}"
        )
    delta = getattr(model, 'mask_token_delta', None)
    if delta is None or int(delta.mask_token_id.item()) != canonical_id:
        raise RuntimeError('mask delta token identity mismatch')
    if not torch.isfinite(delta.delta).all():
        raise RuntimeError('mask delta is non-finite')

    support = harmon_token_support(model.tokenizer)
    for name, digest in support.sha256.items():
        if metadata.get(f'{name}_sha256') != digest:
            raise RuntimeError(f'{name} hash mismatch')
        if metadata.get(f'{name}_count') != len(getattr(support, name)):
            raise RuntimeError(f'{name} count mismatch')

    valid_logit_mask = getattr(model, 'dllm_valid_logit_mask', None)
    if valid_logit_mask is None:
        raise RuntimeError('canonical run requires dllm_valid_logit_mask')
    actual_valid = valid_logit_mask.valid_token_mask
    expected_valid = torch.zeros_like(actual_valid)
    expected_valid[list(support.output_token_ids)] = True
    if not torch.equal(actual_valid, expected_valid):
        raise RuntimeError('valid output token mask identity mismatch')


def checkpoint_state_dict(checkpoint):
    """Extract exactly one canonical model-state envelope.

    Regular B20 checkpoint files may carry their model mapping under one of
    the three MMEngine/DeepSpeed-compatible envelope keys.  Other top-level
    mappings are intentionally rejected rather than treated as state dicts.
    """
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError('stable checkpoint payload must be a mapping')
    for key in ('state_dict', 'module', 'model'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state_dict = dict(candidate)
            normalized = {}
            for name, value in state_dict.items():
                if not isinstance(name, str):
                    raise RuntimeError('stable checkpoint state_dict keys must be strings')
                normalized_name = name[len('module.'):] if name.startswith('module.') else name
                if normalized_name in normalized:
                    raise RuntimeError('stable checkpoint state_dict has duplicate normalized key')
                normalized[normalized_name] = value
            return normalized
    raise RuntimeError('stable checkpoint is missing model state_dict')


def _checkpoint_state_mapping(checkpoint):
    for key in ('state_dict', 'module', 'model'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    raise RuntimeError('stable checkpoint is missing model state_dict')


def validate_stable_checkpoint_state(model, checkpoint, metadata):
    state_dict = checkpoint_state_dict(checkpoint)
    full_state_keys = set(model.state_dict())
    required_keys = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    required_keys.update(CONTRACT_STATE_KEYS)
    actual_keys = set(state_dict)
    missing = sorted(required_keys - actual_keys)
    unexpected = actual_keys - full_state_keys
    ignored_frozen_keys = set()
    if unexpected and all(name.startswith('vae.') for name in unexpected):
        vae = getattr(model, 'vae', None)
        if vae is None:
            raise RuntimeError(
                'stable checkpoint contains frozen VAE state but the model '
                'has no VAE module'
            )
        expected_vae = {
            f'vae.{name}': value
            for name, value in vae.state_dict().items()
        }
        if unexpected != set(expected_vae):
            missing_vae = sorted(set(expected_vae) - unexpected)
            surplus_vae = sorted(unexpected - set(expected_vae))
            raise RuntimeError(
                'stable checkpoint frozen VAE schema mismatch: '
                f'missing={missing_vae}, unexpected={surplus_vae}'
            )
        for name in sorted(unexpected):
            checkpoint_value = state_dict[name]
            model_value = expected_vae[name]
            if (
                not hasattr(checkpoint_value, 'detach')
                or checkpoint_value.dtype != model_value.dtype
                or checkpoint_value.shape != model_value.shape
                or not torch.equal(
                    checkpoint_value.detach().cpu(),
                    model_value.detach().cpu(),
                )
            ):
                raise RuntimeError(
                    f'stable checkpoint frozen VAE value mismatch: {name}'
                )
        ignored_frozen_keys = set(unexpected)
        unexpected = set()
    unexpected = sorted(unexpected)
    if missing or unexpected:
        raise RuntimeError(
            f'stable checkpoint keys mismatch: missing={missing}, '
            f'unexpected={unexpected}'
        )

    canonical_id = int(metadata['mask_token_id'])
    image_id = int(metadata['image_token_id'])
    scalar_contract = {
        'harmon_image_token_id': image_id,
        'harmon_canonical_mask_token_id': canonical_id,
        'harmon_mask_token_id': canonical_id,
        'mask_token_delta.mask_token_id': canonical_id,
        'harmon_legacy_mask_token_id': int(
            model.harmon_legacy_mask_token_id.item()
        ),
    }
    for key, expected in scalar_contract.items():
        value = state_dict[key]
        if value.numel() != 1 or int(value.item()) != expected:
            label = (
                'active mask token'
                if key == 'harmon_mask_token_id'
                else key
            )
            raise RuntimeError(
                f'{label} checkpoint identity mismatch: '
                f'expected {expected}, got {value}'
            )

    checkpoint_valid_mask = state_dict[
        'dllm_valid_logit_mask.valid_token_mask'
    ].detach().cpu()
    model_valid_mask = (
        model.dllm_valid_logit_mask.valid_token_mask.detach().cpu()
    )
    if (
        checkpoint_valid_mask.dtype != model_valid_mask.dtype
        or checkpoint_valid_mask.shape != model_valid_mask.shape
        or not torch.equal(checkpoint_valid_mask, model_valid_mask)
    ):
        raise RuntimeError('checkpoint valid output token mask mismatch')
    if not torch.isfinite(state_dict['mask_token_delta.delta']).all():
        raise RuntimeError('checkpoint mask delta is non-finite')
    return frozenset(ignored_frozen_keys)


def load_checkpoint_for_contract_audit(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        checkpoint_path = (
            checkpoint_path / 'mp_rank_00_model_states.pt'
        )
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f'stable checkpoint audit file not found: {checkpoint_path}'
        )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError('stable checkpoint payload must be a mapping')
    return checkpoint


class StableRunContractHook(Hook):
    """Persist and validate tokenizer identity for canonical S1 runs."""

    priority = 'VERY_HIGH'

    def __init__(self, snapshot_subdir='tokenizer'):
        self.snapshot_subdir = snapshot_subdir
        self.metadata = None

    @staticmethod
    def _is_stable(model):
        return getattr(model, 'mask_token_strategy', 'legacy') == 'canonical'

    def before_train(self, runner):
        model = unwrap_model(runner.model)
        if not self._is_stable(model):
            return
        if getattr(model, 'tokenizer', None) is None:
            raise RuntimeError('canonical run requires a model tokenizer')

        load_from = getattr(runner, '_load_from', None)
        if getattr(runner, '_resume', False) and load_from is None:
            raise RuntimeError(
                'stable resume requires an explicit checkpoint path'
            )
        if load_from is not None:
            checkpoint = load_checkpoint_for_contract_audit(load_from)
            self.after_load_checkpoint(runner, checkpoint)

        snapshot_dir = Path(runner.work_dir) / self.snapshot_subdir
        metadata_file = snapshot_dir / TOKEN_METADATA_FILE
        if is_main_process() and not metadata_file.is_file():
            write_harmon_tokenizer_snapshot(model.tokenizer, snapshot_dir)
        barrier()

        tokenizer, token_ids = load_harmon_tokenizer_snapshot(snapshot_dir)
        del token_ids
        metadata = json.loads(
            metadata_file.read_text(encoding='utf-8')
        )
        if self.metadata is not None and self.metadata != metadata:
            raise RuntimeError(
                'checkpoint tokenizer metadata does not match run snapshot'
            )
        validate_model_token_identity(model, metadata)
        model.tokenizer = tokenizer
        self.metadata = metadata
        runner.stable_tokenizer_metadata = dict(metadata)

    def before_save_checkpoint(self, runner, checkpoint):
        model = unwrap_model(runner.model)
        if not self._is_stable(model):
            return
        if self.metadata is None:
            raise RuntimeError(
                'stable tokenizer metadata was not prepared before checkpointing'
            )
        checkpoint.setdefault('meta', {})['harmon_tokenizer'] = dict(
            self.metadata
        )
        state_mapping = _checkpoint_state_mapping(checkpoint)
        # HarmonDev's frozen VAE is restored from its separately hashed KL-VAE
        # artifact and intentionally excluded from its canonical state_dict.
        # Some MMEngine save paths collect child-module state directly and can
        # reintroduce those tensors, making checkpoints much larger and
        # incompatible with the canonical inference contract.
        for key in list(state_mapping):
            normalized = key[len('module.'):] if key.startswith('module.') else key
            if normalized.startswith('vae.'):
                del state_mapping[key]
        model_state = model.state_dict()
        prefix = (
            'module.'
            if state_mapping
            and all(key.startswith('module.') for key in state_mapping)
            else ''
        )
        for key in CONTRACT_STATE_KEYS:
            target_key = f'{prefix}{key}'
            if target_key not in state_mapping:
                state_mapping[target_key] = model_state[key].detach().clone()

    def after_load_checkpoint(self, runner, checkpoint):
        model = unwrap_model(runner.model)
        if not self._is_stable(model):
            return
        metadata = checkpoint.get('meta', {}).get('harmon_tokenizer')
        if metadata is None:
            raise RuntimeError(
                'stable checkpoint is missing harmon_tokenizer metadata'
            )
        missing = sorted(TOKEN_METADATA_REQUIRED_FIELDS.difference(metadata))
        if missing:
            raise RuntimeError(
                f'stable checkpoint tokenizer metadata missing: {missing}'
            )
        if metadata['schema_version'] != 2:
            raise RuntimeError(
                'stable checkpoint requires tokenizer metadata schema 2'
            )
        validate_model_token_identity(model, metadata)
        validate_stable_checkpoint_state(model, checkpoint, metadata)
        self.metadata = dict(metadata)
