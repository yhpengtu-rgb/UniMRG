"""Canonical Harmon mask-token registration, support sets and checkpoints.

This module is part of the stable dLLM foundation (Task 1). It owns the
canonical registration of the two Harmon tokens (``<image>`` and
``<|dllm_mask|>``), the explicit input / source / output support sets, the
trainable mask-embedding delta, the output logit mask, the tokenizer snapshot
integrity contract and the checkpoint key validator.

Contracts are pinned by:

* ``Harmon/tests/test_mask_token.py``
* ``Harmon/tests/test_tokenizer_checkpoint.py``
* ``Harmon/tests/test_stable_run_contract.py``
* ``Harmon/src/hooks/stable_run_contract.py`` (consumes ``TOKEN_METADATA_FILE``
  and the metadata schema produced here)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer


HARMON_IMAGE_TOKEN = '<image>'
HARMON_MASK_TOKEN = '<|dllm_mask|>'

TOKEN_METADATA_FILE = 'harmon_token_metadata.json'
TOKEN_METADATA_SCHEMA_VERSION = 2


class CheckpointKeyMismatchError(Exception):
    """Raised when a checkpoint's missing / unexpected keys violate policy."""


@dataclass(frozen=True)
class TokenIds:
    """Canonical ids of the two registered Harmon tokens."""

    image_token_id: int
    mask_token_id: int
    tokenizer_length: int


@dataclass(frozen=True)
class HarmonTokenSupport:
    """Explicit, hashable input / source / output token-id support sets.

    ``input_token_ids``  - every id a model may receive as input (the full
                           registered range, including image / mask).
    ``source_token_ids`` - ids eligible to be sampled as a corruption source;
                           excludes specials (PAD/BOS/EOS/...), image and mask.
    ``output_token_ids`` - ids a model may emit as a prediction target;
                           excludes image, mask and unregistered padded rows.
    ``sha256``           - hex digests of each support set, keyed by
                           ``input`` / ``source`` / ``output``.
    """

    input_token_ids: Tuple[int, ...]
    source_token_ids: Tuple[int, ...]
    output_token_ids: Tuple[int, ...]
    sha256: Dict[str, str]


def ensure_embedding_capacity(model: nn.Module, *, tokenizer_length: int) -> int:
    """Expand the model's token embedding to ``tokenizer_length`` if needed.

    The embedding is never shrunk: a tokenizer that is shorter than the
    embedding leaves the embedding untouched. This guards against accidentally
    truncating a 151936-row Qwen embedding down to the 151667-row Harmon
    tokenizer length.
    """

    embedding = model.get_input_embeddings()
    rows = int(embedding.num_embeddings)
    target = int(tokenizer_length)
    if target > rows:
        model.resize_token_embeddings(target)
        rows = int(model.get_input_embeddings().num_embeddings)
    return rows


def register_harmon_tokens(
    tokenizer,
    model: Optional[nn.Module] = None,
) -> TokenIds:
    """Register ``<image>`` and ``<|dllm_mask|>`` idempotently.

    Tokens are appended in fixed order without replacing the existing
    special-token list. When ``model`` is provided the model embedding is grown
    (never shrunk) to match the new tokenizer length.

    Returns a :class:`TokenIds` with the canonical ids and tokenizer length.
    """

    for token in (HARMON_IMAGE_TOKEN, HARMON_MASK_TOKEN):
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token])

    image_id = int(tokenizer.convert_tokens_to_ids(HARMON_IMAGE_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(HARMON_MASK_TOKEN))
    tokenizer_length = int(len(tokenizer))

    if model is not None:
        ensure_embedding_capacity(model, tokenizer_length=tokenizer_length)

    return TokenIds(
        image_token_id=image_id,
        mask_token_id=mask_id,
        tokenizer_length=tokenizer_length,
    )


def load_harmon_tokenizer(
    pretrained_model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    padding_side: str = 'right',
    local_files_only: bool = True,
    **kwargs,
):
    """Load a HuggingFace tokenizer and register the Harmon tokens on it.

    This is the runtime tokenizer builder used by Harmon configs via the
    mmengine ``dict(type=load_harmon_tokenizer, ...)`` convention. It returns
    the registered tokenizer (with ``<image>`` and ``<|dllm_mask|>`` appended).
    """

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        local_files_only=local_files_only,
        **kwargs,
    )
    register_harmon_tokens(tokenizer)
    return tokenizer


def _sha256_of_ids(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(str(int(token_id)).encode('ascii'))
        digest.update(b',')
    return digest.hexdigest()


def harmon_token_support(tokenizer) -> HarmonTokenSupport:
    """Build the explicit input / source / output support sets for a tokenizer.

    The tokenizer must already carry the two Harmon tokens
    (see :func:`register_harmon_tokens`).
    """

    length = int(len(tokenizer))
    image_id = int(tokenizer.convert_tokens_to_ids(HARMON_IMAGE_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(HARMON_MASK_TOKEN))

    special_ids = {int(tid) for tid in tokenizer.all_special_ids}
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        special_ids.add(int(eos_id))

    output_excluded = {image_id, mask_id}
    input_ids = tuple(range(length))
    output_ids = tuple(i for i in range(length) if i not in output_excluded)
    source_ids = tuple(i for i in output_ids if i not in special_ids)

    sha256 = {
        'input_token_ids': _sha256_of_ids(input_ids),
        'source_token_ids': _sha256_of_ids(source_ids),
        'output_token_ids': _sha256_of_ids(output_ids),
    }
    return HarmonTokenSupport(
        input_token_ids=input_ids,
        source_token_ids=source_ids,
        output_token_ids=output_ids,
        sha256=sha256,
    )


class TrainableMaskDelta(nn.Module):
    """Trainable additive delta for the mask-token embedding.

    The base embedding table is frozen; only ``delta`` receives gradient, and
    only at mask-token positions. At initialisation the effective mask-token
    embedding equals the legacy mask row, i.e.
    ``embedding[mask] + delta == embedding[legacy]``.
    """

    def __init__(self, mask_token_id: int, delta: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer(
            'mask_token_id', torch.tensor(int(mask_token_id), dtype=torch.long)
        )
        self.delta = nn.Parameter(delta.detach().clone().to(torch.float32))

    @classmethod
    def from_embeddings(
        cls,
        embedding: nn.Embedding,
        *,
        mask_token_id: int,
        legacy_mask_token_id: int,
    ) -> 'TrainableMaskDelta':
        weight = embedding.weight.detach()
        delta = (
            weight[int(legacy_mask_token_id)] - weight[int(mask_token_id)]
        ).clone()
        return cls(mask_token_id=mask_token_id, delta=delta)

    def forward(self, base: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        mask_positions = input_ids == self.mask_token_id
        if not bool(mask_positions.any()):
            return base
        delta = self.delta.to(dtype=base.dtype, device=base.device)
        addend = mask_positions.unsqueeze(-1).to(dtype=base.dtype) * delta
        return base + addend


class ValidLogitMask(nn.Module):
    """Constrain output logits to a valid, non-forbidden token set.

    Positions outside ``valid_token_ids`` and positions inside
    ``forbidden_token_ids`` are set to ``finfo(dtype).min`` so they can never
    be selected by ``argmax`` / softmax.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        valid_token_ids: Sequence[int],
        forbidden_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        vocab_size = int(vocab_size)
        allowed = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in valid_token_ids:
            token = int(tid)
            if 0 <= token < vocab_size:
                allowed[token] = True
        for tid in (forbidden_token_ids or ()):
            token = int(tid)
            if 0 <= token < vocab_size:
                allowed[token] = False
        self.register_buffer('valid_token_mask', allowed)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        vocab = int(logits.shape[-1])
        mask = self.valid_token_mask
        if mask.shape[0] < vocab:
            pad = torch.zeros(
                vocab - mask.shape[0], dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([mask, pad])
        elif mask.shape[0] > vocab:
            mask = mask[:vocab]
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, neg_inf)


def _tokenizer_tree_sha256(snapshot_dir: Path) -> str:
    """Hash every file in ``snapshot_dir`` except the metadata file.

    Each file contributes its name and the sha256 of its bytes, so any
    tampering with a tokenizer file (e.g. appending a byte) changes the tree
    digest.
    """

    snapshot_dir = Path(snapshot_dir)
    files = sorted(
        p
        for p in snapshot_dir.iterdir()
        if p.is_file() and p.name != TOKEN_METADATA_FILE
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\x00')
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode('ascii'))
        digest.update(b'\x00')
    return digest.hexdigest()


def write_harmon_tokenizer_snapshot(tokenizer, snapshot_dir) -> Dict:
    """Persist ``tokenizer`` plus an integrity metadata file in ``snapshot_dir``.

    Returns the metadata dict that is also written to ``TOKEN_METADATA_FILE``.
    The same dict round-trips through JSON, so ``json.loads(read_text())``
    equals the returned dict.
    """

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(snapshot_dir))

    image_id = int(tokenizer.convert_tokens_to_ids(HARMON_IMAGE_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(HARMON_MASK_TOKEN))
    support = harmon_token_support(tokenizer)

    metadata = {
        'schema_version': TOKEN_METADATA_SCHEMA_VERSION,
        'image_token_id': image_id,
        'mask_token_id': mask_id,
        'tokenizer_length': int(len(tokenizer)),
        'tree_sha256': _tokenizer_tree_sha256(snapshot_dir),
        'input_token_ids_sha256': support.sha256['input_token_ids'],
        'source_token_ids_sha256': support.sha256['source_token_ids'],
        'output_token_ids_sha256': support.sha256['output_token_ids'],
        'input_token_ids_count': len(support.input_token_ids),
        'source_token_ids_count': len(support.source_token_ids),
        'output_token_ids_count': len(support.output_token_ids),
    }
    (snapshot_dir / TOKEN_METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return metadata


def load_harmon_tokenizer_snapshot(snapshot_dir):
    """Load a tokenizer snapshot, verifying the tree integrity first.

    Raises ``ValueError`` mentioning ``tree SHA-256`` when any tokenizer file
    has been tampered with. Returns ``(tokenizer, token_ids)``.
    """

    snapshot_dir = Path(snapshot_dir)
    metadata_path = snapshot_dir / TOKEN_METADATA_FILE
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'missing {TOKEN_METADATA_FILE} in {snapshot_dir}'
        )
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema_version') != TOKEN_METADATA_SCHEMA_VERSION:
        raise ValueError('unsupported harmon tokenizer schema_version')

    tree_sha256 = _tokenizer_tree_sha256(snapshot_dir)
    if tree_sha256 != metadata.get('tree_sha256'):
        raise ValueError(
            'tree SHA-256 mismatch: tokenizer files have been tampered with'
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=True,
        local_files_only=True,
        padding_side='right',
    )
    token_ids = register_harmon_tokens(tokenizer)
    return tokenizer, token_ids


def validate_checkpoint_keys(result, allowed_missing_prefixes: Sequence[str] = ()) -> None:
    """Validate checkpoint load ``result`` against a missing-key allowlist.

    ``result`` must expose ``missing_keys`` and ``unexpected_keys`` (as a
    ``SimpleNamespace``, ``LoadInfo`` or similar).

    * Any unexpected key is a hard error.
    * A missing key is allowed only if it starts with one of
      ``allowed_missing_prefixes``; otherwise it is a hard error.
    """

    unexpected = list(getattr(result, 'unexpected_keys', None) or [])
    if unexpected:
        raise CheckpointKeyMismatchError(
            f'unexpected checkpoint keys: {unexpected}'
        )

    allowed = tuple(allowed_missing_prefixes)
    for key in (getattr(result, 'missing_keys', None) or []):
        if not any(str(key).startswith(prefix) for prefix in allowed):
            raise CheckpointKeyMismatchError(
                f'missing checkpoint key not in allowlist: {key}'
            )
