"""Corruption state, masked corruptor and safe token cross-entropy.

This module is part of the stable dLLM foundation (Task 1). It provides:

* :class:`CorruptionState` - an immutable, hashable description of one
  response-level corruption state (time ``t``, cumulative ``u``, step ``h``,
  realised mask, valid supervision targets and the clean/noisy index map).
* :class:`MaskedCorruptor` - realises a mask-source corruption of a clean
  response under a per-token schedule time ``t``, guaranteeing that a nonempty
  response with valid labels always contributes at least one active target when
  ``enforce_nonempty=True``.
* :func:`safe_token_cross_entropy` - a reduction-safe token CE that returns a
  backpropable finite zero for empty inputs and otherwise matches
  :func:`torch.nn.functional.cross_entropy` mean reduction.

The contracts implemented here are pinned by ``Harmon/tests/test_stable_baseline.py``
and the integration in ``Harmon/src/models/harmon_dev.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EPSILON = 1e-6


@dataclass(frozen=True)
class CorruptionState:
    """Immutable description of one response-level corruption state.

    All tensor fields are aligned to the noisy response token axis, i.e. they
    carry one value per noisy token and ``clean_index_for_noisy`` maps each
    noisy position back to its clean source position.

    Fields
    ------
    source_kind:
        Kind of corruption source. The stable foundation only emits ``"mask"``.
    t:
        Per-noisy-token schedule time in ``[0, 1]`` (``1`` = clean endpoint).
    u:
        Per-noisy-token cumulative signal ``u = -log(max(rho, eps))`` where
        ``rho`` is the local mask probability ``1 - t``.
    h:
        Per-noisy-token step budget ``h``. The stable foundation produces a
        single corruption round, so ``h`` is zero; paired ``k -> k+1`` states
        are introduced by the GUARD extension.
    corruption_prob:
        Per-noisy-token mask probability ``clamp(1 - t, 0, 1)``.
    hazard:
        Per-noisy-token absorbing hazard, equal to ``u`` for the mask source.
    active_mask:
        Boolean per noisy token, ``True`` where the token was replaced by the
        mask token. This includes positions whose label is ``-100``.
    clean_index_for_noisy:
        Index of the clean source token for every noisy position. For the
        stable foundation this is ``arange(n_noisy)``.
    valid_target_mask:
        Boolean per noisy token, ``True`` where the position is a valid
        supervision target (masked, has a real label and is not excluded).
    legitimate_empty:
        ``True`` when the response has no valid-label token at all, in which
        case the contribution must be a finite zero rather than an error.
    """

    source_kind: str
    t: torch.Tensor
    u: torch.Tensor
    h: torch.Tensor
    corruption_prob: torch.Tensor
    hazard: torch.Tensor
    active_mask: torch.Tensor
    clean_index_for_noisy: torch.Tensor
    valid_target_mask: torch.Tensor
    legitimate_empty: bool

    def require_valid_supervision(self) -> None:
        """Fail closed for a nonempty response with zero active targets.

        A response that contains at least one valid-label token (i.e. is not
        ``legitimate_empty``) must contribute at least one valid target. The
        opposite case is the ``nonempty-but-zero-active`` invariant violation
        that the stable foundation rejects rather than silently skips.
        """

        if self.legitimate_empty:
            return
        if not bool(self.valid_target_mask.any()):
            raise RuntimeError(
                'nonempty response has zero active targets'
            )


def _as_bool_tensor(mask: torch.Tensor, *, n: int, device: torch.device) -> torch.Tensor:
    mask = mask.detach().to(device=device)
    if mask.numel() == 1:
        return mask.reshape(()).to(torch.bool).expand(n).clone()
    if mask.numel() != n:
        raise ValueError(
            f'mask length {mask.numel()} does not match noisy token count {n}'
        )
    return mask.to(torch.bool)


class MaskedCorruptor:
    """Realise a mask-source corruption of a clean response.

    Parameters
    ----------
    mask_token_id:
        Token id used to replace masked positions.
    enforce_nonempty:
        When ``True`` a response that has valid-label tokens but draws no mask
        is forced to mask exactly one valid-label candidate, preserving the
        ``nonempty -> >=1 active target`` invariant.
    excluded_token_ids:
        Token ids that are never eligible for masking (e.g. the image
        placeholder and the ``IMAGE_TOKEN_INDEX`` sentinel).
    """

    def __init__(
        self,
        *,
        mask_token_id: int,
        enforce_nonempty: bool = True,
        excluded_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.mask_token_id = int(mask_token_id)
        self.enforce_nonempty = bool(enforce_nonempty)
        self.excluded_token_ids = tuple(
            int(tid) for tid in (excluded_token_ids or ())
        )

    def corrupt(
        self,
        clean_ids: torch.Tensor,
        clean_labels: torch.Tensor,
        *,
        t: torch.Tensor,
        random_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, CorruptionState]:
        """Corrupt ``clean_ids`` and return ``(noisy_ids, state)``.

        ``clean_ids`` and ``clean_labels`` are 1-D tensors of equal length over
        the response. ``t`` and ``random_values`` are per-response-token tensors
        broadcastable to that length.
        """

        if clean_ids.shape != clean_labels.shape:
            raise ValueError('clean_ids and clean_labels must share shape')
        if clean_ids.ndim != 1:
            raise ValueError('clean_ids must be one-dimensional')
        device = clean_ids.device
        n = int(clean_ids.numel())

        t = t.detach().to(device=device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.reshape(()).expand(n)
        if t.numel() != n:
            raise ValueError(
                f't length {t.numel()} does not match response length {n}'
            )
        t = t.clamp(0.0, 1.0)

        random_values = random_values.detach().to(device=device, dtype=torch.float32)
        if random_values.numel() == 1:
            random_values = random_values.reshape(()).expand(n)
        if random_values.numel() != n:
            raise ValueError(
                f'random_values length {random_values.numel()} does not match '
                f'response length {n}'
            )

        excluded = torch.zeros(n, dtype=torch.bool, device=device)
        if self.excluded_token_ids:
            excluded = torch.isin(
                clean_ids,
                torch.tensor(
                    self.excluded_token_ids,
                    dtype=clean_ids.dtype,
                    device=device,
                ),
            )

        has_label = clean_labels.ne(-100)
        valid_label_candidates = has_label & ~excluded

        mask_prob = (1.0 - t).clamp(0.0, 1.0)
        draw = random_values < mask_prob
        masked = draw & ~excluded

        legitimate_empty = not bool(valid_label_candidates.any())
        if not legitimate_empty and not bool((masked & valid_label_candidates).any()):
            # Either the draw masked nothing valid, or the response has valid
            # labels but none were masked. enforce_nonempty guarantees one
            # active target by forcing the most-likely candidate.
            if self.enforce_nonempty:
                candidate_random = random_values.clone()
                candidate_random[~valid_label_candidates] = float('inf')
                forced_index = int(torch.argmin(candidate_random).item())
                masked[forced_index] = True
            # When enforce_nonempty is False we leave masked as-is; the caller
            # is responsible for handling the zero-active case.

        active_mask = masked
        valid_target_mask = masked & valid_label_candidates

        noisy_ids = clean_ids.clone()
        noisy_ids[active_mask] = self.mask_token_id

        rho = mask_prob.clamp(min=EPSILON, max=1.0)
        u = -torch.log(rho)
        hazard = u
        h = torch.zeros_like(t)
        corruption_prob = mask_prob

        clean_index_for_noisy = torch.arange(n, dtype=torch.long, device=device)

        state = CorruptionState(
            source_kind='mask',
            t=t,
            u=u,
            h=h,
            corruption_prob=corruption_prob,
            hazard=hazard,
            active_mask=active_mask,
            clean_index_for_noisy=clean_index_for_noisy,
            valid_target_mask=valid_target_mask,
            legitimate_empty=legitimate_empty,
        )
        return noisy_ids, state


def safe_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reduction-safe token cross-entropy.

    * Empty input (no tokens) returns a backpropable finite zero, so legitimate
      empty responses contribute ``0`` to the loss without producing ``NaN``.
    * Otherwise the loss is the mean over valid (``label >= 0``) tokens, which
      matches :func:`torch.nn.functional.cross_entropy` mean reduction when the
      caller has already filtered ``-100`` labels (as ``harmon_dev`` does).

    The result is always a scalar tensor that stays attached to the autograd
    graph of ``logits`` so gradient flows even for the empty case.
    """

    if labels.numel() == 0:
        # Keep the result attached to logits so backward still flows.
        return logits.sum() * 0.0

    valid = labels >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0

    selected_logits = logits[valid]
    selected_labels = labels[valid].to(dtype=torch.long)
    per_token = F.cross_entropy(
        selected_logits.float(),
        selected_labels,
        reduction='none',
    )
    count = per_token.numel()
    return per_token.sum() / max(count, 1)
