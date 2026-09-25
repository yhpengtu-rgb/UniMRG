from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .corruption import CorruptionState


CORRUPTION_STATE_FIELDS = ('t', 'u', 'h')


def _per_token_state_value(
    value: torch.Tensor,
    *,
    token_count: int,
    field: str,
    device: torch.device,
) -> torch.Tensor:
    value = value.detach().to(device=device, dtype=torch.float32)
    if value.numel() == 1:
        value = value.reshape(()).expand(token_count)
    elif value.ndim == 1 and value.numel() == token_count:
        value = value.reshape(token_count)
    else:
        raise ValueError(
            f'CorruptionState.{field} must be scalar or contain one value '
            f'per noisy token ({token_count})'
        )
    if not torch.isfinite(value).all():
        raise ValueError(f'CorruptionState.{field} must be finite')
    return value


def pack_corruption_state_features(
    corruption_states: Sequence[Optional[CorruptionState]],
    token_types: torch.Tensor,
    *,
    fields: Tuple[str, ...] = CORRUPTION_STATE_FIELDS,
):
    """Align response-level state to valid noisy tokens in a padded batch."""
    if token_types.ndim != 2:
        raise ValueError('token_types must have shape [batch, sequence]')
    if len(corruption_states) != token_types.shape[0]:
        raise ValueError(
            'corruption_states count must equal token_types batch size'
        )
    fields = tuple(fields)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError('state fields must be nonempty and unique')
    unknown_fields = set(fields) - set(CORRUPTION_STATE_FIELDS)
    if unknown_fields:
        raise ValueError(
            f'unsupported corruption state fields: {sorted(unknown_fields)}'
        )

    features = torch.zeros(
        (*token_types.shape, len(fields)),
        device=token_types.device,
        dtype=torch.float32,
    )
    state_mask = torch.zeros_like(token_types, dtype=torch.bool)

    for batch_index, state in enumerate(corruption_states):
        noisy_positions = token_types[batch_index].eq(2).nonzero(
            as_tuple=False
        ).flatten()
        if state is None:
            if noisy_positions.numel():
                raise ValueError(
                    'missing CorruptionState for a sample with noisy tokens'
                )
            continue

        clean_index = state.clean_index_for_noisy
        if clean_index.ndim != 1:
            raise ValueError('clean_index_for_noisy must be one-dimensional')
        token_count = int(clean_index.numel())
        if noisy_positions.numel() != token_count:
            raise ValueError(
                'noisy token count does not match CorruptionState mapping: '
                f'{noisy_positions.numel()} != {token_count}'
            )
        valid = state.valid_target_mask.detach().to(
            device=token_types.device,
            dtype=torch.bool,
        )
        if valid.ndim != 1 or valid.numel() != token_count:
            raise ValueError(
                'valid_target_mask must contain one value per noisy token'
            )

        values = torch.stack([
            _per_token_state_value(
                getattr(state, field),
                token_count=token_count,
                field=field,
                device=token_types.device,
            )
            for field in fields
        ], dim=-1)
        valid_positions = noisy_positions[valid]
        features[batch_index, valid_positions] = values[valid]
        state_mask[batch_index, valid_positions] = True

    return features, state_mask


class CorruptionStateConditioner(nn.Module):
    """Zero-initialized additive timestep baseline for noisy token inputs."""

    def __init__(
        self,
        *,
        hidden_size: int,
        fields: Tuple[str, ...] = CORRUPTION_STATE_FIELDS,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size must be positive')
        fields = tuple(fields)
        if not fields or len(set(fields)) != len(fields):
            raise ValueError('state fields must be nonempty and unique')
        unknown_fields = set(fields) - set(CORRUPTION_STATE_FIELDS)
        if unknown_fields:
            raise ValueError(
                f'unsupported corruption state fields: '
                f'{sorted(unknown_fields)}'
            )
        self.fields = fields
        self.weight = nn.Parameter(
            torch.zeros(len(fields), int(hidden_size), dtype=torch.float32)
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        state_features: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds.ndim != 3:
            raise ValueError('inputs_embeds must have shape [batch, seq, hidden]')
        if state_features.shape != (
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            len(self.fields),
        ):
            raise ValueError(
                'state_features must align with inputs and configured fields'
            )
        if state_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError('state_mask must align with input tokens')
        if (
            state_features.device != inputs_embeds.device
            or state_mask.device != inputs_embeds.device
        ):
            raise ValueError('state tensors and inputs_embeds must share a device')
        if not torch.isfinite(state_features).all():
            raise ValueError('state_features must be finite')

        delta = torch.matmul(
            state_features.float(), self.weight.float()
        ).to(dtype=inputs_embeds.dtype)
        return torch.where(
            state_mask.to(dtype=torch.bool).unsqueeze(-1),
            inputs_embeds + delta,
            inputs_embeds,
        )


@dataclass(frozen=True)
class MatchedCausalView:
    """One side of an explicitly indexed, context-matched causal anchor."""

    logits: torch.Tensor
    matched_indices: torch.Tensor
    prefix_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    valid_vocab_mask: torch.Tensor
    causal_mask: torch.Tensor
    temperature: float


def _validate_causal_view(view: MatchedCausalView, *, name: str) -> None:
    if view.logits.ndim != 3:
        raise ValueError(f'{name} logits must have shape [batch, seq, vocab]')
    indices = view.matched_indices
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f'{name} matched indices must have shape [N,2]')
    if indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError(f'{name} matched indices must be integer')
    if indices.shape[0] == 0:
        raise ValueError(f'{name} matched indices must be nonempty')
    if torch.unique(indices, dim=0).shape[0] != indices.shape[0]:
        raise ValueError(f'{name} matched indices must be unique')
    if (
        (indices[:, 0] < 0).any()
        or (indices[:, 0] >= view.logits.shape[0]).any()
        or (indices[:, 1] < 0).any()
        or (indices[:, 1] >= view.logits.shape[1]).any()
    ):
        raise ValueError(f'{name} matched indices are out of range')
    if view.position_ids.shape != view.logits.shape[:2]:
        raise ValueError(f'{name} position_ids must align with logits')
    if view.causal_mask.shape != view.logits.shape[:2]:
        raise ValueError(f'{name} causal_mask must align with logits')
    selected_causal = view.causal_mask[
        indices[:, 0], indices[:, 1]
    ].to(dtype=torch.bool)
    if not selected_causal.all():
        raise ValueError(
            f'{name} matched indices must select only causal clean tokens'
        )
    if view.valid_vocab_mask.ndim != 1 or (
        view.valid_vocab_mask.numel() != view.logits.shape[-1]
    ):
        raise ValueError(
            f'{name} valid_vocab_mask must match the logits vocabulary'
        )
    if not view.valid_vocab_mask.to(dtype=torch.bool).any():
        raise ValueError(f'{name} valid_vocab_mask must be nonempty')
    temperature = torch.as_tensor(view.temperature, dtype=torch.float32)
    if (
        temperature.numel() != 1
        or not torch.isfinite(temperature)
        or temperature.item() <= 0.0
    ):
        raise ValueError(f'{name} temperature must be finite and positive')


def matched_clean_causal_kl(
    student: MatchedCausalView,
    teacher: MatchedCausalView,
) -> torch.Tensor:
    """Teacher-to-student KL for exactly matched clean causal contexts."""
    _validate_causal_view(student, name='student')
    _validate_causal_view(teacher, name='teacher')
    if student.matched_indices.shape[0] != teacher.matched_indices.shape[0]:
        raise ValueError('matched index counts must be equal')
    if student.logits.shape[-1] != teacher.logits.shape[-1]:
        raise ValueError('vocab sizes must match')
    if (
        student.prefix_ids.shape != teacher.prefix_ids.shape
        or not torch.equal(student.prefix_ids, teacher.prefix_ids)
    ):
        raise ValueError('prefix context must match exactly')

    student_indices = student.matched_indices.to(
        device=student.logits.device, dtype=torch.long
    )
    teacher_indices = teacher.matched_indices.to(
        device=teacher.logits.device, dtype=torch.long
    )
    student_positions = student.position_ids.to(
        device=student.logits.device
    )[student_indices[:, 0], student_indices[:, 1]]
    teacher_positions = teacher.position_ids.to(
        device=teacher.logits.device
    )[teacher_indices[:, 0], teacher_indices[:, 1]]
    if (
        student_positions.shape != teacher_positions.shape
        or not torch.equal(
            student_positions.detach().cpu(),
            teacher_positions.detach().cpu(),
        )
    ):
        raise ValueError('position context must match exactly')
    if (
        student.attention_mask.shape != teacher.attention_mask.shape
        or not torch.equal(
            student.attention_mask.detach().cpu(),
            teacher.attention_mask.detach().cpu(),
        )
    ):
        raise ValueError('attention mask context must match exactly')
    if not torch.equal(
        student.valid_vocab_mask.detach().to(dtype=torch.bool).cpu(),
        teacher.valid_vocab_mask.detach().to(dtype=torch.bool).cpu(),
    ):
        raise ValueError('vocab mask must match exactly')
    if float(student.temperature) != float(teacher.temperature):
        raise ValueError('temperature must match exactly')

    valid_vocab = student.valid_vocab_mask.to(
        device=student.logits.device,
        dtype=torch.bool,
    )
    student_selected = student.logits[
        student_indices[:, 0], student_indices[:, 1]
    ][:, valid_vocab].float()
    teacher_valid_vocab = teacher.valid_vocab_mask.to(
        device=teacher.logits.device,
        dtype=torch.bool,
    )
    teacher_selected = teacher.logits[
        teacher_indices[:, 0], teacher_indices[:, 1]
    ][:, teacher_valid_vocab].detach().float()
    if not torch.isfinite(student_selected).all():
        raise ValueError('student valid-vocabulary logits must be finite')
    if not torch.isfinite(teacher_selected).all():
        raise ValueError('teacher valid-vocabulary logits must be finite')

    temperature = float(student.temperature)
    student_log_probs = torch.log_softmax(
        student_selected / temperature, dim=-1
    )
    teacher_log_probs = torch.log_softmax(
        teacher_selected / temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    return (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1).mean()
