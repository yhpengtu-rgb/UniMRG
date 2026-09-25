"""Lightweight, output-preserving diagnostics for TRACER inference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch


class _Moments:
    """Accumulate tensor moments without synchronising every denoising round."""

    def __init__(self, *, neutral_value: Optional[float] = None) -> None:
        self.count = 0
        self.mean = None
        self.m2 = None
        self.minimum = None
        self.maximum = None
        self.nonneutral = None
        self.neutral_value = neutral_value

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().double().reshape(-1)
        if values.numel() == 0:
            return
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_m2 = (values - batch_mean).square().sum()
        minimum = values.min()
        maximum = values.max()
        if self.mean is None:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = minimum
            self.maximum = maximum
        else:
            combined_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 = (
                self.m2 + batch_m2
                + delta.square() * self.count * batch_count / combined_count
            )
            self.mean = (
                self.mean + delta * batch_count / combined_count
            )
            self.minimum = torch.minimum(self.minimum, minimum)
            self.maximum = torch.maximum(self.maximum, maximum)
        if self.neutral_value is not None:
            changed = (
                (values - float(self.neutral_value)).abs() > 1e-6
            ).sum()
            self.nonneutral = (
                changed if self.nonneutral is None
                else self.nonneutral + changed
            )
        self.count += batch_count

    def finalize(self) -> dict:
        if self.count == 0:
            return {
                'count': 0,
                'mean': None,
                'm2': 0.0,
                'std': None,
                'min': None,
                'max': None,
                'nonneutral_count': 0,
                'nonneutral_fraction': None,
                'neutral_value': self.neutral_value,
            }
        parts = [self.mean, self.m2, self.minimum, self.maximum]
        if self.nonneutral is not None:
            parts.append(self.nonneutral.float())
        values = torch.stack(parts).detach().cpu().tolist()
        mean, m2, minimum, maximum = values[:4]
        variance = max(m2 / self.count, 0.0)
        nonneutral = int(round(values[4])) if len(values) == 5 else 0
        return {
            'count': self.count,
            'mean': mean,
            'm2': m2,
            'std': math.sqrt(variance),
            'min': minimum,
            'max': maximum,
            'nonneutral_count': nonneutral,
            'nonneutral_fraction': (
                nonneutral / self.count
                if self.neutral_value is not None else None
            ),
            'neutral_value': self.neutral_value,
        }


class TracerRuntimeDiagnostics:
    """Collect one generation's control-flow and numerical diagnostics.

    The collector never changes model tensors. GPU reductions stay as scalar
    tensors until :meth:`finalize`, avoiding host synchronisation inside
    layers or denoising rounds.
    """

    def __init__(
        self,
        *,
        router_enabled: bool,
        policy_enabled: bool,
        block_size: int,
        denoising_steps: int,
        max_new_tokens: int,
    ) -> None:
        self.router_enabled = bool(router_enabled)
        self.policy_enabled = bool(policy_enabled)
        self.block_size = int(block_size)
        self.denoising_steps = int(denoising_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.blocks_started = 0
        self.blocks_completed = 0
        self.denoising_rounds = 0
        self.risk_head_calls = 0
        self.risk_policy_rounds = 0
        self.confidence_fallback_rounds = 0
        self.fallback_reasons: dict[str, int] = {}
        self._gate_scale = _Moments(neutral_value=1.0)
        self._active_gate_scale = _Moments(neutral_value=1.0)
        self._risk = _Moments()
        self._revision_count = None
        self._new_commit_count = None
        self._retained_count = None

    @staticmethod
    def _add_scalar(current, value: torch.Tensor):
        value = value.detach().to(dtype=torch.int64)
        return value if current is None else current + value

    def record_block_started(self) -> None:
        self.blocks_started += 1

    def record_block_completed(self) -> None:
        self.blocks_completed += 1

    def record_round_started(self) -> None:
        self.denoising_rounds += 1

    def record_gate_scales(
        self,
        scales: Iterable[torch.Tensor],
        *,
        active: bool,
    ) -> None:
        tensors = [value.detach().float().reshape(-1) for value in scales]
        if not tensors:
            return
        values = torch.cat(tensors)
        self._gate_scale.add(values)
        if active:
            self._active_gate_scale.add(values)

    def record_risk(self, risk: torch.Tensor) -> None:
        self.risk_head_calls += 1
        self._risk.add(risk)

    def record_confidence_fallback(self, reason: str) -> None:
        self.confidence_fallback_rounds += 1
        self.fallback_reasons[reason] = (
            self.fallback_reasons.get(reason, 0) + 1
        )

    def record_policy_decision(self, decision, *, used_risk: bool) -> None:
        if used_risk:
            self.risk_policy_rounds += 1
        self._revision_count = self._add_scalar(
            self._revision_count, decision.remask_mask.sum())
        self._new_commit_count = self._add_scalar(
            self._new_commit_count, decision.new_commit_mask.sum())
        self._retained_count = self._add_scalar(
            self._retained_count, decision.retained_mask.sum())

    def record_fallback_commits(
        self,
        new_commit_mask: torch.Tensor,
        prev_committed: torch.Tensor,
    ) -> None:
        """Count confidence-only commits using the same round semantics."""
        self._new_commit_count = self._add_scalar(
            self._new_commit_count, new_commit_mask.sum())
        self._retained_count = self._add_scalar(
            self._retained_count, prev_committed.sum())

    @staticmethod
    def _scalar_value(value) -> int:
        if value is None:
            return 0
        return int(value.detach().cpu().item())

    def finalize(
        self,
        generated_ids: torch.Tensor,
        *,
        eos_token_id: int,
    ) -> dict:
        generated = generated_ids.detach().cpu()
        output_tokens = []
        eos_emitted = []
        max_token_exhausted = []
        for row in generated:
            eos_positions = (row == int(eos_token_id)).nonzero(
                as_tuple=True)[0]
            emitted = len(eos_positions) > 0
            length = (
                int(eos_positions[0].item()) + 1
                if emitted else int(row.numel())
            )
            output_tokens.append(length)
            eos_emitted.append(emitted)
            max_token_exhausted.append(
                not emitted and length >= self.max_new_tokens)

        return {
            'schema_version': 1,
            'telemetry_only': True,
            'router_enabled': self.router_enabled,
            'policy_enabled': self.policy_enabled,
            'block_size': self.block_size,
            'denoising_steps': self.denoising_steps,
            'max_new_tokens': self.max_new_tokens,
            'blocks_started': self.blocks_started,
            'blocks_completed': self.blocks_completed,
            'denoising_rounds': self.denoising_rounds,
            'risk_head_calls': self.risk_head_calls,
            'risk_policy_rounds': self.risk_policy_rounds,
            'confidence_fallback_rounds': self.confidence_fallback_rounds,
            'fallback_reasons': dict(sorted(self.fallback_reasons.items())),
            'revision_count': self._scalar_value(self._revision_count),
            'new_commit_count': self._scalar_value(self._new_commit_count),
            'retained_count': self._scalar_value(self._retained_count),
            'output_tokens': output_tokens,
            'eos_emitted': eos_emitted,
            'max_token_exhausted': max_token_exhausted,
            'gate_scale': self._gate_scale.finalize(),
            'active_gate_scale': self._active_gate_scale.finalize(),
            'risk': self._risk.finalize(),
        }


def append_runtime_record(path: str | Path, record: dict) -> None:
    """Append one compact JSON record without touching model outputs."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with output.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')
