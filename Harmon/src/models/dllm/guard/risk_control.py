"""Fixed-NFE commit/revision policy for GUARD-LoRA (Path C trajectory evidence).

This module implements the trajectory-evidence risk head and commit/revision
policy described in
``docs/superpowers/specs/2026-08-08-guard-lora-accuracy-design.md`` §5.4'-5.5',
§5.7. Path C (2026-08-15) replaces the original region intervention evidence
(``g_lagged``, ``a_entropy``) with dLLM-native trajectory evidence:
``p_i^k``, ``H_i^k``, ``JS``, ``stable``, ``c_i^{<k}``, ``r_i^k``,
``c_bar_b^k``, ``(t_k, u_k, h_k)``.

Key components:

* :class:`RiskHead` - predicts per-token wrong-commit risk from the trajectory
  evidence in §5.4'.1.
* :class:`TrajectoryFeatures` - bundles the per-round trajectory quantities the
  risk head consumes.
* :class:`CommitRevisionPolicy` - selects the committed set from risk scores
  under the NFE-preserving budget.
* :class:`TrajectoryCollector` - records trajectory features and wrong-commit
  labels during dLLM inference for risk-fit.

The risk head is zero-initialised: at init it predicts 0.5 for every token,
so the policy falls back to the original confidence top-k selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch import nn
import torch.nn.functional as F


EPSILON = 1e-6
MAX_REVISION_FRACTION = 0.10


class RiskHead(nn.Module):
    """Predict per-token wrong-commit risk from dLLM trajectory evidence.

    Path C (spec §5.4'.2): the head consumes the 8 trajectory quantities:

    1. ``p_i^k`` - top-1 confidence (scalar).
    2. ``H_i^k`` - entropy of the candidate distribution (scalar).
    3. ``JS(p_i^k, p_i^{k-1})`` - cross-round Jensen-Shannon divergence
       (dLLM-only; 0 on the first round).
    4. ``1[y_i^k = y_i^{k-1}]`` - candidate stability (dLLM-only).
    5. ``c_i^{<k}`` - historical commit flag (dLLM-only).
    6. ``r_i^k`` - remask flag for the current round (dLLM-only).
    7. ``c_bar_b^k`` - cross-token commit correlation inside the block
       (dLLM-only).
    8. ``(t_k, u_k, h_k)`` - corruption state (dLLM-only).

    Initialisation is zero-risk: at init the head predicts 0.5 for every token,
    so the policy falls back to the original confidence top-k selection.

    Parameters
    ----------
    hidden_size:
        Dimension of the token hidden state used as a dense feature.
    state_dim:
        Dimension of the corruption state (default 3 for ``(t, u, h)``).
    proj_hidden:
        Hidden dimension of the two-layer MLP.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        state_dim: int = 3,
        proj_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.state_dim = int(state_dim)
        # Input features:
        #   hidden (hidden_size)
        # + p_i^k (1)
        # + H_i^k (1)
        # + JS (1)
        # + stable (1)
        # + c_i^{<k} (1)
        # + r_i^k (1)
        # + c_bar_b^k (1)
        # + state (state_dim)
        in_dim = hidden_size + 1 + 1 + 1 + 1 + 1 + 1 + 1 + state_dim
        self.risk_fn = nn.Sequential(
            nn.Linear(in_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, 1),
        )
        # Init final layer to zero so sigmoid output = 0.5 (zero-risk init).
        nn.init.zeros_(self.risk_fn[-1].weight)
        nn.init.zeros_(self.risk_fn[-1].bias)
        # Temperature for post-hoc calibration (Path C §5.5' ECE fix).
        # T=1 means no scaling; fit on validation set to minimise ECE.
        self.register_buffer('temperature', torch.ones(1))

    def forward(
        self,
        hidden: torch.Tensor,
        confidence: torch.Tensor,
        entropy: torch.Tensor,
        js_div: torch.Tensor,
        stable: torch.Tensor,
        committed_history: torch.Tensor,
        remask: torch.Tensor,
        block_commit_corr: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-token wrong-commit risk.

        Parameters
        ----------
        hidden:
            Token hidden state, shape ``[batch, seq, hidden_size]``.
        confidence:
            Top-1 softmax probability ``p_i^k``, shape ``[batch, seq]`` or
            ``[batch, seq, 1]``.
        entropy:
            Candidate distribution entropy ``H_i^k``, shape ``[batch, seq]``
            or ``[batch, seq, 1]``.
        js_div:
            JS divergence between consecutive rounds, shape ``[batch, seq]``
            or ``[batch, seq, 1]``.
        stable:
            Candidate stability indicator ``1[y^k = y^{k-1}]``, shape
            ``[batch, seq]`` or ``[batch, seq, 1]``.
        committed_history:
            Historical commit flag ``c_i^{<k}``, shape ``[batch, seq]`` or
            ``[batch, seq, 1]``.
        remask:
            Current-round remask flag ``r_i^k``, shape ``[batch, seq]`` or
            ``[batch, seq, 1]``.
        block_commit_corr:
            Cross-token commit correlation ``c_bar_b^k``, shape
            ``[batch, seq]`` or ``[batch, seq, 1]``.
        state:
            Corruption state ``(t, u, h)``, shape ``[batch, 3]`` or ``[3]``.

        Returns
        -------
        torch.Tensor
            Per-token risk in ``[0, 1]``, shape ``[batch, seq, 1]``.
        """

        batch, seq, _ = hidden.shape
        confidence = self._ensure_3d(confidence)
        entropy = self._ensure_3d(entropy)
        js_div = self._ensure_3d(js_div)
        stable = self._ensure_3d(stable)
        committed_history = self._ensure_3d(committed_history)
        remask = self._ensure_3d(remask)
        block_commit_corr = self._ensure_3d(block_commit_corr)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_exp = state.unsqueeze(1).expand(batch, seq, self.state_dim)

        # The base LLM is typically trained in bfloat16, so ``hidden`` and
        # ``self.risk_fn``'s weights inherit bfloat16 after ``model.to(cuda)``.
        # The other trajectory features (confidence/entropy/js_div/...) are
        # computed from softmax/etc. and are float32.  Align everything to
        # ``hidden.dtype`` (the LLM activation dtype) so
        # ``torch.cat([...])`` and ``F.linear`` don't raise
        # "mat1 and mat2 must have the same dtype".  This is idempotent and
        # cheap (RiskHead has ~200k params).
        target_dtype = hidden.dtype
        confidence = confidence.to(target_dtype)
        entropy = entropy.to(target_dtype)
        js_div = js_div.to(target_dtype)
        stable = stable.to(target_dtype)
        committed_history = committed_history.to(target_dtype)
        remask = remask.to(target_dtype)
        block_commit_corr = block_commit_corr.to(target_dtype)
        state_exp = state_exp.to(target_dtype)
        own_param_dtype = next(self.risk_fn.parameters()).dtype
        if own_param_dtype != target_dtype:
            self.risk_fn.to(dtype=target_dtype)

        risk_in = torch.cat(
            [
                hidden,
                confidence,
                entropy,
                js_div,
                stable,
                committed_history,
                remask,
                block_commit_corr,
                state_exp,
            ],
            dim=-1,
        )
        risk_logit = self.risk_fn(risk_in)  # [batch, seq, 1]
        # Apply temperature scaling: logit / T then sigmoid. T>1 sharpens, T<1
        # smooths. Trained to minimise ECE on validation set (see Path C
        # §5.5' Go/No-Go ECE fix in scripts/train_risk_head.py).
        calibrated_logit = risk_logit / self.temperature
        return torch.sigmoid(calibrated_logit)

    @torch.no_grad()
    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor,
                        t_range: tuple = (0.05, 5.0)) -> float:
        """Fit ``self.temperature`` to minimise ECE on the given logits.

        Parameters
        ----------
        logits:
            Pre-sigmoid risk logits from the trained risk head, shape ``[N]``
            or ``[N, 1]``. Calibration is post-hoc, so the head weights are
            frozen at this point.
        labels:
            Wrong-commit labels (0 or 1), shape ``[N]``.
        t_range:
            Search range for T (default ``(0.05, 5.0)``, log-space grid + 5
            rounds of refinement).

        Returns
        -------
        float
            The fitted temperature value.
        """
        logits = logits.flatten().float().cpu()
        labels = labels.flatten().float().cpu()

        def ece_of(t):
            probs = torch.sigmoid(logits / t)
            return _ece(probs, labels)

        # Coarse grid search in log-space.
        best_t = 1.0
        best_ece = ece_of(best_t)
        grid = torch.exp(torch.linspace(math.log(t_range[0]), math.log(t_range[1]), 200))
        for t in grid.tolist():
            e = ece_of(t)
            if e < best_ece:
                best_ece, best_t = e, t
        # 5 rounds of refinement around the best T.
        lo, hi = max(1e-3, best_t * 0.5), best_t * 2.0
        for _ in range(5):
            grid2 = torch.exp(torch.linspace(math.log(lo), math.log(hi), 200))
            for t in grid2.tolist():
                e = ece_of(t)
                if e < best_ece:
                    best_ece, best_t = e, t
            lo, hi = max(1e-3, best_t * 0.7), best_t * 1.3
        self.temperature.fill_(best_t)
        return float(best_t)

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(-1)
        return x


def _ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error (single-call helper for temperature fit)."""
    probs = probs.flatten().cpu()
    labels = labels.flatten().cpu()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.numel()
    for i in range(n_bins):
        if i == 0:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() > 0:
            avg_conf = probs[mask].mean().item()
            avg_acc = labels[mask].mean().item()
            ece += (mask.sum().item() / n) * abs(avg_conf - avg_acc)
    return float(ece)


class IsotonicCalibrator:
    """Non-parametric monotone calibrator for RiskHead probabilities.

    Fits a non-decreasing mapping ``p -> p_calibrated`` on the validation
    set via the pool-adjacent-violators (PAV) algorithm. Strictly more
    flexible than temperature scaling (Path C §5.5' ECE fix): it can correct
    any monotone distortion, while temperature scaling can only sharpen or
    flatten the global slope.

    Storage is two 1D tensors ``xs`` (sorted inputs) and ``ys`` (PAV-fitted
    values). Calibration at inference uses ``np.interp``-style lookup.
    """

    def __init__(self) -> None:
        self._xs: Optional[torch.Tensor] = None  # sorted unique inputs
        self._ys: Optional[torch.Tensor] = None  # calibrated values at xs
        self._device_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def fit(self, probs: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit the PAV isotonic regression of ``labels`` on ``probs``.

        Parameters
        ----------
        probs:
            Pre-calibration probabilities (e.g. ``sigmoid(logits / T)``).
        labels:
            0/1 wrong-commit labels.
        """
        probs = probs.flatten().float().cpu()
        labels = labels.flatten().float().cpu()
        order = torch.argsort(probs, stable=True)
        x_sorted = probs[order]
        y_sorted = labels[order]
        # Pool-adjacent-violators: walk left-to-right, averaging backwards
        # whenever a non-increasing block is detected.
        # We use a stack of (sum, count, n_blocks) entries.
        sums: List[float] = []
        counts: List[int] = []
        block_starts: List[int] = []
        for i in range(len(y_sorted)):
            sums.append(float(y_sorted[i].item()))
            counts.append(1)
            block_starts.append(i)
            while len(sums) >= 2 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
                # Merge last two blocks.
                sums[-2] += sums[-1]
                counts[-2] += counts[-1]
                block_starts.pop()
                sums.pop()
                counts.pop()
        # Reconstruct calibrated values at the unique input means.
        ys_fitted = torch.empty_like(y_sorted)
        idx = 0
        for s, c in zip(sums, counts):
            ys_fitted[idx:idx + c] = s / c
            idx += c
        # Store as unique (mean_x, fitted_y) pairs for fast lookup.
        # Because x_sorted may have ties, collapse them by averaging.
        self._xs = x_sorted.clone()
        self._ys = ys_fitted
        self._device_cache.clear()

    @torch.no_grad()
    def transform(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply the fitted isotonic mapping to new probabilities.

        Linear interpolation between stored breakpoints. Out-of-range
        inputs are clamped to the nearest endpoint.
        """
        if self._xs is None or self._ys is None:
            return probs
        input_shape = probs.shape
        probs = probs.flatten().float()
        # Cache the fitted table on each inference device. Moving ``probs``
        # to CPU here forces a device-wide synchronisation once per denoising
        # round and dominates the tiny risk-policy calculation.
        device_key = str(probs.device)
        cached = self._device_cache.get(device_key)
        if cached is None:
            cached = (
                self._xs.to(device=probs.device, dtype=probs.dtype),
                self._ys.to(device=probs.device, dtype=probs.dtype),
            )
            self._device_cache[device_key] = cached
        xs, ys = cached
        # right=True: returns number of xs <= v.
        idx_right = torch.searchsorted(xs, probs, right=True)
        # Clamp to valid interpolation range.
        idx_lo = (idx_right - 1).clamp(min=0)
        idx_hi = idx_right.clamp(max=xs.numel() - 1)
        # For inputs below xs[0] or above xs[-1], use the nearest endpoint.
        # For interior points, linearly interpolate.
        x_lo = xs[idx_lo]
        x_hi = xs[idx_hi]
        y_lo = ys[idx_lo]
        y_hi = ys[idx_hi]
        # Avoid division by zero when x_lo == x_hi (ties).
        denom = (x_hi - x_lo).clamp(min=1e-12)
        t = (probs - x_lo) / denom
        # When idx_lo == idx_hi (exact match or out of range), y_lo == y_hi.
        out = y_lo + t * (y_hi - y_lo)
        # Clamp to [0, 1] and restore the original tensor shape.
        return out.clamp(0.0, 1.0).reshape(input_shape)


@dataclass
class TrajectoryFeatures:
    """Per-round trajectory quantities consumed by the risk head (spec §5.4'.1)."""

    hidden: torch.Tensor  # [batch, seq, hidden_size]
    confidence: torch.Tensor  # [batch, seq] top-1 softmax probability
    entropy: torch.Tensor  # [batch, seq] candidate entropy
    js_div: torch.Tensor  # [batch, seq] JS divergence vs previous round
    stable: torch.Tensor  # [batch, seq] 1[y^k == y^{k-1}]
    committed_history: torch.Tensor  # [batch, seq] 1[committed at round <k]
    remask: torch.Tensor  # [batch, seq] 1[remasked at round k]
    block_commit_corr: torch.Tensor  # [batch, seq] mean(committed_history in block)
    state: torch.Tensor  # [batch, 3] corruption state (t, u, h)
    candidates: torch.Tensor  # [batch, seq] argmax token ids (for label construction)
    is_first_round: bool = False


def compute_trajectory_features(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    prev_probs: Optional[torch.Tensor],
    prev_candidates: Optional[torch.Tensor],
    committed_history: torch.Tensor,
    remask: torch.Tensor,
    state: torch.Tensor,
    block_size: int,
) -> TrajectoryFeatures:
    """Compute the §5.4'.1 trajectory features from a round's forward output.

    Parameters
    ----------
    hidden:
        Token hidden state from the LLM forward, ``[batch, seq, hidden_size]``.
    logits:
        Output logits, ``[batch, seq, vocab]``.
    prev_probs:
        Softmax probabilities from the previous round, ``[batch, seq, vocab]``
        or ``None`` for the first round.
    prev_candidates:
        Argmax token ids from the previous round, ``[batch, seq]`` or ``None``.
    committed_history:
        Cumulative commit flag ``c_i^{<k}``, ``[batch, seq]``.
    remask:
        Current-round remask flag ``r_i^k``, ``[batch, seq]``.
    state:
        Corruption state ``(t, u, h)``, ``[batch, 3]``.
    block_size:
        Block size used to compute ``c_bar_b^k`` (cross-token commit
        correlation inside the block).

    Returns
    -------
    TrajectoryFeatures
        The trajectory features for this round.
    """

    probs = F.softmax(logits.float(), dim=-1)
    confidence, candidates = probs.max(dim=-1)  # [batch, seq]
    entropy = -(probs * torch.log(probs + EPSILON)).sum(dim=-1)  # [batch, seq]

    if prev_probs is not None:
        m = 0.5 * (probs + prev_probs)
        kl_pm = (probs * (torch.log(probs + EPSILON) - torch.log(m + EPSILON))).sum(-1)
        kl_qm = (prev_probs * (torch.log(prev_probs + EPSILON) - torch.log(m + EPSILON))).sum(-1)
        js_div = 0.5 * (kl_pm + kl_qm)
        stable = (candidates == prev_candidates).float()
    else:
        js_div = torch.zeros_like(confidence)
        stable = torch.zeros_like(confidence)

    # c_bar_b^k: cross-token commit correlation inside the block.
    # For sequences shorter than block_size this falls back to the per-token
    # mean of committed_history across the whole sequence.
    batch, seq = confidence.shape
    if seq <= block_size:
        block_commit_corr = committed_history.float().mean(
            dim=1, keepdim=True
        ).expand(batch, seq)
    else:
        # Reshape to blocks and average within each block.
        n_blocks = (seq + block_size - 1) // block_size
        pad_len = n_blocks * block_size - seq
        if pad_len > 0:
            padded = F.pad(committed_history.float(), (0, pad_len))
        else:
            padded = committed_history.float()
        padded = padded.view(batch, n_blocks, block_size)
        per_block_mean = padded.mean(dim=2)  # [batch, n_blocks]
        block_commit_corr = per_block_mean.repeat_interleave(
            block_size, dim=1
        )[:, :seq]

    return TrajectoryFeatures(
        hidden=hidden,
        confidence=confidence,
        entropy=entropy,
        js_div=js_div,
        stable=stable,
        committed_history=committed_history,
        remask=remask,
        block_commit_corr=block_commit_corr,
        state=state,
        candidates=candidates,
        is_first_round=(prev_probs is None),
    )


class TrajectoryCollector:
    """Collect trajectory features and wrong-commit labels during dLLM inference.

    Used by ``scripts/collect_trajectory_evidence.py`` to build the risk-fit
    dataset. The collector is inserted as a hook into the dLLM inference loop:
    each round the caller invokes :meth:`record_round` with the trajectory
    features and the ground-truth tokens; the collector accumulates them for
    later offline training of the risk head.
    """

    def __init__(self, block_size: int) -> None:
        self.block_size = int(block_size)
        # Per-sample list of per-round records.
        self.records: List[Dict[str, torch.Tensor]] = []
        self._current: List[Dict[str, torch.Tensor]] = []

    def reset_sample(self) -> None:
        """Start a new sample."""
        if self._current:
            self.records.append(self._current)
        self._current = []

    def record_round(
        self,
        features: TrajectoryFeatures,
        block_start: int,
        block_end: int,
        ground_truth_ids: torch.Tensor,
    ) -> None:
        """Record a single round's trajectory for the given block.

        Parameters
        ----------
        features:
            Trajectory features for this round (from
            :func:`compute_trajectory_features`).
        block_start, block_end:
            Absolute positions of the block inside the full response.
        ground_truth_ids:
            Ground-truth response token ids, shape ``[batch, full_seq]``.
            Only the slice ``[block_start:block_end]`` is used.
        """

        # Slice ground truth to the current block.
        gt_block = ground_truth_ids[:, block_start:block_end]  # [batch, block]
        # wrong-commit label: 1[candidate != ground truth]
        wrong_commit = (features.candidates != gt_block).float()  # [batch, block]

        self._current.append(
            {
                "hidden": features.hidden.detach().cpu(),
                "confidence": features.confidence.detach().cpu(),
                "entropy": features.entropy.detach().cpu(),
                "js_div": features.js_div.detach().cpu(),
                "stable": features.stable.detach().cpu(),
                "committed_history": features.committed_history.detach().cpu(),
                "remask": features.remask.detach().cpu(),
                "block_commit_corr": features.block_commit_corr.detach().cpu(),
                "state": features.state.detach().cpu(),
                "candidates": features.candidates.detach().cpu(),
                "wrong_commit": wrong_commit.detach().cpu(),
                "is_first_round": features.is_first_round,
                "block_start": int(block_start),
                "block_end": int(block_end),
            }
        )

    def finalize(self) -> List[Dict[str, torch.Tensor]]:
        """Flush the last sample and return all records.

        Returns
        -------
        list of dict
            One entry per sample; each entry is a list of per-round dicts.
        """
        if self._current:
            self.records.append(self._current)
            self._current = []
        out = self.records
        self.records = []
        return out


@dataclass(frozen=True)
class CommitRevisionDecision:
    """Auditable fixed-budget committed-set transition.

    ``committed_mask`` is the complete current set after the transition;
    the other masks partition how it differs from ``prev_committed``.
    """

    committed_mask: torch.Tensor
    new_commit_mask: torch.Tensor
    remask_mask: torch.Tensor
    retained_mask: torch.Tensor


class CommitRevisionPolicy:
    """Update the cumulative committed set without increasing main-model NFE.

    ``target_committed_count`` is the cumulative budget from the original
    transfer schedule.  A risk-aware round may replace a bounded number of
    previously committed positions, but only with strictly lower-risk
    candidates.  First-round, missing-head, and uniform-risk states preserve
    previous commits and fill the cumulative budget by confidence.
    """

    def __init__(
        self,
        *,
        block_size: int,
        max_revision_fraction: float = MAX_REVISION_FRACTION,
    ) -> None:
        self.block_size = int(block_size)
        self.max_revisions = max(
            0, math.ceil(max_revision_fraction * self.block_size)
        )

    def select_committed(
        self,
        features: TrajectoryFeatures,
        target_committed_count: int,
        prev_committed: Optional[torch.Tensor] = None,
        risk_head: Optional[RiskHead] = None,
        risk_scores: Optional[torch.Tensor] = None,
    ) -> CommitRevisionDecision:
        """Select the complete committed set for the current round.

        Parameters
        ----------
        features:
            Trajectory features for the current round.
        target_committed_count:
            Cumulative number of committed tokens required after this round.
        prev_committed:
            Boolean mask of previously committed positions, shape
            ``[batch, seq]``. ``None`` means no prior commitments.
        risk_head:
            Trained risk head. If ``None`` or on the first round, fall back to
            confidence top-k.
        risk_scores:
            Precomputed output of the shared RiskHead, shape ``[batch, seq]``
            or ``[batch, seq, 1]``.  When supplied it is consumed directly so
            the router and policy use the exact same evidence tensor.

        Returns
        -------
        CommitRevisionDecision
            Complete set plus new/retained/remasked transition masks.
        """

        batch, seq = features.hidden.shape[0], features.hidden.shape[1]
        device = features.hidden.device
        target = min(max(int(target_committed_count), 0), seq)

        if prev_committed is None:
            prev_committed = torch.zeros(
                batch, seq, dtype=torch.bool, device=device)
        else:
            if prev_committed.shape != (batch, seq):
                raise ValueError(
                    'prev_committed must have shape '
                    f'{(batch, seq)}, got {tuple(prev_committed.shape)}')
            prev_committed = prev_committed.to(device=device, dtype=torch.bool)

        prev_counts = prev_committed.sum(dim=-1)
        if bool((prev_counts > target).any()):
            raise ValueError(
                'target_committed_count must be nondecreasing: '
                f'previous counts={prev_counts.tolist()}, target={target}')

        use_risk = (
            (risk_scores is not None or risk_head is not None)
            and not features.is_first_round
        )

        if not use_risk:
            committed_mask = self._confidence_fill(
                features.confidence, target, prev_committed)
            return self._decision(committed_mask, prev_committed)

        if risk_scores is None:
            risk = risk_head(
                features.hidden,
                features.confidence,
                features.entropy,
                features.js_div,
                features.stable,
                features.committed_history,
                features.remask,
                features.block_commit_corr,
                features.state,
            ).squeeze(-1)  # [batch, seq]
        else:
            risk = risk_scores.detach().to(device=device)
            if risk.ndim == 3 and risk.shape[-1] == 1:
                risk = risk.squeeze(-1)
            if risk.shape != (batch, seq):
                raise ValueError(
                    'risk_scores must have shape '
                    f'{(batch, seq)} or {(batch, seq, 1)}, '
                    f'got {tuple(risk_scores.shape)}')

        committed_mask = prev_committed.clone()
        for b in range(batch):
            risk_b = risk[b].float()
            prev_b = prev_committed[b]

            # A zero-initialised/uninformative RiskHead must reproduce the
            # confidence baseline instead of relying on arbitrary index ties.
            if float(risk_b.max() - risk_b.min()) <= 1e-8:
                committed_mask[b] = self._confidence_fill(
                    features.confidence[b:b + 1],
                    target,
                    prev_b.unsqueeze(0),
                )[0]
                continue

            need = target - int(prev_b.sum().item())
            available = (~prev_b).nonzero(as_tuple=False).flatten()
            if need > 0:
                add_order = self._risk_order(
                    risk_b[available],
                    features.confidence[b, available],
                )
                additions = available[add_order[:need]]
                committed_mask[b, additions] = True

            # Revisions only evict positions that were committed before this
            # round. Mandatory additions are not charged as revisions.
            outside = (~committed_mask[b]).nonzero(
                as_tuple=False).flatten()
            old = (prev_b & committed_mask[b]).nonzero(
                as_tuple=False).flatten()
            if self.max_revisions == 0 or outside.numel() == 0 or old.numel() == 0:
                continue

            outside_order = self._risk_order(
                risk_b[outside], features.confidence[b, outside])
            old_order = torch.argsort(risk_b[old], descending=True)
            pair_count = min(
                self.max_revisions, outside.numel(), old.numel())
            for j in range(pair_count):
                candidate = outside[outside_order[j]]
                evicted = old[old_order[j]]
                if not bool(risk_b[candidate] < risk_b[evicted]):
                    break
                committed_mask[b, evicted] = False
                committed_mask[b, candidate] = True

        return self._decision(committed_mask, prev_committed)

    @staticmethod
    def _decision(
        committed_mask: torch.Tensor,
        prev_committed: torch.Tensor,
    ) -> CommitRevisionDecision:
        return CommitRevisionDecision(
            committed_mask=committed_mask,
            new_commit_mask=committed_mask & ~prev_committed,
            remask_mask=prev_committed & ~committed_mask,
            retained_mask=prev_committed & committed_mask,
        )

    @staticmethod
    def _risk_order(
        risk: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """Sort by low risk, using high confidence for deterministic ties."""

        tie_break = torch.finfo(risk.dtype).eps * confidence.float()
        return torch.argsort(risk - tie_break, descending=False)

    @staticmethod
    def _confidence_fill(
        confidence: torch.Tensor,
        target_committed_count: int,
        prev_committed: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve old commits and fill a cumulative budget by confidence."""

        batch, seq = confidence.shape
        target = min(max(int(target_committed_count), 0), seq)
        mask = prev_committed.clone()
        for b in range(batch):
            need = target - int(mask[b].sum().item())
            if need <= 0:
                continue
            score = confidence[b].clone()
            score[mask[b]] = float('-inf')
            available = int((~mask[b]).sum().item())
            k = min(need, available)
            if k > 0:
                top_idx = score.topk(k, largest=True).indices
                mask[b, top_idx] = True
        return mask


def apply_commit_revision(
    *,
    block_ids: torch.Tensor,
    sampled_ids: torch.Tensor,
    mask_token_id: int,
    decision: CommitRevisionDecision,
) -> torch.Tensor:
    """Apply a committed-set decision, including a real token-to-mask move."""

    if block_ids.shape != sampled_ids.shape:
        raise ValueError('block_ids and sampled_ids must share shape')
    expected = block_ids.shape
    for name in (
        'committed_mask', 'new_commit_mask', 'remask_mask', 'retained_mask'
    ):
        value = getattr(decision, name)
        if value.shape != expected or value.dtype != torch.bool:
            raise ValueError(
                f'decision.{name} must be bool with shape {tuple(expected)}')

    # Retained positions keep the token committed in an earlier round. New
    # positions receive this round's sample. Everything else, including an
    # evicted old commitment, is an actual mask in the next model forward.
    next_ids = torch.full_like(block_ids, int(mask_token_id))
    next_ids = torch.where(decision.retained_mask, block_ids, next_ids)
    next_ids = torch.where(decision.new_commit_mask, sampled_ids, next_ids)
    return next_ids


def advance_corruption_state(
    state: torch.Tensor,
    *,
    committed_fraction: float,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Construct the adjacent next state for fixed-schedule training.

    ``state`` contains ``(t_k, u_k, h_k)``.  The returned state uses the
    schedule transition ``rho_{k+1}=max(rho_k-committed_fraction, epsilon)``
    and defines ``h_{k+1}=u_{k+1}-u_k``.  Recomputing ``u_k`` from ``t_k``
    avoids inconsistent mean-pooled ``t/u`` pairs.
    """

    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[-1] != 3:
        raise ValueError('state must have shape [batch, 3] or [3]')
    fraction = float(committed_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('committed_fraction must be in [0, 1]')

    t_k = state[:, 0].float().clamp(0.0, 1.0)
    rho_k = (1.0 - t_k).clamp(min=epsilon, max=1.0)
    u_k = -torch.log(rho_k)
    rho_kp1 = (rho_k - fraction).clamp(min=epsilon, max=1.0)
    t_kp1 = 1.0 - rho_kp1
    u_kp1 = -torch.log(rho_kp1)
    h_kp1 = u_kp1 - u_k
    return torch.stack((t_kp1, u_kp1, h_kp1), dim=-1).to(
        device=state.device, dtype=state.dtype)


def advance_masked_tokens(
    *,
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    response_mask: torch.Tensor,
    block_indices: torch.Tensor,
    mask_token_id: int,
    committed_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a real adjacent training input from teacher predictions.

    Each positive block independently commits
    ``ceil(block_length * committed_fraction)`` currently masked positions,
    ranked by teacher confidence.  This mirrors the inference transfer
    schedule and prevents a student state from advancing while its token
    embeddings remain at the previous corruption level.
    """

    if input_ids.ndim != 2:
        raise ValueError('input_ids must have shape [batch, seq]')
    batch, seq = input_ids.shape
    if logits.ndim != 3 or logits.shape[:2] != (batch, seq):
        raise ValueError('logits must have shape [batch, seq, vocab]')
    for name, value in (
        ('response_mask', response_mask),
        ('block_indices', block_indices),
    ):
        if value.shape != (batch, seq):
            raise ValueError(f'{name} must have shape {(batch, seq)}')
    fraction = float(committed_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError('committed_fraction must be in [0, 1]')

    probabilities = F.softmax(logits.float(), dim=-1)
    confidence, candidates = probabilities.max(dim=-1)
    response_mask = response_mask.to(
        device=input_ids.device, dtype=torch.bool)
    block_indices = block_indices.to(device=input_ids.device)
    eligible = response_mask & input_ids.eq(int(mask_token_id))
    commit_mask = torch.zeros_like(eligible)

    for b in range(batch):
        block_ids = torch.unique(block_indices[b][response_mask[b]])
        block_ids = block_ids[block_ids > 0]
        if bool(response_mask[b].any()) and block_ids.numel() == 0:
            raise ValueError(
                'response positions require positive block_indices')
        for block_id in block_ids.tolist():
            in_block = response_mask[b] & block_indices[b].eq(block_id)
            candidates_in_block = (eligible[b] & in_block).nonzero(
                as_tuple=False).flatten()
            if candidates_in_block.numel() == 0:
                continue
            target = math.ceil(int(in_block.sum().item()) * fraction)
            k = min(max(target, 0), int(candidates_in_block.numel()))
            if k == 0:
                continue
            order = torch.argsort(
                confidence[b, candidates_in_block], descending=True)
            commit_mask[b, candidates_in_block[order[:k]]] = True

    next_ids = torch.where(commit_mask, candidates, input_ids)
    return next_ids, commit_mask


def transfer_schedule_to_mask_probs(
    transfer_schedule: List[int],
    block_size: int,
    *,
    epsilon: float = EPSILON,
) -> torch.Tensor:
    """Convert a transfer schedule to per-round mask probabilities.

    A model forward at round ``k`` sees the state *before* that round's
    committed-set transition.  Thus the cumulative committed count is
    ``sum_{j<k} transfer_schedule[j]``.  ``h_k`` is the forward delta to the
    state after applying ``transfer_schedule[k]``.

    Parameters
    ----------
    transfer_schedule:
        Per-round committed-count budget (length ``K``).
    block_size:
        Number of tokens in the block.
    epsilon:
        Numerical floor for ``mask_prob`` to avoid ``log(0)``.

    Returns
    -------
    torch.Tensor
        Corruption states, shape ``[K, 3]`` (columns ``t, u, h``).
    """

    K = len(transfer_schedule)
    cum_before = 0
    states = []
    for k in range(K):
        remaining = max(0, block_size - cum_before)
        mask_prob = max(remaining / block_size, epsilon)
        t = 1.0 - mask_prob
        u = -math.log(mask_prob)
        cum_after = min(
            block_size, cum_before + int(transfer_schedule[k]))
        next_remaining = max(0, block_size - cum_after)
        next_mask_prob = max(next_remaining / block_size, epsilon)
        next_u = -math.log(next_mask_prob)
        h = next_u - u
        states.append((t, u, h))
        cum_before = cum_after
    return torch.tensor(states, dtype=torch.float32)
