"""GUARD lagged conditional rank field (spec §5.6, Path C main line).

Implements the lagged rank-routed LoRA with RMS normalization described in
``docs/superpowers/specs/2026-08-08-guard-lora-accuracy-design.md`` §5.6:

    Delta v_{l,i}^{k+1} = B_l Diag(s_{l,i}^{k+1}) A_l v_{l,i}^{k+1}

    s_{l,i}^{k+1} = 1 + beta_l * tanh( f_l(t_{k+1}, u_{k+1}, h_{k+1},
                                          stopgrad(g_hat_i^k)) )

where ``g_hat_i^k`` is the trajectory evidence scalar produced by the
frozen :class:`RiskHead` (spec §5.4'.2) at round ``k``, stop-gradiented
before entering the gate, and consumed only at round ``k+1`` (lagged
design).  Region attention evidence from the failed §5.4 path
(``a_hat``, ``g_hat = max_R ReLU(...)``) is no longer used.

Key design constraints:
    - At init the gate produces ``s = 1`` (static baseline): the gate-function
      final layers are zero-init, so ``tanh(gate_fn(...)) = 0``.  The bounded
      amplitude starts at a small nonzero value, avoiding the zero-times-zero
      parameterisation in which neither factor receives a gradient.
    - ``beta = tanh(raw_beta)``, so ``|beta| <= 1``; the effective gate
      amplitude is bounded by 1.
    - Clean endpoint and first round (no lagged evidence) are exactly neutral:
      the gate returns ``s = 1`` even after training.
    - Evidence is always stop-gradient (detached) before entering the gate.
    - The gate output is RMS-normalised in the rank dimension (§5.6).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


EPSILON = 1e-6
INITIAL_BETA = 0.1

# Valid submodel identifiers (plan §S3, R7/R8 handled externally).
# R3-R6 are retained from the original design but now consume the
# trajectory evidence scalar g_hat (dim 1) instead of (g_hat, a_hat).
SUBMODELS = ('R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6')


class RankGate(nn.Module):
    """Per-token, per-rank lagged conditional gate for a single LoRA layer.

    The gate consumes the corruption state ``(t, u, h)`` and the lagged,
    stop-gradient trajectory evidence ``g_hat`` (scalar per token) from
    the previous round, and produces a per-rank scale ``s`` with
    ``s = 1`` at init (static baseline preserved).

    Parameters
    ----------
    rank:
        LoRA rank ``r``.
    state_dim:
        Dimension of the corruption state (default 3 for ``(t, u, h)``).
    evidence_dim:
        Dimension of the lagged evidence (default 1 for the scalar
        trajectory evidence ``g_hat`` from §5.4'.2).  Only the first
        dimension is the scalar ``g_hat``; additional dimensions are
        reserved for ablations and currently unused on the main line.
    gate_hidden:
        Hidden width of the gate MLP.
    submodel:
        Which submodel to use (R0-R6).  See module docstring.
    """

    def __init__(
        self,
        rank: int,
        *,
        state_dim: int = 3,
        evidence_dim: int = 1,
        gate_hidden: int = 64,
        submodel: str = 'R6',
    ) -> None:
        super().__init__()
        if submodel not in SUBMODELS:
            raise ValueError(
                f'submodel must be one of {SUBMODELS}, got {submodel!r}')
        self.rank = int(rank)
        self.state_dim = int(state_dim)
        self.evidence_dim = int(evidence_dim)
        self.submodel = submodel

        in_dim = state_dim + evidence_dim

        # Sub-MLPs for each component of the gate (§5.6).
        # s = 1 + beta * tanh( f_l(state, g_hat) ), where
        # f_l decomposes as c + u_t(t,h) + u_g(g) + u_tg(t,h,g).
        # Each MLP ends in ``rank`` outputs; zero-init so the gate starts at
        # s = 1 (tanh(0) = 0 -> s = 1).
        self.u_t = self._build_mlp(state_dim, gate_hidden, rank)
        self.u_g = self._build_mlp(evidence_dim, gate_hidden, rank)
        self.u_tg = self._build_mlp(in_dim, gate_hidden, rank)
        # Per-layer bias c_l (zero-init -> tanh(0) = 0 -> s = 1 at init).
        self.c = nn.Parameter(torch.zeros(rank))

        # raw_beta controls the amplitude of the gate.  beta = tanh(raw_beta)
        # so |beta| <= 1.  A small nonzero beta and a zero gate function keep
        # s exactly 1 while giving the final gate layers a usable gradient on
        # the first optimisation step.
        self.raw_beta = nn.Parameter(torch.empty(1))
        self.reset_identity_parameters()

    @staticmethod
    def _build_mlp(in_dim: int, hidden: int, rank: int) -> nn.Sequential:
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        # Zero-init final layer so the MLP outputs 0 at init.
        nn.init.zeros_(mlp[-1].weight)
        nn.init.zeros_(mlp[-1].bias)
        return mlp

    def reset_identity_parameters(self) -> None:
        """Restore the exact-identity, gradient-live gate initialization."""
        with torch.no_grad():
            self.c.zero_()
            self.raw_beta.fill_(math.atanh(INITIAL_BETA))
            for mlp in (self.u_t, self.u_g, self.u_tg):
                mlp[-1].weight.zero_()
                mlp[-1].bias.zero_()

    def forward(
        self,
        state: torch.Tensor,
        evidence: Optional[torch.Tensor],
        *,
        n_tokens: int,
        routing_enabled: bool = True,
    ) -> torch.Tensor:
        """Compute the per-rank scale ``s`` for each token.

        Parameters
        ----------
        state:
            Corruption state ``(t, u, h)`` for the current round, shape
            ``[batch, 3]`` or ``[3]``.  The state is broadcast across all
            tokens in the round.
        evidence:
            Lagged, stop-gradient trajectory evidence ``g_hat_i^{k-1}`` from
            the previous round, shape ``[batch, n_tokens, evidence_dim]`` or
            ``[batch, evidence_dim]`` (broadcast to ``n_tokens``).
            ``None`` means "no lagged context" (clean endpoint or first
            round), so the gate returns the neutral scale exactly.
        n_tokens:
            Number of tokens to broadcast ``s`` over.
        routing_enabled:
            Factorial switch. ``False`` keeps the same module/parameters but
            returns the neutral static-LoRA scale exactly.

        Returns
        -------
        torch.Tensor
            Per-token, per-rank scale, shape ``[batch, n_tokens, rank]``.
            At init this is all ones (the static baseline).
        """

        if state.ndim == 1:
            state = state.unsqueeze(0)
        batch = state.shape[0]
        device = state.device

        # Strict k -> k+1 contract: without evidence from a preceding round
        # (first denoising round or clean/prefix forward), the routed adapter
        # must be exactly the static LoRA even after the gate has trained.
        if not routing_enabled or evidence is None:
            return torch.ones(
                batch, n_tokens, self.rank,
                device=device, dtype=state.dtype,
            )

        if evidence is not None:
            evidence = evidence.detach()
            # ``state`` is constructed in Python (e.g. from the transfer
            # schedule in ``generate_dllm``) and may still live on CPU when
            # the gate is called from a CUDA forward path.  Move it to the
            # evidence's device so the downstream ``torch.cat`` in
            # ``_compute_per_token_f`` (state, evidence) does not crash with
            # "Expected all tensors to be on the same device".
            if evidence.device != device:
                state = state.to(evidence.device)
                device = state.device
            if evidence.ndim == 1:
                evidence = evidence.unsqueeze(0)
                is_per_token = False
            elif evidence.ndim == 2:
                # [batch, evidence_dim] -> broadcast across tokens.
                is_per_token = False
            else:
                # [batch, n_tokens, evidence_dim] -> per-token evidence.
                is_per_token = (evidence.shape[1] == n_tokens)

        # Run the gate MLPs in float32 to avoid bfloat16 underflow in the
        # small-amplitude gate outputs and gradients.
        in_dtype = state.dtype
        state_f = state.float()
        evidence_f = evidence.float()

        # ``self`` (the gate MLPs) may still live on CPU when ``state`` /
        # ``evidence`` are on CUDA.  This happens when the gate was created
        # in ``HarmonDev.__init__`` (before ``MMDistributedDataParallel``
        # moves the wrapped model to the rank device) and — for reasons
        # particular to PEFT's ``LoraModel`` wrapping — ``model.to(cuda)``
        # does not recurse into submodules registered via
        # ``module.register_module('guard_rank_gate', gate)`` on each
        # ``LoRALinear``.  Aligning here at the call site is the most
        # defensive fix: it is idempotent (no-op once the gate is already
        # on the right device/dtype) and stays correct regardless of where
        # the outer runner decides to place the model or what dtype it
        # casts the base model to (e.g. bfloat16 Qwen2.5).
        #
        # The base LLM is typically trained in bfloat16, so the gate MLP
        # weights inherit bfloat16 after ``model.to(cuda)``.  But we run
        # the gate math in float32 (see ``state_f`` / ``evidence_f`` above)
        # to avoid bfloat16 underflow in the small-amplitude gate outputs
        # and gradients.  Without this dtype alignment,
        # ``F.linear(float32_input, bfloat16_weight)`` raises
        # "mat1 and mat2 must have the same dtype".  Casting ``self`` to
        # float32 once is idempotent and cheap (gate has ~430k params).
        target_device = state_f.device
        own_param = next(self.parameters())
        if own_param.device != target_device or own_param.dtype != torch.float32:
            self.to(device=target_device, dtype=torch.float32)

        sm = self.submodel

        # Branch on per-token evidence BEFORE calling the MLPs: the u_tg
        # sub-MLP concatenates state (per-sample) and evidence (per-token)
        # along the last dim, so a 3D evidence tensor would crash it.  The
        # per-token path flattens to [batch*n, dim] first.
        if is_per_token:
            log_f = self._compute_per_token_f(
                state_f, evidence_f, n_tokens)
        else:
            # Compute each component of f_l = c + u_t + u_g + u_tg.
            if sm == 'R0':
                # Static: s = 1.
                log_f = self.c.unsqueeze(0).float()
            elif sm == 'R1':
                # Random same-norm gate: random init (caller re-inits MLPs).
                log_f = (self.c.unsqueeze(0).float()
                         + self.u_tg(
                             torch.cat([state_f, evidence_f], dim=-1)))
            elif sm == 'R2':
                # Uniform gate: only c.
                log_f = self.c.unsqueeze(0).float()
            elif sm == 'R3':
                # t-only: u_g=0, u_tg=0.
                log_f = (self.c.unsqueeze(0).float()
                         + self.u_t(state_f))
            elif sm == 'R4':
                # g-only: u_t=0, u_tg=0.
                log_f = (self.c.unsqueeze(0).float()
                         + self.u_g(evidence_f))
            elif sm == 'R5':
                # additive: u_tg=0.
                log_f = (self.c.unsqueeze(0).float()
                         + self.u_t(state_f) + self.u_g(evidence_f))
            else:  # sm == 'R6'
                # Full interaction.
                log_f = (self.c.unsqueeze(0).float()
                         + self.u_t(state_f)
                         + self.u_g(evidence_f)
                         + self.u_tg(
                             torch.cat([state_f, evidence_f], dim=-1)))
            # log_f is [batch, rank]; broadcast across tokens.
            log_f = log_f.unsqueeze(1).expand(batch, n_tokens, self.rank)

        # beta = tanh(raw_beta), |beta| <= 1.
        # Diagnostic override: set GUARD_DISABLE_RANK_GATE=1 to force
        # beta=0 (s=1, static baseline) and verify the dLLM path works
        # without RankGate modulation.  This isolates whether RankGate is
        # the root cause of inference degradation.
        import os as _os
        if _os.environ.get('GUARD_DISABLE_RANK_GATE', '0') in ('1', 'true', 'yes'):
            beta = torch.zeros_like(self.raw_beta)
        else:
            beta = torch.tanh(self.raw_beta)
        # s = 1 + beta * tanh(f).
        # At init: f=0 -> tanh(0)=0 -> s = 1 (static baseline).
        s = 1.0 + beta * torch.tanh(log_f)

        # RMS normalization in the rank dimension (§5.6).
        rms = torch.sqrt((s ** 2).mean(dim=-1, keepdim=True) + EPSILON)
        s = s / rms

        s = s.to(in_dtype)
        return s

    def _compute_per_token_f(
        self,
        state_f: torch.Tensor,
        evidence_f: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        """Compute per-token gate output ``f_l`` when evidence is per-token.

        ``state_f`` is ``[batch, state_dim]`` (broadcast across tokens) and
        ``evidence_f`` is ``[batch, n_tokens, evidence_dim]``.  The result
        is ``[batch, n_tokens, rank]``.
        """
        batch = state_f.shape[0]
        # Broadcast state to per-token: [batch, n_tokens, state_dim].
        state_exp = state_f.unsqueeze(1).expand(batch, n_tokens, self.state_dim)

        sm = self.submodel
        c = self.c.unsqueeze(0).unsqueeze(1).float()  # [1, 1, rank]
        log_f = c.expand(batch, n_tokens, self.rank)

        if sm == 'R0' or sm == 'R2':
            return log_f
        if sm == 'R3':
            # t-only: state is per-sample, so reshape to [batch*n, state_dim].
            flat_state = state_exp.reshape(batch * n_tokens, self.state_dim)
            out = self.u_t(flat_state).view(batch, n_tokens, self.rank)
            return log_f + out
        if sm == 'R4':
            flat_evidence = evidence_f.reshape(
                batch * n_tokens, self.evidence_dim)
            out = self.u_g(flat_evidence).view(batch, n_tokens, self.rank)
            return log_f + out
        if sm == 'R5':
            flat_state = state_exp.reshape(batch * n_tokens, self.state_dim)
            flat_evidence = evidence_f.reshape(
                batch * n_tokens, self.evidence_dim)
            out_t = self.u_t(flat_state).view(batch, n_tokens, self.rank)
            out_g = self.u_g(flat_evidence).view(batch, n_tokens, self.rank)
            return log_f + out_t + out_g
        # sm == 'R6' or 'R1': full interaction.
        flat_state = state_exp.reshape(batch * n_tokens, self.state_dim)
        flat_evidence = evidence_f.reshape(
            batch * n_tokens, self.evidence_dim)
        flat_in = torch.cat([flat_state, flat_evidence], dim=-1)
        out_t = self.u_t(flat_state).view(batch, n_tokens, self.rank)
        out_g = self.u_g(flat_evidence).view(batch, n_tokens, self.rank)
        out_tg = self.u_tg(flat_in).view(batch, n_tokens, self.rank)
        return log_f + out_t + out_g + out_tg


class GuardContext:
    """Cross-round context carrying stop-gradient trajectory evidence.

    The lagged contract (§5.6) requires round ``k`` evidence to control
    round ``k+1`` routing. :class:`GuardContext` stores the stop-gradient
    trajectory evidence ``g_hat`` from the previous round and the current
    corruption state, and reports whether lagged evidence is available
    (``has_lagged`` is ``False`` for the first round and the clean
    endpoint).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._evidence: Optional[torch.Tensor] = None  # [batch, seq, 1]
        self._state: Optional[torch.Tensor] = None
        self._is_clean: bool = False

    @property
    def has_lagged(self) -> bool:
        return self._evidence is not None and not self._is_clean

    def set_round_evidence(
        self,
        evidence: torch.Tensor,
        state: torch.Tensor,
        *,
        is_clean: bool = False,
    ) -> None:
        self._evidence = evidence.detach()
        self._state = state.detach()
        self._is_clean = bool(is_clean)

    def get_lagged_evidence(self) -> Optional[torch.Tensor]:
        if not self.has_lagged:
            return None
        return self._evidence

    def get_current_state(self) -> Optional[torch.Tensor]:
        return self._state


def scale_peft_lora_output(
    lora_delta: torch.Tensor,
    gate_scale: torch.Tensor,
) -> torch.Tensor:
    """Scale a PEFT LoRA output by the per-token, per-rank gate.

    This is a fallback that scales the final output by the mean gate across
    ranks.  The precise per-rank scaling is done in
    :class:`GuardedLoRAManager._patch_forward`.
    """

    mean_scale = gate_scale.mean(dim=-1, keepdim=False)
    return lora_delta * mean_scale.unsqueeze(-1)


class GuardedLoRAManager:
    """Patch PEFT LoRA layers to apply per-token RankGate scaling.

    The manager walks the PEFT model, wraps each ``peft.tuners.lora.Linear``
    with a :class:`RankGate`, and monkey-patches its ``forward`` to apply
    the diagonal rank scaling ``Diag(s)`` between ``lora_A`` and ``lora_B``.

    With a small nonzero bounded amplitude and zero-output gate MLPs, the
    gate output is ``s = 1`` and the patched forward is identical to the
    PEFT forward, while the gate can receive a gradient immediately.

    The manager accepts per-token trajectory evidence ``g_hat`` of shape
    ``[batch, seq, 1]`` (§5.6 lagged design), which the RankGate consumes
    alongside the corruption state to produce the per-rank scale.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int,
        submodel: str = 'R6',
    ) -> None:
        self.rank = int(rank)
        self.submodel = submodel
        self.gates: dict[str, RankGate] = {}
        # Current per-token evidence: [batch, seq, 1] or None.
        self.current_evidence: Optional[torch.Tensor] = None
        self.current_state: Optional[torch.Tensor] = None
        self.routing_enabled: bool = True
        self._original_forwards: dict[str, callable] = {}
        # Detached diagnostics; training captures a separate live cache only
        # when explicitly requested. Keyed by layer name, value
        # is a tensor of shape [batch, seq, rank].  Reset by ``reset()``.
        self.last_gate_scales: dict[str, torch.Tensor] = {}
        self.regularizer_scales: dict[str, torch.Tensor] = {}
        self.capture_regularizer = False
        self._install(model)

    def _install(self, model: nn.Module) -> None:
        try:
            from peft.tuners.lora import Linear as LoRALinear
        except ImportError:
            return

        for name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            gate = RankGate(
                rank=self.rank,
                submodel=self.submodel,
            )
            self.gates[name] = gate
            module.register_module('guard_rank_gate', gate)
            self._patch_forward(module, name)

    def _patch_forward(self, module: nn.Module, name: str) -> None:
        gate = self.gates[name]
        original_forward = module.forward
        self._original_forwards[name] = original_forward
        manager = self

        def guarded_forward(x, *args, **kwargs):
            result = module.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            adapter_names = kwargs.pop("adapter_names", None)
            if adapter_names is not None or module.merged:
                return original_forward(x, *args, **kwargs)

            for active_adapter in module.active_adapters:
                if active_adapter not in module.lora_A:
                    continue
                lora_A = module.lora_A[active_adapter]
                lora_B = module.lora_B[active_adapter]
                dropout = module.lora_dropout[active_adapter]
                scaling = module.scaling[active_adapter]
                x_cast = module._cast_input_dtype(x, lora_A.weight.dtype)

                ax = lora_A(dropout(x_cast))

                seq = ax.shape[1]
                state = manager.current_state
                evidence = manager.current_evidence
                if state is None:
                    gate_scale = torch.ones(
                        ax.shape[0], seq, gate.rank,
                        device=ax.device, dtype=ax.dtype,
                    )
                else:
                    # ``state`` is constructed in Python (e.g. in
                    # ``image2text_loss`` from ``state_features``) and may
                    # be on a different device than ``ax`` (the LoRA's
                    # activation, which follows the model's placement).
                    # The gate MLP weights themselves may also still be on
                    # CPU when ``MMDistributedDataParallel`` did not recurse
                    # into submodules registered via
                    # ``register_module('guard_rank_gate', gate)`` on the
                    # PEFT-wrapped LoRA layer.  Align everything to
                    # ``ax.device`` (the device the rest of this forward
                    # runs on) at the call site.
                    if state.device != ax.device:
                        state = state.to(ax.device)
                    if evidence is not None and evidence.device != ax.device:
                        evidence = evidence.to(ax.device)
                    gate_dev = next(gate.parameters()).device
                    if gate_dev != ax.device:
                        gate.to(ax.device)
                    # Evidence shape is [batch, full_seq, 1]; slice to the
                    # current sequence length (the LoRA layer is applied
                    # to the active token window).
                    if evidence is not None and evidence.shape[1] != seq:
                        # The evidence sequence length may differ from the
                        # LoRA input sequence length (e.g. if the forward
                        # is on a slice).  We align by tail-position: the
                        # current forward is on the last ``seq`` tokens.
                        ev_seq = evidence.shape[1]
                        if ev_seq >= seq:
                            ev_slice = evidence[:, ev_seq - seq:, :]
                        else:
                            # Pad with zeros at the head (no lagged context).
                            pad = torch.zeros(
                                evidence.shape[0], seq - ev_seq, 1,
                                device=evidence.device,
                                dtype=evidence.dtype,
                            )
                            ev_slice = torch.cat([pad, evidence], dim=1)
                    else:
                        ev_slice = evidence
                    gate_scale = gate(
                        state,
                        ev_slice,
                        n_tokens=seq,
                        routing_enabled=manager.routing_enabled,
                    )
                    if gate_scale.device != ax.device:
                        gate_scale = gate_scale.to(ax.device)
                    if gate_scale.dtype != ax.dtype:
                        gate_scale = gate_scale.to(ax.dtype)

                # PACT counterfactuals may hold controller values fixed while
                # retaining adapter gradients. Historical callers never set it.
                if getattr(manager, 'detach_gate_scales', False):
                    gate_scale = gate_scale.detach()
                ax_scaled = ax * gate_scale
                delta = lora_B(ax_scaled) * scaling
                result = result + delta
                # Cache the most recent gate_scale (detached) for the gate
                # regularizer L_gate (spec §6.1).  Kept on the gate's output
                # device/dtype to avoid extra copies in the hot path.
                manager.last_gate_scales[name] = gate_scale.detach()
                if manager.capture_regularizer and torch.is_grad_enabled():
                    manager.regularizer_scales[name] = gate_scale

            return result.to(torch_result_dtype)

        module.forward = guarded_forward

    def set_round_context(
        self,
        state: Optional[torch.Tensor],
        evidence: Optional[torch.Tensor] = None,
        *,
        routing_enabled: bool = True,
        capture_regularizer: bool = False,
    ) -> None:
        """Set the corruption state and per-token lagged evidence.

        Parameters
        ----------
        state:
            Corruption state ``(t, u, h)`` for the current round, shape
            ``[batch, 3]``.
        evidence:
            Per-token trajectory evidence ``g_hat`` from the previous
            round, shape ``[batch, seq, 1]`` or ``None`` (first round /
            clean endpoint).
        routing_enabled:
            Factorial runtime switch. It does not remove or replace modules.
        """
        self.current_state = state
        self.current_evidence = evidence
        self.routing_enabled = bool(routing_enabled)
        self.capture_regularizer = bool(capture_regularizer)
        self.regularizer_scales.clear()

    def reset(self) -> None:
        self.current_state = None
        self.current_evidence = None
        self.routing_enabled = True
        self.last_gate_scales.clear()
        self.regularizer_scales.clear()
        self.capture_regularizer = False

    def gate_parameters(self) -> list[nn.Parameter]:
        params = []
        for gate in self.gates.values():
            params.extend(gate.parameters())
        return params
