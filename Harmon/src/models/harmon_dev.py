import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, replace
from numbers import Real
from torch.nn.modules.module import T
from typing import Optional
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX
from transformers.cache_utils import DynamicCache
from .harmon import Harmon
from .dllm import (
    CorruptionStateConditioner,
    MaskedCorruptor,
    TrainableMaskDelta,
    ValidLogitMask,
    harmon_token_support,
    pack_corruption_state_features,
    register_harmon_tokens,
    safe_token_cross_entropy,
)
from torch.nn.utils.rnn import pad_sequence


def _normalise_ddp_state_dict(state_dict):
    """Return checkpoint keys in the namespace of a non-DDP model.

    MMEngine checkpoints produced by DDP may retain a leading ``module.``
    even after ``guess_load_checkpoint`` unwraps the outer checkpoint dict.
    ``strict=False`` does not report this as an exception, so normalise the
    namespace explicitly and reject ambiguous key collisions.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(
            'pretrained checkpoint must resolve to a state_dict mapping, '
            f'got {type(state_dict).__name__}')

    normalised = {}
    stripped_keys = 0
    for original_key, value in state_dict.items():
        key = original_key
        if key.startswith('module.'):
            key = key[len('module.'):]
            stripped_keys += 1
        if key in normalised:
            raise RuntimeError(
                'pretrained checkpoint contains colliding keys after DDP '
                f'prefix normalisation: {original_key!r} -> {key!r}')
        normalised[key] = value
    return normalised, stripped_keys


def _load_pretrained_weights(
    model,
    pretrained_pth,
    required_key_groups=(),
):
    """Load and audit a hot-start checkpoint instead of failing silently."""
    raw_state_dict = guess_load_checkpoint(pretrained_pth)
    state_dict, stripped_keys = _normalise_ddp_state_dict(raw_state_dict)
    model_keys = set(model.state_dict())
    # Raw Harmon checkpoints precede PEFT wrapping. Map only exact target keys;
    # never silently discard the pretrained language model during cold starts.
    if any(k.startswith('llm.') for k in state_dict) and not any(
            k.startswith('llm.base_model.') for k in state_dict):
        remapped = {}
        for key, value in state_dict.items():
            target = key
            if key.startswith('llm.') and key not in model_keys:
                candidate = 'llm.base_model.model.' + key[len('llm.'):]
                if candidate in model_keys:
                    target = candidate
                else:
                    prefix, suffix = candidate.rsplit('.', 1)
                    candidate = prefix + '.base_layer.' + suffix
                    if candidate in model_keys:
                        target = candidate
                    else:
                        raise RuntimeError(f'Unmatched raw Harmon LLM key: {key}')
            if target in remapped:
                raise RuntimeError(f'Checkpoint key collision: {target}')
            remapped[target] = value
        state_dict = remapped
    matched_keys = model_keys.intersection(state_dict)
    if not matched_keys:
        sample_keys = list(state_dict)[:3]
        raise RuntimeError(
            f'Pretrained checkpoint {pretrained_pth!r} matched 0 model keys '
            f'out of {len(state_dict)} checkpoint keys. Sample checkpoint '
            f'keys: {sample_keys}')

    required_group_counts = {}
    for marker in required_key_groups:
        if not isinstance(marker, str) or not marker:
            raise ValueError(
                'pretrained_required_key_groups entries must be non-empty '
                f'strings, got {marker!r}')
        checkpoint_group = {
            key for key in state_dict if marker in key
        }
        matched_group = checkpoint_group.intersection(matched_keys)
        if not checkpoint_group:
            raise RuntimeError(
                f'Pretrained checkpoint {pretrained_pth!r} is missing '
                f'required key group {marker!r}')
        if matched_group != checkpoint_group:
            unmatched = sorted(checkpoint_group - matched_group)[:3]
            raise RuntimeError(
                f'Pretrained checkpoint {pretrained_pth!r} required key '
                f'group {marker!r} matched {len(matched_group)}/'
                f'{len(checkpoint_group)} keys; unmatched sample={unmatched}')
        required_group_counts[marker] = len(matched_group)

    info = model.load_state_dict(state_dict, strict=False)
    loaded_keys = len(state_dict) - len(info.unexpected_keys)
    report = {
        'path': str(pretrained_pth),
        'checkpoint_keys': len(state_dict),
        'matched_keys': loaded_keys,
        'namespace_matched_keys': len(matched_keys),
        'stripped_ddp_keys': stripped_keys,
        'missing_keys': len(info.missing_keys),
        'unexpected_keys': len(info.unexpected_keys),
        'missing_key_names': list(info.missing_keys),
        'unexpected_key_names': list(info.unexpected_keys),
        'required_key_groups': required_group_counts,
    }
    model.pretrained_load_report = report
    print_log(
        'Loaded pretrained weights from {}: matched={}/{}, '
        'stripped_ddp={}, missing={}, unexpected={}'.format(
            pretrained_pth,
            report['matched_keys'],
            report['checkpoint_keys'],
            report['stripped_ddp_keys'],
            report['missing_keys'],
            report['unexpected_keys'],
        ))
    return report


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@dataclass(frozen=True)
class _Image2TextLossResult:
    base_loss: torch.Tensor
    last_hidden_state: torch.Tensor
    response_mask: Optional[torch.Tensor]
    loss_mask: Optional[torch.Tensor]


class HarmonDev(Harmon, BaseModel):
    def __init__(self,
                 grad_scale=0.1,
                 loss_weights={'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0, 'edit': 1.0},
                 pretrained_pth=None,
                 pretrained_required_key_groups=(),
                 freeze_llm=False,
                 freeze_mar_encoder=False,
                 freeze_mar_decoder=False,
                 freeze_proj_in=False,
                 freeze_proj_out=False,
                 gradient_checkpointing=True,
                 dllm=False,
                 block_size=32,
                 prior_dist = 'Mask',
                 min_mask_rate = 0.001,
                 max_mask_rate = 1.0,
                 lora=None,
                 mask_token_strategy='legacy',
                 legacy_mask_token_id=151671,
                 trainable_mask_delta=False,
                 enforce_nonempty_dllm_targets=False,
                 safe_dllm_loss=False,
                 mask_invalid_dllm_logits=False,
                 state_conditioner=False,
                 state_conditioner_fields=('t', 'u', 'h'),
                 state_conditioner_mode='additive_zero_init',
                 clean_anchor_weight=0.0,
                 guard_enabled=False,
                 guard_submodel='R6',
                 guard_risk_head_path=None,
                 guard_risk_head_hidden=None,
                 guard_risk_head_proj_hidden=128,
                 guard_risk_head_state_dim=3,
                 guard_train_rounds=1,
                 guard_gate_regularizer_weight=0.0,
                 guard_base_loss_weight=0.0,
                 tracer_router_enabled=True,
                 tracer_policy_enabled=True,
                 mrare_enabled=False,
                 mrare_distill_enabled=False,
                 mrare_lambda_positive=0.0,
                 mrare_lambda_rank=0.0,
                 mrare_lambda_distill=0.0,
                 mrare_margin=0.25,
                 mrare_tau_energy=1.0,
                 mrare_tau_token=1.0,
                 mrare_force_changed_token=False,
                 mrare_positive_only=False,
                 mrfc_enabled=False,
                 mrfc_critic_path=None,
                 mrfc_lambda_pair=0.0,
                 mrfc_lambda_distill=0.0,
                 mrfc_margin=0.25,
                 mrfc_tau_critic=1.0,
                 **kwargs
                 ):
        if not isinstance(mrare_enabled, bool):
            raise TypeError('mrare_enabled must be a bool')
        if not isinstance(mrare_distill_enabled, bool):
            raise TypeError('mrare_distill_enabled must be a bool')
        if mrare_distill_enabled and not mrare_enabled:
            raise ValueError(
                'mrare_distill_enabled requires mrare_enabled=True')
        if not isinstance(mrare_positive_only, bool):
            raise TypeError('mrare_positive_only must be a bool')
        if mrare_positive_only and not mrare_enabled:
            raise ValueError(
                'mrare_positive_only requires mrare_enabled=True')
        if mrare_positive_only and mrare_distill_enabled:
            raise ValueError(
                'mrare_positive_only is incompatible with distillation')
        if not isinstance(mrfc_enabled, bool):
            raise TypeError('mrfc_enabled must be a bool')
        if mrfc_enabled and mrare_enabled:
            raise ValueError('MR-FC and MR-ARE cannot be enabled together')

        mrare_scalars = {
            'mrare_lambda_positive': mrare_lambda_positive,
            'mrare_lambda_rank': mrare_lambda_rank,
            'mrare_lambda_distill': mrare_lambda_distill,
            'mrare_margin': mrare_margin,
            'mrare_tau_energy': mrare_tau_energy,
            'mrare_tau_token': mrare_tau_token,
        }
        for name, value in mrare_scalars.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'{name} must be a real scalar')
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite')
        for name in (
            'mrare_lambda_positive',
            'mrare_lambda_rank',
            'mrare_lambda_distill',
            'mrare_margin',
        ):
            if float(mrare_scalars[name]) < 0.0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('mrare_tau_energy', 'mrare_tau_token'):
            if float(mrare_scalars[name]) <= 0.0:
                raise ValueError(f'{name} must be positive')
        mrfc_scalars = {
            'mrfc_lambda_pair': mrfc_lambda_pair,
            'mrfc_lambda_distill': mrfc_lambda_distill,
            'mrfc_margin': mrfc_margin,
            'mrfc_tau_critic': mrfc_tau_critic,
        }
        for name, value in mrfc_scalars.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f'{name} must be a real scalar')
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite')
        for name in ('mrfc_lambda_pair', 'mrfc_lambda_distill', 'mrfc_margin'):
            if float(mrfc_scalars[name]) < 0.0:
                raise ValueError(f'{name} must be nonnegative')
        if float(mrfc_tau_critic) <= 0.0:
            raise ValueError('mrfc_tau_critic must be positive')
        if mrfc_enabled and not mrfc_critic_path:
            raise ValueError('mrfc_enabled requires mrfc_critic_path')

        super().__init__(**kwargs)
        self.mrare_enabled = mrare_enabled
        self.mrare_distill_enabled = mrare_distill_enabled
        self.mrare_lambda_positive = float(mrare_lambda_positive)
        self.mrare_lambda_rank = float(mrare_lambda_rank)
        self.mrare_lambda_distill = float(mrare_lambda_distill)
        self.mrare_margin = float(mrare_margin)
        self.mrare_tau_energy = float(mrare_tau_energy)
        self.mrare_tau_token = float(mrare_tau_token)
        if not isinstance(mrare_force_changed_token, bool):
            raise TypeError('mrare_force_changed_token must be a bool')
        if mrare_force_changed_token and not mrare_enabled:
            raise ValueError(
                'mrare_force_changed_token requires mrare_enabled=True')
        self.mrare_force_changed_token = mrare_force_changed_token
        self.mrare_positive_only = mrare_positive_only
        self.mrfc_enabled = mrfc_enabled
        self.mrfc_lambda_pair = float(mrfc_lambda_pair)
        self.mrfc_lambda_distill = float(mrfc_lambda_distill)
        self.mrfc_margin = float(mrfc_margin)
        self.mrfc_tau_critic = float(mrfc_tau_critic)
        self.mrfc_critic_ensemble = None
        if self.mrfc_enabled:
            from .dllm.mrfc import FrozenMRFCCriticEnsemble
            self.mrfc_critic_ensemble = FrozenMRFCCriticEnsemble(
                str(mrfc_critic_path))
            print_log(
                'MR-FC: loaded {} frozen main critics from {} (sha256={})'
                .format(
                    len(self.mrfc_critic_ensemble.critics),
                    mrfc_critic_path,
                    self.mrfc_critic_ensemble.checkpoint_sha256,
                ))
        if mask_token_strategy not in ('legacy', 'canonical'):
            raise ValueError(
                "mask_token_strategy must be 'legacy' or 'canonical'")
        self.mask_token_strategy = mask_token_strategy
        self.enforce_nonempty_dllm_targets = bool(
            enforce_nonempty_dllm_targets)
        self.safe_dllm_loss = bool(safe_dllm_loss)
        self.mask_invalid_dllm_logits = bool(mask_invalid_dllm_logits)
        if state_conditioner_mode != 'additive_zero_init':
            raise ValueError(
                'state_conditioner_mode must be additive_zero_init'
            )
        self.state_conditioner_enabled = bool(state_conditioner)
        self.state_conditioner_fields = tuple(state_conditioner_fields)
        self.state_conditioner_mode = state_conditioner_mode
        self.clean_anchor_weight = float(clean_anchor_weight)
        if self.clean_anchor_weight < 0.0:
            raise ValueError('clean_anchor_weight must be nonnegative')
        self.dllm_state_conditioner = None
        if self.state_conditioner_enabled:
            self.dllm_state_conditioner = CorruptionStateConditioner(
                hidden_size=int(self.llm.config.hidden_size),
                fields=self.state_conditioner_fields,
            )

        # GUARD lagged conditional rank field (spec §5.6-5.7, Path C main
        # line).  A zero-output gate function makes the routed LoRA exactly
        # identical to static LoRA at initialization, while a small nonzero
        # bounded amplitude keeps its first-step gradient alive.  The
        # RiskHead is frozen and produces stop-gradient evidence g_hat.
        self.guard_enabled = bool(guard_enabled)
        self.guard_submodel = str(guard_submodel)
        # New experiments expose TRACER names. ``guard_*`` remains the
        # checkpoint-compatible internal namespace for historical weights.
        self.tracer_router_enabled = bool(tracer_router_enabled)
        self.tracer_policy_enabled = bool(tracer_policy_enabled)
        self.guard_context = None
        self._guard_lagged_evidence = None  # [batch, seq, 1] or None
        self._guard_prev_logits = None  # for JS divergence on next round
        self._guard_prev_candidates = None
        self._guard_committed_history = None  # [batch, seq] bool
        self._guard_risk_head = None
        self._guard_isotonic = None
        # §6.1 multi-round training: number of dLLM denoising rounds per
        # batch.  ``1`` = single forward (legacy S1 stable baseline, gate
        # stays at s=1 because lagged evidence is None on the first round).
        # ``2`` (recommended) = round k teacher forward produces lagged
        # ``g_hat`` that feeds round k+1 student forward whose RankGate
        # actually uses the lagged signal — this is the §5.6 training-time
        # activation of the lagged conditional rank field.  Cost = Kx
        # main-model forward (within §1's 1.5-2.0x budget for K=2).
        self.guard_train_rounds = int(guard_train_rounds)
        # Opt-in direct supervision of the original masked first round.
        # Zero preserves historical teacher-only training exactly.
        self.guard_base_loss_weight = float(guard_base_loss_weight)
        if not 0.0 <= self.guard_base_loss_weight <= 1.0:
            raise ValueError('guard_base_loss_weight must be in [0, 1]')
        if self.guard_base_loss_weight > 0.0:
            if not (self.guard_enabled and dllm and self.guard_train_rounds == 2):
                raise ValueError('Base supervision requires two-round GUARD dLLM')
            if gradient_checkpointing:
                raise ValueError('Base supervision requires gradient_checkpointing=False '
                                 'to preserve per-round routing context during backward')
        # §6.1 total loss: L = L_dLLM + λ_clean L_clean + λ_gate L_gate
        # + λ_risk L_risk.  The gate regularizer L_gate (route entropy +
        # effective rank bound) is computed on the round k+1 gate_scale
        # when guard_train_rounds >= 2.  Default 0 = no regularizer.
        self.guard_gate_regularizer_weight = float(
            guard_gate_regularizer_weight)
        # Cache for the most recent gate_scale tensor (used by L_gate).
        self._guard_last_gate_scale = None
        if self.guard_enabled:
            from src.models.dllm.guard import GuardContext
            self.guard_context = GuardContext()
            self._guard_load_risk_head(
                risk_head_path=guard_risk_head_path,
                hidden_size_hint=guard_risk_head_hidden,
                proj_hidden=guard_risk_head_proj_hidden,
                state_dim=guard_risk_head_state_dim,
            )

        image_token_id = -1
        canonical_mask_token_id = -1
        if self.tokenizer is not None:
            token_ids = register_harmon_tokens(self.tokenizer, self.llm)
            image_token_id = token_ids.image_token_id
            canonical_mask_token_id = token_ids.mask_token_id
        elif mask_token_strategy == 'canonical':
            raise ValueError(
                'canonical mask strategy requires a tokenizer')

        active_mask_token_id = (
            canonical_mask_token_id
            if mask_token_strategy == 'canonical'
            else int(legacy_mask_token_id)
        )
        self.register_buffer(
            'harmon_image_token_id',
            torch.tensor(image_token_id, dtype=torch.long),
        )
        self.register_buffer(
            'harmon_canonical_mask_token_id',
            torch.tensor(canonical_mask_token_id, dtype=torch.long),
        )
        self.register_buffer(
            'harmon_legacy_mask_token_id',
            torch.tensor(int(legacy_mask_token_id), dtype=torch.long),
        )
        self.register_buffer(
            'harmon_mask_token_id',
            torch.tensor(active_mask_token_id, dtype=torch.long),
        )
        self.dllm = dllm
        self.min_mask_rate = min_mask_rate
        self.max_mask_rate = max_mask_rate

        self.block_size = block_size
        self.prior_dist = prior_dist
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.debuged = False
        self._input_require_grads_enabled = False

        # NOTE: ``pretrained_pth`` is loaded *after* LoRA + Guard setup below,
        # not here.  S1 stable checkpoints are saved AFTER ``get_peft_model``
        # wraps the LLM, so their state_dict keys look like
        # ``llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.*``
        # (after ``guess_load_checkpoint`` strips the DDP ``module.`` prefix).
        # Loading them here, before ``_setup_lora``, would silently fail to
        # match the unwrapped LLM's parameter names and drop the LoRA weights.

        # Freeze specific modules
        if freeze_llm:
            self.llm.requires_grad_(False)
            print_log('Frozen LLM')
        
        if freeze_mar_encoder:
            # Freeze MAR encoder blocks and related components
            for param in self.mar.encoder_blocks.parameters():
                param.requires_grad = False
            self.mar.z_proj.requires_grad_(False)
            self.mar.z_proj_ln.requires_grad_(False)
            self.mar.encoder_pos_embed_learned.requires_grad = False
            self.mar.encoder_norm.requires_grad_(False)
            self.mar.class_emb.requires_grad_(False)
            self.mar.fake_latent.requires_grad = False
            print_log('Frozen MAR Encoder')
        
        if freeze_mar_decoder:
            # Freeze MAR decoder blocks and related components
            for param in self.mar.decoder_blocks.parameters():
                param.requires_grad = False
            self.mar.decoder_embed.requires_grad_(False)
            self.mar.mask_token.requires_grad = False
            self.mar.decoder_pos_embed_learned.requires_grad = False
            self.mar.decoder_norm.requires_grad_(False)
            self.mar.diffusion_pos_embed_learned.requires_grad = False
            self.mar.diffloss.requires_grad_(False)
            print_log('Frozen MAR Decoder')
        
        if freeze_proj_in:
            self.proj_in.requires_grad_(False)
            print_log('Frozen proj_in')
        
        if freeze_proj_out:
            self.proj_out.requires_grad_(False)
            print_log('Frozen proj_out')

        self.mask_token_delta = None
        if trainable_mask_delta:
            if self.mask_token_strategy != 'canonical':
                raise ValueError(
                    'trainable_mask_delta requires canonical strategy')
            self.mask_token_delta = TrainableMaskDelta.from_embeddings(
                self.llm.get_input_embeddings(),
                mask_token_id=int(self.harmon_mask_token_id.item()),
                legacy_mask_token_id=int(
                    self.harmon_legacy_mask_token_id.item()),
            )

        self.dllm_valid_logit_mask = None
        if self.mask_invalid_dllm_logits:
            if self.tokenizer is None:
                raise ValueError(
                    'mask_invalid_dllm_logits requires a tokenizer')
            valid_token_ids = harmon_token_support(
                self.tokenizer
            ).output_token_ids
            output_rows = int(
                self.llm.get_output_embeddings().weight.shape[0])
            self.dllm_valid_logit_mask = ValidLogitMask(
                vocab_size=output_rows,
                valid_token_ids=valid_token_ids,
                forbidden_token_ids=(
                    int(self.harmon_image_token_id.item()),
                    int(self.harmon_canonical_mask_token_id.item()),
                    int(self.harmon_legacy_mask_token_id.item()),
                ),
            )

        if lora is not None:
            self._setup_lora(lora)

        # MMEngine calls ``model.init_weights()`` after construction.  PEFT
        # forwards that call to the wrapped Hugging Face model, which resets
        # every LoRA adapter (and any RankGate children) after the hot-start
        # below.  Retain the audited load contract so ``init_weights`` can
        # restore the checkpoint after all framework initializers have run.
        self._pretrained_pth_after_init = pretrained_pth
        self._pretrained_required_key_groups_after_init = tuple(
            pretrained_required_key_groups)

        # Load pretrained weights AFTER LoRA + Guard setup so the state_dict
        # keys match: S1 checkpoints were saved with ``module.`` prefix (DDP)
        # + ``llm.base_model.model...`` (PEFT) + ``guard_rank_gate.*`` (Guard).
        # The loader below strips a retained DDP prefix and fails when zero
        # keys match. ``strict=False`` then tolerates (a) missing
        # ``guard_rank_gate.*`` keys when hot-starting
        # from a non-GUARD checkpoint (kept at zero-init so s=1 at step 0),
        # (b) ``vae.*`` keys filtered out by ``HarmonDev.state_dict``, and
        # (c) any other surplus keys in the checkpoint.
        if pretrained_pth is not None:
            _load_pretrained_weights(
                self,
                pretrained_pth,
                required_key_groups=pretrained_required_key_groups,
            )

        # gradient checkpointing
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def _setup_lora(self, lora):
        if not isinstance(lora, dict):
            raise TypeError(
                '`lora` must be a dict containing PEFT LoraConfig fields, '
                f'but got {type(lora).__name__}.')

        lora_config_dict = dict(lora)
        target_modules = lora_config_dict.get('target_modules')
        if not target_modules:
            raise ValueError(
                '`lora.target_modules` must explicitly select the Qwen '
                'modules that receive LoRA adapters.')

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise ImportError(
                'LoRA training requires PEFT. Install it in the training '
                'environment before setting `model.lora`.') from error

        lora_config = LoraConfig(**lora_config_dict)
        self.llm = get_peft_model(self.llm, lora_config)

        # GUARD: install RankGate on every LoRA layer (spec §5.6).
        # At init the gate produces s=1 (static baseline), so enabling this
        # alone does not change the loss curve.
        if self.guard_enabled:
            from src.models.dllm.guard import GuardedLoRAManager
            lora_rank = int(lora_config_dict.get('r', 16))
            self.guard_lora_manager = GuardedLoRAManager(
                self.llm,
                rank=lora_rank,
                submodel=self.guard_submodel,
            )
            for _name, _gate in self.guard_lora_manager.gates.items():
                _gate = _gate.to(self.llm.device)
                for _p in _gate.parameters():
                    _p.requires_grad = True
            gate_params = sum(
                p.numel() for p in self.guard_lora_manager.gate_parameters())
            print_log(
                f'GUARD: submodel={self.guard_submodel}, '
                f'{len(self.guard_lora_manager.gates)} LoRA layers, '
                f'{gate_params} gate params.')

        trainable_params = sum(
            parameter.numel()
            for parameter in self.llm.parameters()
            if parameter.requires_grad)
        total_params = sum(
            parameter.numel() for parameter in self.llm.parameters())
        print_log(
            f'Enabled LoRA: {trainable_params:,} trainable LLM parameters '
            f'out of {total_params:,}.')

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if self._input_require_grads_enabled:
            self.llm.disable_input_require_grads()
            self._input_require_grads_enabled = False
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if (hasattr(self.llm, 'peft_config')
                and not self._input_require_grads_enabled):
            self.llm.enable_input_require_grads()
            self._input_require_grads_enabled = True
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if ('vae.' not in k
                          and not k.startswith((
                              'mrfc_critic_ensemble.',
                              'module.mrfc_critic_ensemble.',
                          )))}

        return state_dict

    def init_weights(self):
        """Run framework initialization, then restore audited hot-starts."""
        super().init_weights()
        # Hugging Face initializers also visit the Linear layers nested in
        # RankGate and overwrite their identity-preserving zero output layer.
        # Restore the gradient-live identity state first; a future checkpoint
        # containing trained gates is loaded immediately afterwards and wins.
        guard_manager = getattr(self, 'guard_lora_manager', None)
        if guard_manager is not None:
            for gate in guard_manager.gates.values():
                gate.reset_identity_parameters()
        pretrained_pth = getattr(
            self, '_pretrained_pth_after_init', None)
        if pretrained_pth is not None:
            _load_pretrained_weights(
                self,
                pretrained_pth,
                required_key_groups=getattr(
                    self,
                    '_pretrained_required_key_groups_after_init',
                    (),
                ),
            )

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        x_enc = self.forward_mae_encoder(x, mask, input_ids=input_ids,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def _guard_load_risk_head(
        self,
        *,
        risk_head_path: Optional[str],
        hidden_size_hint: Optional[int],
        proj_hidden: int,
        state_dim: int,
    ) -> None:
        """Load the frozen RiskHead (§5.4'.2) used for trajectory evidence.

        The risk head is produced by ``scripts/train_risk_head.py`` on the
        risk-fit split and is frozen during GUARD training.  It is used to
        compute the trajectory evidence scalar ``g_hat_i^k`` from the
        §5.4'.1 trajectory features at each round.
        """
        if risk_head_path is None:
            # RiskHead not provided; the lagged evidence will be ``None``
            # (treated as zero by the RankGate, i.e. neutral gate).
            return
        import os
        if not os.path.isfile(risk_head_path):
            raise FileNotFoundError(
                f'guard_risk_head_path not found: {risk_head_path}'
            )
        from src.models.dllm.guard import RiskHead, IsotonicCalibrator
        ckpt = torch.load(risk_head_path, map_location='cpu')
        hidden_size = int(
            ckpt.get('hidden_size', hidden_size_hint or self.llm.config.hidden_size)
        )
        head = RiskHead(
            hidden_size=hidden_size,
            state_dim=int(ckpt.get('state_dim', state_dim)),
            proj_hidden=int(ckpt.get('proj_hidden', proj_hidden)),
        )
        missing, unexpected = head.load_state_dict(
            ckpt['state_dict'], strict=False)
        if missing:
            # Only the temperature buffer may be missing on legacy checkpoints.
            head.temperature.fill_(float(ckpt.get('temperature', 1.0)))
        head = head.to(self.device).eval()
        for p in head.parameters():
            p.requires_grad = False
        self._guard_risk_head = head
        # Load isotonic calibrator if present (post-hoc calibration, §5.5').
        if ckpt.get('iso_xs') is not None and ckpt.get('iso_ys') is not None:
            iso = IsotonicCalibrator()
            iso._xs = ckpt['iso_xs'].cpu()
            iso._ys = ckpt['iso_ys'].cpu()
            self._guard_isotonic = iso
        print_log(
            f'GUARD: loaded frozen RiskHead from {risk_head_path} '
            f'(hidden={hidden_size}, T={head.temperature.item():.4f}, '
            f'iso={self._guard_isotonic is not None})'
        )

    @torch.no_grad()
    def _guard_compute_trajectory_features(
        self,
        output,
        state: torch.Tensor,
        response_mask: torch.Tensor,
    ):
        """Compute the §5.4'.1 trajectory features from an LLM forward.

        Parameters
        ----------
        output:
            LLM forward output (``last_hidden_state`` on the response).
        state:
            Corruption state ``(t, u, h)`` for this round, shape
            ``[batch, 3]``.
        response_mask:
            Boolean mask of active response positions, shape
            ``[batch, seq]``.

        Returns
        -------
        TrajectoryFeatures
        """
        from src.models.dllm.guard.risk_control import (
            compute_trajectory_features,
        )
        last_hidden = output.last_hidden_state  # [batch, seq, hidden]
        # Build logits for the active response positions.
        logits = self.llm.get_output_embeddings()(last_hidden)
        if self.dllm_valid_logit_mask is not None:
            logits = self.dllm_valid_logit_mask(logits)
        # Only score positions that are part of the response (active mask).
        # We still compute features everywhere; the manager slices evidence
        # to the LoRA input window.  The committed_history is maintained
        # across rounds by the caller via ``self._guard_committed_history``.
        committed_history = self._guard_committed_history
        if committed_history is None or committed_history.shape != (
            last_hidden.shape[0], last_hidden.shape[1]
        ):
            committed_history = torch.zeros(
                last_hidden.shape[0], last_hidden.shape[1],
                dtype=torch.bool, device=last_hidden.device,
            )
        remask = torch.zeros_like(committed_history)
        features = compute_trajectory_features(
            hidden=last_hidden,
            logits=logits,
            prev_probs=self._guard_prev_logits,
            prev_candidates=self._guard_prev_candidates,
            committed_history=committed_history,
            remask=remask,
            state=state,
            block_size=self.block_size,
        )
        # Cache prev_* for the next round's JS / stability computation.
        self._guard_prev_logits = F.softmax(logits.float(), dim=-1).detach()
        self._guard_prev_candidates = features.candidates.detach()
        return features

    @torch.no_grad()
    def _guard_update_lagged_evidence(self, output, state, response_mask=None):
        """Update lagged trajectory evidence from LLM forward output.

        Spec §5.4'.2 / §5.6: the frozen RiskHead consumes the §5.4'.1
        trajectory features at round ``k`` and produces the scalar
        ``g_hat_i^k`` (shape ``[batch, seq, 1]``), stop-gradiented before it
        enters the round ``k+1`` RankGate.
        """
        if self._guard_risk_head is None:
            self._guard_lagged_evidence = None
            return
        features = self._guard_compute_trajectory_features(
            output, state, response_mask
        )
        g_hat = self._guard_risk_head(
            features.hidden,
            features.confidence,
            features.entropy,
            features.js_div,
            features.stable,
            features.committed_history.float(),
            features.remask.float(),
            features.block_commit_corr,
            features.state,
        )  # [batch, seq, 1]
        if self._guard_isotonic is not None:
            # Apply post-hoc isotonic calibration (§5.5').
            g_hat = self._guard_isotonic.transform(g_hat).view_as(g_hat)
        self._guard_lagged_evidence = g_hat.detach()

    def _guard_gate_regularizer(self) -> torch.Tensor:
        """Live second-round penalty, zero at uniform routing.

        Detached diagnostic caches cannot train the gate. The live cache is
        consumed BEFORE manager.reset(); only the scalar loss survives reset.
        """
        mgr = getattr(self, 'guard_lora_manager', None)
        scales = getattr(mgr, 'regularizer_scales', {})
        if not scales:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        terms, counts = [], []
        for scale in scales.values():
            energy = scale.float().square()
            p = energy / energy.sum(-1, keepdim=True).clamp_min(1e-8)
            rank = p.shape[-1]
            entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
            # amax distributes gradients at ties; max would select one atom.
            penalty = (1 - entropy / math.log(rank) + p.amax(-1) - 1 / rank
                       if rank > 1 else p.sum(-1) * 0)
            terms.append(penalty.sum())
            counts.append(penalty.numel())
        return torch.stack(terms).sum() / sum(counts)

    def dllm_token_loss(
        self,
        last_hidden_state,
        labels,
        loss_mask,
        response_mask,
        sample_weights=None,
    ):
        active_mask = loss_mask & response_mask & labels.ne(-100)
        selected_hidden = last_hidden_state[active_mask]
        selected_labels = labels[active_mask]
        logits = self.llm.get_output_embeddings()(selected_hidden)
        if self.dllm_valid_logit_mask is not None:
            logits = self.dllm_valid_logit_mask(logits)
        if sample_weights is not None:
            if (not torch.is_tensor(sample_weights)
                    or sample_weights.ndim != 1
                    or sample_weights.shape[0] != labels.shape[0]):
                raise ValueError(
                    'visnec_weight must align with the image2text batch')
            sample_weights = sample_weights.to(
                device=last_hidden_state.device, dtype=torch.float32)
            if (not torch.isfinite(sample_weights).all()
                    or bool((sample_weights < 0.5).any())
                    or bool((sample_weights > 1.5).any())):
                raise ValueError(
                    'visnec_weight must be finite in [0.5, 1.5]')
            token_losses = F.cross_entropy(
                logits.float(), selected_labels, reduction='none')
            active_rows = active_mask.nonzero(as_tuple=False)[:, 0]
            sums = token_losses.new_zeros(labels.shape[0])
            counts = token_losses.new_zeros(labels.shape[0])
            sums.scatter_add_(0, active_rows, token_losses)
            counts.scatter_add_(
                0, active_rows, torch.ones_like(token_losses))
            valid_samples = counts.gt(0)
            if not bool(valid_samples.any()):
                return logits.sum() * 0.0
            per_sample = sums[valid_samples] / counts[valid_samples]
            weights = sample_weights[valid_samples]
            return (per_sample * weights).sum() / weights.sum()
        if self.safe_dllm_loss:
            return safe_token_cross_entropy(logits, selected_labels)
        return F.cross_entropy(input=logits, target=selected_labels)

    def _image2text_loss_result(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        # 获取 dLLM 特有的 masks (如果存在)
        response_mask = data_dict.get('response_mask', None)
        if response_mask is not None:
            response_mask = response_mask.to(self.device)
            
        loss_mask = data_dict.get('loss_mask', None)
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        # 【新增】：取出 create_dllm_batch 构造的 position_ids（仅 dllm 分支存在）
        position_ids = data_dict.get('position_ids', None)
        if position_ids is not None:
            position_ids = position_ids.to(self.device)

        block_indices = data_dict.get('block_indices', None)
        if block_indices is not None:
            block_indices = block_indices.to(self.device)

        state_features = data_dict.get('corruption_state_features')
        state_mask = data_dict.get('corruption_state_mask')
        if self.dllm_state_conditioner is not None and self.dllm:
            if state_features is None or state_mask is None:
                raise ValueError(
                    'enabled state conditioner requires packed '
                    'CorruptionState features and mask'
                )
            state_features = state_features.to(
                device=self.device, dtype=torch.float32
            )
            state_mask = state_mask.to(
                device=self.device, dtype=torch.bool
            )

        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            assert False
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            for i in range(input_ids.shape[0]):
                assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
            input_ids = torch.cat(
                (
                    input_ids[:, :3],
                    IMAGE_TOKEN_INDEX * torch.ones(
                        input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                    input_ids[:, 4:],
                ), dim=1
            )
            if attention_mask.dim() == 4:
                # 4D Mask: [bsz, 1, q_len, k_len]
                # 1. 沿 Query 维度 (dim=2) 复制图像占位符的可见性
                attention_mask = torch.cat([
                    attention_mask[:, :, :3, :],
                    attention_mask[:, :, 3:4, :].repeat(1, 1, 1088, 1),
                    attention_mask[:, :, 4:, :]
                ], dim=2)
                
                # 2. 沿 Key 维度 (dim=3) 复制图像占位符的可见性
                attention_mask = torch.cat([
                    attention_mask[:, :, :, :3],
                    attention_mask[:, :, :, 3:4].repeat(1, 1, 1, 1088),
                    attention_mask[:, :, :, 4:]
                ], dim=3)
            else:
                # 兼容传统的 2D Mask: [bsz, seq_len]
                attention_mask = torch.cat(
                    (
                        attention_mask[:, :3],
                        torch.ones(
                            attention_mask.shape[0], 1088, dtype=attention_mask.dtype, device=attention_mask.device),
                        attention_mask[:, 4:],
                    ), dim=1
                )
            
            labels = torch.cat(
                (
                    labels[:, :3],
                    labels[0, 3] * torch.ones(
                        labels.shape[0], 1088 , dtype=torch.long, device=labels.device),
                    labels[:, 4:],
                ), dim=1
            )
            if response_mask is not None:
                response_mask = torch.cat(
                    (
                        response_mask[:, :3],
                        torch.zeros(response_mask.shape[0], 1088, dtype=torch.bool, device=response_mask.device),
                        response_mask[:, 4:],
                    ), dim=1
                )
                # 存回字典，供后续提取 Logits 使用
                data_dict['response_mask'] = response_mask
                
            if loss_mask is not None:
                loss_mask = torch.cat(
                    (
                        loss_mask[:, :3],
                        torch.zeros(loss_mask.shape[0], 1088, dtype=torch.bool, device=loss_mask.device),
                        loss_mask[:, 4:],
                    ), dim=1
                )
                data_dict['loss_mask'] = loss_mask

            # 【新增】：图像占位 1→1088 同步扩展 position_ids，保持与 input_ids 对齐，
            # 且重新映射到“im_start user...IMG_1...IMG_1088...”的康康位置，保证 RoPE 与纯 AR 一致。
            if position_ids is not None:
                base_pos_at_img = position_ids[:, 3:4]  # [b, 1]
                img_offsets = torch.arange(1088, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(0)
                image_pos_ids = base_pos_at_img + img_offsets  # [b, 1088]
                after_pos = position_ids[:, 4:] + 1087
                position_ids = torch.cat([position_ids[:, :3], image_pos_ids, after_pos], dim=1)

            if state_features is not None:
                state_features = torch.cat(
                    (
                        state_features[:, :3],
                        torch.zeros(
                            state_features.shape[0],
                            1088,
                            state_features.shape[-1],
                            dtype=state_features.dtype,
                            device=state_features.device,
                        ),
                        state_features[:, 4:],
                    ),
                    dim=1,
                )
                state_mask = torch.cat(
                    (
                        state_mask[:, :3],
                        torch.zeros(
                            state_mask.shape[0],
                            1088,
                            dtype=torch.bool,
                            device=state_mask.device,
                        ),
                        state_mask[:, 4:],
                    ),
                    dim=1,
                )

            if block_indices is not None:
                block_indices = torch.cat(
                    (
                        block_indices[:, :3],
                        torch.zeros(
                            block_indices.shape[0],
                            1088,
                            dtype=block_indices.dtype,
                            device=block_indices.device,
                        ),
                        block_indices[:, 4:],
                    ),
                    dim=1,
                )

            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            if self.mask_token_delta is not None:
                inputs_embeds = self.mask_token_delta(
                    inputs_embeds,
                    input_ids,
                )
            if self.dllm_state_conditioner is not None:
                inputs_embeds = self.dllm_state_conditioner(
                    inputs_embeds,
                    state_features,
                    state_mask,
                )
            loss_null = 0.0

        if getattr(self, "dllm", False) and attention_mask.dtype == torch.bool:
            float_mask = torch.zeros_like(attention_mask, dtype=inputs_embeds.dtype)
            float_mask = float_mask.masked_fill(~attention_mask, torch.finfo(inputs_embeds.dtype).min)
            attention_mask = float_mask

        # 【修改】：dllm 分支显式传 position_ids，否则 HF 会自动 arange 导致 clean/noisy 段数据分布不对齐。
        llm_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if getattr(self, "dllm", False) and position_ids is not None:
            llm_kwargs['position_ids'] = position_ids

        # GUARD §6.1 multi-round training forward path.
        #
        # ``guard_train_rounds`` selects the cost/innovation trade-off:
        #   * ``1`` = legacy single forward.  Lagged evidence is ``None``
        #     on the first round, so RankGate returns ``s=1`` explicitly and
        #     receives no routing gradient. This is the S1 stable fallback.
        #   * ``>=2`` = §6.1 multi-round training.  Round k (teacher)
        #     produces ``g_hat_i^k`` from frozen RiskHead; round k+1
        #     (student) consumes the lagged evidence, so RankGate uses
        #     ``s = 1 + beta*tanh(f_l(state_{k+1}, g_hat_i^k))`` and
        #     ``raw_beta`` gets a non-trivial gradient.  Cost = Kx
        #     main-model forward (within §1's 1.5-2.0x budget for K=2).
        #
        # Per-batch lagged caches are reset at the start to avoid
        # cross-sample leakage (spec §12.1 "lagged context 对错样本").
        guard_mgr = getattr(self, 'guard_lora_manager', None)
        guard_multi = (
            guard_mgr is not None
            and self.guard_train_rounds >= 2
            and self.dllm
        )
        base_weight = getattr(self, 'guard_base_loss_weight', 0.0)
        original_mask_loss = None
        # Original (round-k) corruption state built from the batch.
        if guard_mgr is not None:
            # Reset per-batch lagged caches to avoid cross-sample leakage.
            self._guard_lagged_evidence = None
            self._guard_prev_logits = None
            self._guard_prev_candidates = None
            self._guard_committed_history = None
            if state_features is not None and state_mask is not None:
                guard_state = state_features.to(
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                mask_f = state_mask.to(
                    device=inputs_embeds.device, dtype=guard_state.dtype
                ).unsqueeze(-1)
                denom = mask_f.sum(dim=1).clamp(min=1.0)
                guard_state_k = (guard_state * mask_f).sum(dim=1) / denom
            else:
                guard_state_k = torch.zeros(
                    inputs_embeds.shape[0], 3,
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype,
                )
        else:
            guard_state_k = None

        if not guard_multi:
            # Legacy single-forward path (S1 stable baseline).  RankGate
            # runs with ``evidence=None`` (zero evidence -> s=1).
            if guard_mgr is not None:
                guard_mgr.set_round_context(
                    state=guard_state_k,
                    evidence=None,
                    routing_enabled=self.tracer_router_enabled,
                )
            output = self.llm_model(**llm_kwargs)
            if guard_mgr is not None:
                guard_mgr.reset()
        else:
            # §6.1 multi-round training.  Round k = teacher forward that
            # produces lagged evidence; round k+1 = student forward that
            # consumes it and produces the dLLM loss.
            #
            # Round k uses the existing ``inputs_embeds`` (already built
            # from the batch's corruption state) with ``evidence=None``
            # (first round, neutral gate, s=1, §5.6). The opt-in base
            # objective supervises the ORIGINAL mask before any commits.
            guard_mgr.set_round_context(
                state=guard_state_k,
                evidence=None,
                routing_enabled=self.tracer_router_enabled,
            )
            with torch.set_grad_enabled(torch.is_grad_enabled() and base_weight > 0.0):
                output_k = self.llm_model(**llm_kwargs)
            guard_mgr.reset()
            if base_weight > 0.0:
                if loss_mask is None:
                    raise ValueError('Base supervision requires the original loss_mask')
                original_mask_loss = self.dllm_token_loss(
                    output_k.last_hidden_state, labels, loss_mask,
                    response_mask, data_dict.get('visnec_weight'))
            # Build per-token lagged evidence ``g_hat_i^k`` from round k.
            # This calls the frozen RiskHead under no_grad, so both the
            # LLM hidden states and RiskHead weights are detached.
            with torch.no_grad():
                resp_mask_k = None
                if response_mask is not None:
                    resp_mask_k = response_mask.to(
                        device=inputs_embeds.device, dtype=torch.bool)
                self._guard_update_lagged_evidence(
                    output_k, guard_state_k, response_mask=resp_mask_k)
                # Snapshot committed_history so round k+1 sees the
                # actual commit history from round k (all-False on round
                # 0 since the teacher path doesn't commit).
                committed_k = self._guard_committed_history
                if committed_k is None:
                    committed_k = torch.zeros(
                        inputs_embeds.shape[0], inputs_embeds.shape[1],
                        dtype=torch.bool,
                        device=inputs_embeds.device,
                    )
                    self._guard_committed_history = committed_k
            # Round k+1 uses a real adjacent token state: each response block
            # commits its fixed-schedule fraction of the teacher's most
            # confident candidates. Reusing the identical masked embeddings
            # while only changing (t,u,h) would be a train/inference mismatch.
            if response_mask is None or block_indices is None:
                raise ValueError(
                    'TRACER multi-round training requires response_mask and '
                    'block_indices')
            if self.dllm_state_conditioner is not None:
                raise ValueError(
                    'TRACER paired-token training currently requires the '
                    'state conditioner to be disabled')
            from src.models.dllm.guard import (
                advance_corruption_state,
                advance_masked_tokens,
            )
            # Teacher candidates and discrete commits never receive gradients.
            # In particular, do not retain the full-vocabulary logits graph
            # when direct first-round supervision is enabled.
            with torch.no_grad():
                logits_k = self.llm.get_output_embeddings()(
                    output_k.last_hidden_state)
                if self.dllm_valid_logit_mask is not None:
                    logits_k = self.dllm_valid_logit_mask(logits_k)
                student_ids, teacher_commit_mask = advance_masked_tokens(
                    input_ids=input_ids,
                    logits=logits_k,
                    response_mask=response_mask,
                    block_indices=block_indices,
                    mask_token_id=int(self.harmon_mask_token_id.item()),
                    committed_fraction=1.0 / max(int(self.block_size), 1),
                )
            del logits_k, output_k
            student_inputs_embeds = inputs_embeds.clone()
            if bool(teacher_commit_mask.any()):
                committed_embeddings = self.llm.get_input_embeddings()(
                    student_ids[teacher_commit_mask])
                student_inputs_embeds[teacher_commit_mask] = (
                    committed_embeddings.to(student_inputs_embeds.dtype))
                # A committed token is now visible in the student input and
                # is no longer a masked-denoising target at round k+1.
                # Keeping it in the loss would leak the target token through
                # the bidirectional student representation.
                if loss_mask is not None:
                    loss_mask = loss_mask & ~teacher_commit_mask
            llm_kwargs_student = dict(llm_kwargs)
            llm_kwargs_student['inputs_embeds'] = student_inputs_embeds

            guard_state_kp1 = advance_corruption_state(
                guard_state_k,
                committed_fraction=1.0 / max(int(self.block_size), 1),
            )
            guard_mgr.set_round_context(
                state=guard_state_kp1,
                evidence=self._guard_lagged_evidence,
                routing_enabled=self.tracer_router_enabled,
                capture_regularizer=self.guard_gate_regularizer_weight > 0.0,
            )
            try:
                output = self.llm_model(**llm_kwargs_student)
                if self.guard_gate_regularizer_weight > 0.0:
                    loss_gate = self._guard_gate_regularizer()
            finally:
                guard_mgr.reset()

        if getattr(self, "dllm", False):
            last_hidden_state = output.last_hidden_state#[:, :-1]
            loss_i2t = self.dllm_token_loss(
                last_hidden_state,
                labels,
                loss_mask,
                response_mask,
                data_dict.get('visnec_weight'),
            )
            if original_mask_loss is not None:
                loss_i2t = (base_weight * original_mask_loss
                            + (1.0 - base_weight) * loss_i2t)
        else:
            last_hidden_state = output.last_hidden_state[:, :-1]
            labels = labels[:, 1:]
            last_hidden_state = last_hidden_state[labels >= 0]
            labels = labels[labels >= 0]
            logits = self.llm.get_output_embeddings()(last_hidden_state)

            loss_i2t = F.cross_entropy(input=logits, target=labels)

        # §6.1 total loss: L = L_dLLM + λ_gate L_gate.
        # L_gate is only added when the gate actually routed (multi-round
        # path).  λ_gate = self.guard_gate_regularizer_weight (default 0).
        if guard_multi and self.guard_gate_regularizer_weight > 0.0:
            loss_i2t = loss_i2t + self.guard_gate_regularizer_weight * loss_gate

        # Optional online-commit extension. Historical models never enter this
        # branch. Pass the *expanded* layout and already encoded image; the
        # extension must not reconstruct RoPE positions or expose clean targets.
        commit_forward = getattr(self, '_commit_training_forward', None)
        if commit_forward is not None:
            commit_forward(
                inputs_embeds=inputs_embeds, input_ids=input_ids,
                attention_mask=attention_mask, position_ids=position_ids,
                labels=labels, response_mask=response_mask,
                block_indices=block_indices,
            )

        return _Image2TextLossResult(
            base_loss=loss_i2t + loss_null,
            last_hidden_state=last_hidden_state,
            response_mask=response_mask,
            loss_mask=loss_mask,
        )

    def image2text_loss(self, data_dict):
        """Return the historical scalar image-to-text loss."""
        return self._image2text_loss_result(data_dict).base_loss

    @staticmethod
    def _mrare_diagnostic(reference, value):
        return reference.detach().new_tensor(float(value), dtype=torch.float32)

    def _mrare_zero_auxiliary(
        self,
        reference,
        diagnostics=None,
    ):
        diagnostics = dict(diagnostics or {})
        connected_zero = reference.sum() * 0.0
        result = {
            'weighted_positive': connected_zero,
            'weighted_rank': connected_zero,
            'weighted_distill': connected_zero,
        }
        for name in self._mrare_diagnostic_names():
            result[name] = self._mrare_diagnostic(
                reference, diagnostics.get(name, 0))
        return result

    @staticmethod
    def _mrare_diagnostic_names():
        return (
            'mrare_valid_count',
            'mrare_invalid_count',
            'mrare_empty_relevance_count',
            'mrare_invalid_vocab_count',
            'mrare_invalid_condition_count',
            'mrare_invalid_energy_count',
            'mrare_nonfinite_target_count',
            'mrare_distill_active_count',
            'mrare_distill_inactive_count',
        )

    def _mrare_auxiliary_losses(
        self,
        data_dict,
        last_hidden_state,
        response_mask,
        loss_mask,
    ):
        """Compute answer-conditioned paired representation losses.

        Structural batch violations fail with their field name. Invalid cache
        records and inactive distillation positions are excluded with detached
        reason diagnostics and backward-connected finite zeros.
        """
        from .dllm.mrare import mrare_losses, paired_reverse_energy
        required_keys = (
            'mrare_valid',
            'mrare_target_pixel_values',
            'mrare_relevance_mask',
            'mrare_positive_condition_ids',
            'mrare_positive_condition_mask',
            'mrare_negative_condition_ids',
            'mrare_negative_condition_mask',
            'mrare_changed_response_offset',
            'mrare_changed_condition_offset',
            'mrare_positive_token_id',
            'mrare_negative_token_id',
            'mrare_energy_scale',
        )
        if last_hidden_state.ndim != 3:
            raise ValueError(
                'last_hidden_state must have shape [batch, sequence, hidden]')
        batch_size = int(last_hidden_state.shape[0])
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f'MR-ARE batch is missing required field {key}')

        def _require_tensor(key, ndim):
            value = data_dict[key]
            if not torch.is_tensor(value):
                raise ValueError(f'{key} must be a tensor')
            if value.ndim != ndim:
                raise ValueError(f'{key} must have {ndim} dimensions')
            if value.shape[0] != batch_size:
                raise ValueError(f'{key} must align with batch size')
            return value.to(device=last_hidden_state.device)

        valid = _require_tensor('mrare_valid', 1)
        if valid.dtype != torch.bool:
            raise ValueError('mrare_valid must be a boolean tensor')
        target_pixels = _require_tensor('mrare_target_pixel_values', 4).to(
            dtype=self.dtype)
        relevance = _require_tensor('mrare_relevance_mask', 3)
        if relevance.dtype != torch.bool:
            raise ValueError('mrare_relevance_mask must be a boolean tensor')
        integer_dtypes = {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }
        positive_ids = _require_tensor('mrare_positive_condition_ids', 2)
        if positive_ids.dtype not in integer_dtypes:
            raise ValueError(
                'mrare_positive_condition_ids must be an integer tensor')
        positive_ids = positive_ids.to(dtype=torch.long)
        positive_attention = _require_tensor(
            'mrare_positive_condition_mask', 2)
        if positive_attention.dtype != torch.bool:
            raise ValueError(
                'mrare_positive_condition_mask must be a boolean tensor')
        negative_ids = _require_tensor('mrare_negative_condition_ids', 2)
        if negative_ids.dtype not in integer_dtypes:
            raise ValueError(
                'mrare_negative_condition_ids must be an integer tensor')
        negative_ids = negative_ids.to(dtype=torch.long)
        negative_attention = _require_tensor(
            'mrare_negative_condition_mask', 2)
        if negative_attention.dtype != torch.bool:
            raise ValueError(
                'mrare_negative_condition_mask must be a boolean tensor')
        if positive_attention.shape != positive_ids.shape:
            raise ValueError(
                'mrare_positive_condition_mask must match '
                'mrare_positive_condition_ids shape')
        if negative_attention.shape != negative_ids.shape:
            raise ValueError(
                'mrare_negative_condition_mask must match '
                'mrare_negative_condition_ids shape')
        def _require_integer_vector(key):
            value = _require_tensor(key, 1)
            if value.dtype not in integer_dtypes:
                raise ValueError(f'{key} must be an integer tensor')
            return value.to(dtype=torch.long)

        changed_response_offsets = _require_integer_vector(
            'mrare_changed_response_offset')
        changed_condition_offsets = _require_integer_vector(
            'mrare_changed_condition_offset')
        positive_token_ids = _require_integer_vector(
            'mrare_positive_token_id')
        negative_token_ids = _require_integer_vector(
            'mrare_negative_token_id')
        energy_scale = _require_tensor(
            'mrare_energy_scale', 1).to(dtype=torch.float32)

        if self.mrare_distill_enabled:
            expected_mask_shape = last_hidden_state.shape[:2]
            if not torch.is_tensor(response_mask):
                raise ValueError('response_mask must be a tensor')
            if response_mask.shape != expected_mask_shape:
                raise ValueError(
                    'response_mask must match last_hidden_state sequence shape')
            if response_mask.dtype != torch.bool:
                raise ValueError('response_mask must be a boolean tensor')
            if not torch.is_tensor(loss_mask):
                raise ValueError('loss_mask must be a tensor')
            if loss_mask.shape != expected_mask_shape:
                raise ValueError(
                    'loss_mask must match last_hidden_state sequence shape')
            if loss_mask.dtype != torch.bool:
                raise ValueError('loss_mask must be a boolean tensor')
            response_mask = response_mask.to(
                device=last_hidden_state.device, dtype=torch.bool)
            loss_mask = loss_mask.to(
                device=last_hidden_state.device, dtype=torch.bool)

        valid_indices = valid.nonzero(as_tuple=False).flatten()
        diagnostics = {
            'mrare_invalid_count': batch_size - int(valid_indices.numel()),
        }
        if valid_indices.numel() == 0:
            return self._mrare_zero_auxiliary(
                last_hidden_state, diagnostics)

        target_pixels = target_pixels.index_select(0, valid_indices)
        relevance = relevance.index_select(0, valid_indices)
        positive_ids = positive_ids.index_select(0, valid_indices)
        positive_attention = positive_attention.index_select(0, valid_indices)
        negative_ids = negative_ids.index_select(0, valid_indices)
        negative_attention = negative_attention.index_select(0, valid_indices)
        changed_response_offsets = changed_response_offsets.index_select(
            0, valid_indices)
        changed_condition_offsets = changed_condition_offsets.index_select(
            0, valid_indices)
        positive_token_ids = positive_token_ids.index_select(0, valid_indices)
        negative_token_ids = negative_token_ids.index_select(0, valid_indices)
        energy_scale = energy_scale.index_select(0, valid_indices)

        positive_lengths = positive_attention.sum(dim=1)
        negative_lengths = negative_attention.sum(dim=1)
        positive_expected_mask = (
            torch.arange(positive_ids.shape[1], device=self.device)
            .unsqueeze(0) < positive_lengths.unsqueeze(1))
        negative_expected_mask = (
            torch.arange(negative_ids.shape[1], device=self.device)
            .unsqueeze(0) < negative_lengths.unsqueeze(1))
        condition_offsets_valid = (
            changed_condition_offsets.ge(0)
            & changed_condition_offsets.lt(positive_ids.shape[1])
            & changed_condition_offsets.lt(negative_ids.shape[1])
        )
        condition_width = min(
            positive_ids.shape[1], negative_ids.shape[1])
        if condition_width > 0:
            safe_condition_offsets = changed_condition_offsets.clamp(
                min=0, max=condition_width - 1)
            row_indices = torch.arange(
                valid_indices.numel(), device=self.device)
            condition_offsets_valid &= (
                positive_attention[row_indices, safe_condition_offsets]
                & negative_attention[row_indices, safe_condition_offsets]
                & positive_ids[row_indices, safe_condition_offsets].eq(
                    positive_token_ids)
                & negative_ids[row_indices, safe_condition_offsets].eq(
                    negative_token_ids)
            )
        condition_valid = (
            positive_lengths.gt(0)
            & negative_lengths.gt(0)
            & positive_attention.eq(positive_expected_mask).all(dim=1)
            & negative_attention.eq(negative_expected_mask).all(dim=1)
            & condition_offsets_valid
        )

        input_rows = int(self.llm.get_input_embeddings().weight.shape[0])
        output_rows = int(self.llm.get_output_embeddings().weight.shape[0])
        positive_vocab_valid = (
            ((positive_ids >= 0) & (positive_ids < input_rows))
            | ~positive_attention
        ).all(dim=1)
        negative_vocab_valid = (
            ((negative_ids >= 0) & (negative_ids < input_rows))
            | ~negative_attention
        ).all(dim=1)
        vocab_valid = (
            positive_vocab_valid
            & negative_vocab_valid
            & positive_token_ids.ge(0)
            & positive_token_ids.lt(output_rows)
            & negative_token_ids.ge(0)
            & negative_token_ids.lt(output_rows)
        )
        relevance_nonempty = relevance.flatten(1).any(dim=1)
        if bool((~relevance_nonempty).any()):
            raise ValueError(
                'mrare_relevance_mask must be nonempty for every valid record')
        scale_valid = torch.isfinite(energy_scale) & energy_scale.gt(0)
        target_finite = torch.isfinite(target_pixels).flatten(1).all(dim=1)
        usable = (
            condition_valid
            & vocab_valid
            & relevance_nonempty
            & scale_valid
            & target_finite
        )
        diagnostics.update({
            'mrare_empty_relevance_count': int(
                (~relevance_nonempty).sum().item()),
            'mrare_invalid_vocab_count': int((~vocab_valid).sum().item()),
            'mrare_invalid_condition_count': int(
                (~condition_valid).sum().item()),
            'mrare_invalid_energy_count': int((~scale_valid).sum().item()),
            'mrare_nonfinite_target_count': int(
                (~target_finite).sum().item()),
        })
        rejected_count = int((~usable).sum().item())
        diagnostics['mrare_invalid_count'] += rejected_count
        usable_indices = usable.nonzero(as_tuple=False).flatten()
        if usable_indices.numel() == 0:
            return self._mrare_zero_auxiliary(
                last_hidden_state, diagnostics)

        usable_batch_indices = valid_indices.index_select(0, usable_indices)
        target_pixels = target_pixels.index_select(0, usable_indices)
        relevance = relevance.index_select(0, usable_indices)
        positive_ids = positive_ids.index_select(0, usable_indices)
        positive_attention = positive_attention.index_select(0, usable_indices)
        negative_ids = negative_ids.index_select(0, usable_indices)
        negative_attention = negative_attention.index_select(0, usable_indices)
        changed_response_offsets = changed_response_offsets.index_select(
            0, usable_indices)
        positive_token_ids = positive_token_ids.index_select(0, usable_indices)
        negative_token_ids = negative_token_ids.index_select(0, usable_indices)
        energy_scale = energy_scale.index_select(0, usable_indices)
        positive_ids = positive_ids.masked_fill(~positive_attention, 0)
        negative_ids = negative_ids.masked_fill(~negative_attention, 0)

        # These modules are fixed in Stable LoRAOpt. Keep autograd enabled
        # through their operations so conditions can still update LoRA and
        # proj_in, while ensuring they never acquire MR-ARE gradients.
        self.vae.requires_grad_(False)
        self.mar.requires_grad_(False)
        self.proj_out.requires_grad_(False)

        target_latents_grid = self.encode(target_pixels)
        if target_latents_grid.ndim != 4:
            raise ValueError(
                'mrare_target_pixel_values must encode to a 4D latent grid')
        item_count, height, width, channels = target_latents_grid.shape
        target_latents = target_latents_grid.detach().reshape(
            item_count, height * width, channels)
        relevance = relevance.reshape(item_count, -1)
        if relevance.shape[1] != target_latents.shape[1]:
            raise ValueError(
                'mrare_relevance_mask must match encoded target latent grid')
        decoder_mask = relevance.to(dtype=target_latents_grid.dtype)
        positive_encoding = self.forward_mae_encoder(
            target_latents_grid,
            decoder_mask,
            input_ids=positive_ids,
            attention_mask=positive_attention,
        )
        positive_condition = self.mar.forward_mae_decoder(
            positive_encoding,
            decoder_mask,
            image_shape=(height, width),
        )
        if getattr(self, 'mrare_positive_only', False):
            from src.models.dllm.mrare.reverse_energy import (
                single_reverse_energy)
            positive_energy = single_reverse_energy(
                self.mar, positive_condition, target_latents, relevance)
            # P1 has no rank or distillation target. A detached placeholder
            # keeps common diagnostics typed without running any negative MAR
            # encoder, decoder, or diffusion forward.
            negative_energy = positive_energy.detach()
        else:
            negative_encoding = self.forward_mae_encoder(
                target_latents_grid,
                decoder_mask,
                input_ids=negative_ids,
                attention_mask=negative_attention,
            )
            negative_condition = self.mar.forward_mae_decoder(
                negative_encoding,
                decoder_mask,
                image_shape=(height, width),
            )
            positive_energy, negative_energy = paired_reverse_energy(
                self.mar,
                positive_condition,
                negative_condition,
                target_latents,
                relevance,
            )

        finite_energy = (
            torch.isfinite(positive_energy)
            & torch.isfinite(negative_energy)
            & torch.isfinite(energy_scale)
            & energy_scale.gt(0)
        )
        finite_indices = finite_energy.nonzero(as_tuple=False).flatten()
        nonfinite_energy_count = int((~finite_energy).sum().item())
        diagnostics['mrare_invalid_energy_count'] += nonfinite_energy_count
        diagnostics['mrare_invalid_count'] += nonfinite_energy_count
        if finite_indices.numel() == 0:
            return self._mrare_zero_auxiliary(
                last_hidden_state, diagnostics)

        positive_energy = positive_energy.index_select(0, finite_indices)
        negative_energy = negative_energy.index_select(0, finite_indices)
        finite_scale = energy_scale.index_select(0, finite_indices)
        valid_losses = mrare_losses(
            positive_energy,
            negative_energy,
            finite_scale,
            margin=self.mrare_margin,
            tau_energy=self.mrare_tau_energy,
            lambda_positive=self.mrare_lambda_positive,
            lambda_rank=self.mrare_lambda_rank,
            lambda_distill=0.0,
        )
        weighted_positive = (
            valid_losses['positive'] * self.mrare_lambda_positive)
        weighted_rank = valid_losses['rank'] * self.mrare_lambda_rank

        distill_logits = []
        distill_energy_indices = []
        distill_inactive_count = 0
        if self.mrare_distill_enabled and self.mrare_lambda_distill > 0.0:
            output_embeddings = self.llm.get_output_embeddings()
            for finite_position, energy_index in enumerate(
                finite_indices.tolist()
            ):
                batch_index = int(
                    usable_batch_indices[energy_index].item())
                offset = int(
                    changed_response_offsets[energy_index].item())
                response_positions = response_mask[batch_index].nonzero(
                    as_tuple=False).flatten()
                if not 0 <= offset < response_positions.numel():
                    distill_inactive_count += 1
                    continue
                position = int(response_positions[offset].item())
                if not bool(loss_mask[batch_index, position].item()):
                    distill_inactive_count += 1
                    continue
                logits = output_embeddings(
                    last_hidden_state[batch_index, position])
                positive_token_id = int(
                    positive_token_ids[energy_index].item())
                negative_token_id = int(
                    negative_token_ids[energy_index].item())
                distill_logits.append(
                    (logits[positive_token_id]
                     - logits[negative_token_id])
                    / self.mrare_tau_token)
                distill_energy_indices.append(finite_position)

        connected_zero = last_hidden_state.sum() * 0.0
        if distill_logits:
            token_preference_logits = torch.stack(distill_logits)
            distill_energy_indices = torch.tensor(
                distill_energy_indices,
                device=positive_energy.device,
                dtype=torch.long,
            )
            distill_losses = mrare_losses(
                positive_energy.index_select(0, distill_energy_indices),
                negative_energy.index_select(0, distill_energy_indices),
                finite_scale.index_select(0, distill_energy_indices),
                token_preference_logits=token_preference_logits,
                margin=self.mrare_margin,
                tau_energy=self.mrare_tau_energy,
                lambda_positive=0.0,
                lambda_rank=0.0,
                lambda_distill=self.mrare_lambda_distill,
            )
            weighted_distill = (
                distill_losses['distill'] * self.mrare_lambda_distill)
        else:
            weighted_distill = connected_zero

        diagnostics.update({
            'mrare_valid_count': int(finite_indices.numel()),
            'mrare_distill_active_count': len(distill_logits),
            'mrare_distill_inactive_count': distill_inactive_count,
        })
        result = {
            'weighted_positive': weighted_positive,
            'weighted_rank': weighted_rank,
            'weighted_distill': weighted_distill,
        }
        for name in self._mrare_diagnostic_names():
            result[name] = self._mrare_diagnostic(
                last_hidden_state, diagnostics.get(name, 0))
        return result

    def _mrfc_auxiliary_losses(self, data_dict, reference):
        """Apply the frozen feature-critic to one replay pair per SFT row."""
        connected_zero = reference.sum() * 0.0
        if (self.mrfc_lambda_pair == 0.0
                and self.mrfc_lambda_distill == 0.0):
            return {
                'weighted_pair': connected_zero,
                'weighted_distill': connected_zero,
                'mrfc_active_pairs': connected_zero.detach(),
                'mrfc_score_gap': connected_zero.detach(),
                'mrfc_pair_raw': connected_zero.detach(),
                'mrfc_distill_raw': connected_zero.detach(),
            }
        from .dllm.mrfc import spatial_answer_pool, spatial_visual_pool
        required = (
            'mrfc_valid',
            'mrfc_representation',
            'mrfc_target_pixel_values',
            'mrfc_relevance_mask',
            'mrfc_positive_condition_ids',
            'mrfc_positive_condition_mask',
            'mrfc_negative_condition_ids',
            'mrfc_negative_condition_mask',
            'mrfc_changed_condition_offset',
            'mrfc_positive_token_id',
            'mrfc_negative_token_id',
        )
        for key in required:
            if key not in data_dict:
                raise ValueError(f'MR-FC batch is missing required field {key}')
        valid = data_dict['mrfc_valid']
        if (not torch.is_tensor(valid) or valid.ndim != 1
                or valid.dtype != torch.bool or not bool(valid.all())):
            raise ValueError('MR-FC replay must mark every SFT row valid')
        batch_size = int(valid.shape[0])
        representations = data_dict['mrfc_representation']
        if (not isinstance(representations, (list, tuple))
                or len(representations) != batch_size
                or not set(representations).issubset({'rgb', 'depth', 'seg'})):
            raise ValueError('MR-FC representation assignment is invalid')

        def _tensor(key, ndim, dtype=None):
            value = data_dict[key]
            if (not torch.is_tensor(value) or value.ndim != ndim
                    or value.shape[0] != batch_size):
                raise ValueError(
                    f'{key} must have {ndim} dimensions and align with batch')
            value = value.to(device=self.device)
            return value.to(dtype=dtype) if dtype is not None else value

        target_pixels = _tensor(
            'mrfc_target_pixel_values', 4, self.dtype)
        relevance = _tensor('mrfc_relevance_mask', 3)
        if relevance.dtype != torch.bool:
            raise ValueError('mrfc_relevance_mask must be boolean')
        positive_ids = _tensor(
            'mrfc_positive_condition_ids', 2, torch.long)
        positive_mask = _tensor('mrfc_positive_condition_mask', 2)
        negative_ids = _tensor(
            'mrfc_negative_condition_ids', 2, torch.long)
        negative_mask = _tensor('mrfc_negative_condition_mask', 2)
        if positive_mask.dtype != torch.bool or negative_mask.dtype != torch.bool:
            raise ValueError('MR-FC condition masks must be boolean')
        changed_offsets = _tensor(
            'mrfc_changed_condition_offset', 1, torch.long)
        positive_token_ids = _tensor(
            'mrfc_positive_token_id', 1, torch.long)
        negative_token_ids = _tensor(
            'mrfc_negative_token_id', 1, torch.long)

        width = max(positive_ids.shape[1], negative_ids.shape[1])
        if positive_ids.shape[1] != width:
            pad = width - positive_ids.shape[1]
            positive_ids = F.pad(positive_ids, (0, pad), value=0)
            positive_mask = F.pad(positive_mask, (0, pad), value=False)
        if negative_ids.shape[1] != width:
            pad = width - negative_ids.shape[1]
            negative_ids = F.pad(negative_ids, (0, pad), value=0)
            negative_mask = F.pad(negative_mask, (0, pad), value=False)
        row = torch.arange(batch_size, device=self.device)
        safe_offset = changed_offsets.clamp(min=0, max=max(width - 1, 0))
        offsets_valid = changed_offsets.gt(0) & changed_offsets.lt(width)
        if width:
            offsets_valid &= (
                positive_mask[row, safe_offset]
                & negative_mask[row, safe_offset]
                & positive_ids[row, safe_offset].eq(positive_token_ids)
                & negative_ids[row, safe_offset].eq(negative_token_ids)
            )
        changed_positions = (
            positive_ids.ne(negative_ids) & positive_mask & negative_mask)
        offsets_valid &= changed_positions.sum(dim=1).eq(1)
        if width:
            offsets_valid &= changed_positions[row, safe_offset]
        if not bool(offsets_valid.all()):
            raise ValueError('MR-FC condition pair contract is invalid')
        if (not torch.isfinite(target_pixels).all()
                or not bool(relevance.flatten(1).any(dim=1).all())
                or not bool((~relevance.flatten(1)).any(dim=1).all())):
            raise ValueError('MR-FC target or relevance mask is invalid')

        target_grid = self.encode(target_pixels)
        if target_grid.ndim != 4:
            raise ValueError('MR-FC target must encode to a spatial grid')
        _, height, width_grid, _ = target_grid.shape
        visible_mask = torch.zeros(
            batch_size, height * width_grid,
            device=self.device, dtype=target_grid.dtype)
        base_visual, projected_visual = self.extract_visual_feature(
            target_grid, mask=visible_mask, detach=False)
        if relevance.flatten(1).shape[1] != height * width_grid:
            raise ValueError('MR-FC relevance mask does not match target grid')

        condition_ids = torch.cat([positive_ids, negative_ids], dim=0)
        condition_mask = torch.cat([positive_mask, negative_mask], dim=0)
        projected_pair = torch.cat(
            [projected_visual, projected_visual], dim=0)
        inputs = self.prepare_forward_input(
            x=projected_pair,
            input_ids=condition_ids,
            attention_mask=condition_mask,
        )
        output = self.llm_model(**inputs, return_dict=True)
        llm_visual = output.last_hidden_state[:, -projected_pair.shape[1]:]
        buffer_size = int(self.mar.buffer_size)
        llm_visual = torch.cat([
            llm_visual[:, -buffer_size:],
            llm_visual[:, :-buffer_size],
        ], dim=1)
        base_spatial = base_visual[:, buffer_size:].float()
        answer_spatial = llm_visual[:, buffer_size:].float()
        positive_spatial = answer_spatial[:batch_size]
        negative_spatial = answer_spatial[batch_size:]
        flat_relevance = relevance.reshape(batch_size, -1)
        visual_features = spatial_visual_pool(
            base_spatial, flat_relevance)
        positive_features = spatial_answer_pool(
            positive_spatial, flat_relevance)
        negative_features = spatial_answer_pool(
            negative_spatial, flat_relevance)

        gap_by_index = {}
        for representation in ('rgb', 'depth', 'seg'):
            indices = [index for index, assigned in enumerate(representations)
                       if assigned == representation]
            if not indices:
                continue
            selected = torch.tensor(
                indices, device=self.device, dtype=torch.long)
            positive_score, negative_score = (
                self.mrfc_critic_ensemble.score_pair(
                    representation,
                    visual_features.index_select(0, selected),
                    positive_features.index_select(0, selected),
                    negative_features.index_select(0, selected),
                ))
            for local_index, batch_index in enumerate(indices):
                gap_by_index[batch_index] = (
                    positive_score[local_index] - negative_score[local_index])
        critic_gap = torch.stack([
            gap_by_index[index] for index in range(batch_size)])
        pair_raw = F.softplus(self.mrfc_margin - critic_gap).mean()

        # The changed token is predicted from the identical causal prefix.
        # Critic confidence is stop-gradiented before becoming the soft target.
        prefix_offsets = changed_offsets - 1
        prefix_hidden = output.last_hidden_state[
            row, prefix_offsets]
        token_logits = self.llm.get_output_embeddings()(prefix_hidden)
        positive_logits = token_logits[row, positive_token_ids]
        negative_logits = token_logits[row, negative_token_ids]
        token_gap = positive_logits - negative_logits
        critic_target = torch.sigmoid(
            critic_gap.detach().float() / self.mrfc_tau_critic)
        distill_raw = F.binary_cross_entropy_with_logits(
            token_gap.float(), critic_target)

        finite = torch.stack([
            pair_raw.float(), distill_raw.float(), critic_gap.float().mean()])
        if not torch.isfinite(finite).all():
            raise FloatingPointError('MR-FC produced a non-finite loss or score')
        return {
            'weighted_pair': pair_raw * self.mrfc_lambda_pair,
            'weighted_distill': (
                distill_raw * self.mrfc_lambda_distill),
            'mrfc_active_pairs': pair_raw.detach().new_tensor(
                float(batch_size)),
            'mrfc_score_gap': critic_gap.detach().mean(),
            'mrfc_pair_raw': pair_raw.detach(),
            'mrfc_distill_raw': distill_raw.detach(),
        }

    def recon_loss(self, data_dict):

        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        _, z_enc = self.extract_visual_feature(x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def edit_loss(self, data_dict):
        """Image editing loss.
        
        Unlike reconstruction which uses the same image as input/output,
        editing uses source_image as condition and edited_image as target.
        """
        # Encode source image (condition)
        source_x = data_dict['source_pixel_values'].to(dtype=self.dtype, device=self.device)
        source_x = self.encode(source_x)  # b m n c
        
        # Encode target edited image
        target_x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        target_x = self.encode(target_x)   # b m n c
        b, m, n, _ = target_x.shape
        gt_latents = target_x.clone().detach().view(b, m*n, -1)
        
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        
        # Extract visual features from source image (as condition)
        _, z_enc = self.extract_visual_feature(source_x)
        
        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)
        
        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], 1088, dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ), dim=1
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], 1088, dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ), dim=1
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        # Generate target edited image
        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(target_x.flatten(1, 2), orders)
        
        x_enc = self.forward_mae_encoder(
            target_x, 
            mask, 
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask
        )
        
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def joint_loss(self, data_dict):
        """Joint loss for SFT and Depth estimation in a single step.
        
        This method manually batches inputs for SFT and Depth tasks to avoid
        calling shared modules (like Visual Encoder and LLM) multiple times
        in a single forward pass, which causes "Gradient computed twice" errors in DeepSpeed.
        """
        # --- 1. Prepare Inputs ---
        
        # SFT Data
        sft_input_ids = data_dict['sft_input_ids'].to(self.device)
        sft_labels = data_dict['sft_labels'].to(self.device)
        sft_attention_mask = data_dict['sft_attention_mask'].to(self.device)
        sft_pixel_values = data_dict['sft_pixel_values'].to(dtype=self.dtype, device=self.device)
        
        # Depth Data
        depth_source_pixel_values = data_dict['depth_source_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_target_pixel_values = data_dict['depth_pixel_values'].to(dtype=self.dtype, device=self.device)
        depth_input_ids = data_dict['depth_input_ids'].to(self.device)
        depth_attention_mask = data_dict['depth_attention_mask'].to(self.device)
        
        b_sft = sft_input_ids.shape[0]
        b_depth = depth_input_ids.shape[0]
        
        # --- 2. Shared VAE Encoding (Safe to call multiple times if frozen, but efficient to batch) ---
        # Concatenate all images for encoding: [SFT, Depth_Source, Depth_Target]
        all_images = torch.cat([sft_pixel_values, depth_source_pixel_values, depth_target_pixel_values], dim=0)
        all_latents = self.encode(all_images)
        
        sft_latents, depth_source_latents, depth_target_latents = torch.split(all_latents, [b_sft, b_depth, b_depth], dim=0)
        
        # --- 3. Shared Visual Encoder (MUST batch this if trainable) ---
        # Concatenate latents for visual feature extraction: [SFT, Depth_Source]
        # Note: Depth_Target also needs visual features inside forward_mae_encoder, but let's see.
        # In forward_mae_encoder(target_x), it calls extract_visual_feature(target_x).
        # So we need to extract features for ALL 3 sets of latents.
        
        all_vis_latents = torch.cat([sft_latents, depth_source_latents, depth_target_latents], dim=0)
        
        # We also need to generate masks for Depth_Target (for MAR)
        # SFT and Depth_Source don't use masks (mask=None/zeros) for feature extraction usually
        # But forward_mae_encoder passes a mask.
        
        # Generate masks for Depth_Target
        orders = self.mar.sample_orders(bsz=b_depth, seq_len=depth_target_latents.shape[1]*depth_target_latents.shape[2])
        depth_mask = self.mar.random_masking(depth_target_latents.flatten(1, 2), orders)
        
        # For SFT and Source, mask is zeros (no masking)
        # We need to create a combined mask for batching
        # Mask shape for one image: [b, mn]
        mask_zeros = torch.zeros(b_sft + b_depth, depth_mask.shape[1], device=self.device, dtype=depth_mask.dtype)
        all_masks = torch.cat([mask_zeros, depth_mask], dim=0)
        
        # Extract Visual Features (One pass)
        # This calls mar.forward_mae_encoder -> BUT wait, forward_mae_encoder calls llm_model!
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder. 
        # Is self.mar.forward_mae_encoder the same as self.forward_mae_encoder?
        # NO. self.mar is the MAR module; self.forward_mae_encoder is a method on the Harmon class.
        # Harmon.extract_visual_feature calls self.mar.forward_mae_encoder.
        # Harmon.forward_mae_encoder calls extract_visual_feature AND llm_model.
        
        # So extract_visual_feature runs the Vision Transformer.
        # We run this ONCE.
        
        all_x_enc, all_z_enc = self.extract_visual_feature(all_vis_latents, mask=all_masks)
        
        # Split results
        # z_enc is the projected features used as inputs to LLM
        # x_enc is the features from Vision Encoder (before projection?) No, extract_visual_feature returns proj_in(x_enc).
        # Actually: x_enc from mar.forward_mae_encoder, z_enc = proj_in(x_enc).
        # We need z_enc for SFT and Depth Source.
        # We need x_enc for Depth Target (residual connection).
        
        z_enc_sft, z_enc_source, z_enc_target = torch.split(all_z_enc, [b_sft, b_depth, b_depth], dim=0)
        x_enc_target = torch.split(all_x_enc, [b_sft, b_depth, b_depth], dim=0)[2]
        
        # Apply scale gradient if needed
        if self.grad_scale is not None:
            z_enc_sft = _ScaleGradient.apply(z_enc_sft, self.grad_scale)
            z_enc_source = _ScaleGradient.apply(z_enc_source, self.grad_scale)
            # Target features usually don't get scaled? edit_loss doesn't scale target features explicitly, 
            # but forward_mae_encoder calls extract_visual_feature which returns z_enc.
            # In forward_mae_encoder, z_enc is used as input to LLM.
            # Does it scale? In edit_loss code:
            # x_enc = self.forward_mae_encoder(...)
            # Inside forward_mae_encoder: x_enc, z_enc = self.extract_visual_feature(..., detach=detach)
            # It doesn't seem to apply scale gradient inside forward_mae_encoder.
            # So we apply it only for SFT and Source (Condition).
        
        # --- 4. Prepare LLM Inputs ---
        
        # Get actual visual token length from encoder output
        vis_token_len = z_enc_sft.shape[1]

        # A. SFT Inputs construction (Logic from image2text_loss)
        for i in range(sft_input_ids.shape[0]):
            assert sft_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        # Insert Image Tokens placeholders
        sft_input_ids_exp = torch.cat((
            sft_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_input_ids[:, 4:]), dim=1)
            
        sft_attention_mask_exp = torch.cat((
            sft_attention_mask[:, :3],
            torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_attention_mask[:, 4:]), dim=1)
            
        # SFT Labels (for loss later)
        sft_labels_exp = torch.cat((
            sft_labels[:, :3],
            sft_labels[0, 3] * torch.ones(b_sft, vis_token_len, dtype=torch.long, device=self.device),
            sft_labels[:, 4:]), dim=1)
            
        # Embeddings
        sft_inputs_embeds = z_enc_sft.new_zeros(*sft_input_ids_exp.shape, self.llm.config.hidden_size)
        sft_inputs_embeds[sft_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_sft.flatten(0, 1)
        mask_text = sft_input_ids_exp != IMAGE_TOKEN_INDEX
        sft_inputs_embeds[mask_text] = self.llm.get_input_embeddings()(sft_input_ids_exp[mask_text])
        
        # B. Depth Inputs construction (Logic from edit_loss -> forward_mae_encoder)
        # First prepare the "Condition" embeddings (Source Image + Prompt)
        for i in range(depth_input_ids.shape[0]):
            assert depth_input_ids[i, 3] == IMAGE_TOKEN_INDEX
            
        depth_input_ids_exp = torch.cat((
            depth_input_ids[:, :3],
            IMAGE_TOKEN_INDEX * torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_input_ids[:, 4:]), dim=1)
            
        depth_attention_mask_exp = torch.cat((
            depth_attention_mask[:, :3],
            torch.ones(b_depth, vis_token_len, dtype=torch.long, device=self.device),
            depth_attention_mask[:, 4:]), dim=1)
            
        # Condition Embeddings (Source + Prompt)
        depth_cond_embeds = z_enc_source.new_zeros(*depth_input_ids_exp.shape, self.llm.config.hidden_size)
        depth_cond_embeds[depth_input_ids_exp == IMAGE_TOKEN_INDEX] = z_enc_source.flatten(0, 1)
        mask_cond_text = depth_input_ids_exp != IMAGE_TOKEN_INDEX
        depth_cond_embeds[mask_cond_text] = self.llm.get_input_embeddings()(depth_input_ids_exp[mask_cond_text])
        
        # Now combine Condition with Target (Logic from prepare_forward_input)
        # inputs_embeds = torch.cat([inputs_embeds, x], dim=1) where x is z_enc_target
        depth_full_embeds = torch.cat([depth_cond_embeds, z_enc_target], dim=1)
        
        # Prepare Mask for Depth (Logic from prepare_forward_input)
        # attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)
        l_target = z_enc_target.shape[1]
        depth_full_mask = torch.cat([
            depth_attention_mask_exp, 
            depth_attention_mask_exp.new_ones(b_depth, l_target)
        ], dim=1)
        
        # --- 5. Batch LLM Forward ---
        # We need to concat sft_inputs_embeds and depth_full_embeds.
        # They might have different lengths. We need to pad.
        
        len_sft = sft_inputs_embeds.shape[1]
        len_depth = depth_full_embeds.shape[1]
        max_len = max(len_sft, len_depth)
        
        # Pad SFT
        if len_sft < max_len:
            pad_len = max_len - len_sft
            pad_embeds = torch.zeros(b_sft, pad_len, sft_inputs_embeds.shape[2], device=self.device, dtype=self.dtype)
            sft_inputs_embeds = torch.cat([sft_inputs_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_sft, pad_len, device=self.device, dtype=sft_attention_mask_exp.dtype)
            sft_attention_mask_exp = torch.cat([sft_attention_mask_exp, pad_mask], dim=1)
            
        # Pad Depth
        if len_depth < max_len:
            pad_len = max_len - len_depth
            pad_embeds = torch.zeros(b_depth, pad_len, depth_full_embeds.shape[2], device=self.device, dtype=self.dtype)
            depth_full_embeds = torch.cat([depth_full_embeds, pad_embeds], dim=1)
            pad_mask = torch.zeros(b_depth, pad_len, device=self.device, dtype=depth_full_mask.dtype)
            depth_full_mask = torch.cat([depth_full_mask, pad_mask], dim=1)
            
        # Concat
        joint_inputs_embeds = torch.cat([sft_inputs_embeds, depth_full_embeds], dim=0)
        joint_attention_mask = torch.cat([sft_attention_mask_exp, depth_full_mask], dim=0)
        
        # Forward LLM (ONE PASS!)
        output = self.llm_model(inputs_embeds=joint_inputs_embeds,
                                attention_mask=joint_attention_mask,
                                return_dict=True)
        
        last_hidden_state = output.last_hidden_state
        
        # --- 6. Compute Losses ---
        
        # A. SFT Loss
        # Slice out SFT part
        sft_hidden = last_hidden_state[:b_sft, :len_sft] # Remove padding
        
        # Logic from image2text_loss
        # shift labels and hidden states
        sft_hidden = sft_hidden[:, :-1]
        sft_labels_shifted = sft_labels_exp[:, 1:]
        
        # Flatten and filter
        # Note: sft_labels_exp was not padded in Step 5, so we use original shape logic
        # BUT sft_hidden is taken from padded batch. We sliced to len_sft.
        # However, sft_labels_exp matches len_sft (the length before padding).
        
        sft_hidden_flat = sft_hidden[sft_labels_shifted >= 0]
        sft_labels_flat = sft_labels_shifted[sft_labels_shifted >= 0]
        
        sft_logits = self.llm.get_output_embeddings()(sft_hidden_flat)
        loss_sft = F.cross_entropy(input=sft_logits, target=sft_labels_flat)
        
        # B. Depth Loss
        # Slice out Depth part
        depth_hidden = last_hidden_state[b_sft:, :len_depth]
        
        # Logic from forward_mae_encoder: z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]
        # z_enc here refers to z_enc_target (length 256 usually)
        # depth_hidden contains [Condition, Target]. We want the part corresponding to Target.
        # Target length is l_target
        z_llm_depth = depth_hidden[:, -l_target:]
        
        # Logic: move buffers back (from forward_mae_encoder)
        z_llm_depth = torch.cat([
            z_llm_depth[:, -self.mar.buffer_size:],
            z_llm_depth[:, :-self.mar.buffer_size]], dim=1)
            
        # Residual learning
        x_enc_final = x_enc_target + self.proj_out(z_llm_depth)
        
        # Decoder and Loss
        gt_latents = depth_target_latents.clone().detach().view(b_depth, -1, depth_target_latents.shape[-1]) # Flatten m*n
        z_dec = self.mar.forward_mae_decoder(x_enc_final, depth_mask, image_shape=(16, 16)) # Assuming 16x16 patches?
        # Need to get m, n from latents shape.
        # all_latents shape: [b, m, n, c]
        m, n = depth_target_latents.shape[1], depth_target_latents.shape[2]
        
        loss_depth = self.mar.forward_loss(z=z_dec, target=gt_latents, mask=depth_mask)
        
        return loss_sft, loss_depth

    def find_response_span(self, input_ids):
        n = len(input_ids)
        IM_END_ID = 151645
        header_prefix = [151644, 77091]
        m = len(header_prefix)

        start_idx = -1

        for i in range(n - m, -1, -1):
            # 匹配 <|im_start|>assistant\n
            if input_ids[i : i+m].tolist() == header_prefix:
                # 找到了 <|im_start|>assistant
                # 我们跳过这个 header 以及紧随其后的“换行/格式”Token
                # 无论它是 \n, \n\n 还是 .\n
                start_idx = i + m + 1 # 内容从 header 之后开始
                break

        if start_idx == -1: 
            return None, None

        end_idx = -1
        for i in range(start_idx, n):
            if input_ids[i].item() == IM_END_ID:
                end_idx = i + 1 # 包含这个 <|im_end|> 本身
                break

        # 没找到，取到序列末尾进行兜底。
        if end_idx == -1:
            end_idx = n 

        return start_idx, end_idx

    def _sample_block_times(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        # ``low``/``high`` are mask probabilities; t = 1 - mask_prob (plan §3.1).
        t = 1.0 - torch.rand(num_blocks, device=device).clamp(low, high)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=t.dtype) / num_blocks
            t = t / num_blocks + offset
            t = t[torch.randperm(num_blocks, device=device)]
        return t
    
    def create_attention_mask(self, prefix_len, res_len, block_size, max_len):
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        # Prefix: 每个 token 独占一个 block_id，让 base_mask 退化为严格 token 级 causal，
        # 避免把 Qwen 预训练的因果分布破坏成块内双向。
        block_ids[:clean_start] = torch.arange(clean_start)
        token_types[:clean_start] = 0

        # 响应段的 block_id 从 prefix_len 起递增（保持与 prefix 完全隔离）
        response_block_offset = clean_start

        if clean_end > clean_start:
            clean_len = clean_end - clean_start
            clean_rel_positions = torch.arange(clean_len)
            clean_blocks = response_block_offset + clean_rel_positions // block_size
            block_ids[clean_start:clean_end] = clean_blocks
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            noisy_len = seq_len - noisy_start
            noisy_rel_positions = torch.arange(noisy_len)
            noisy_blocks = response_block_offset + noisy_rel_positions // block_size
            block_ids[noisy_start:seq_len] = noisy_blocks
            token_types[noisy_start:seq_len] = 2

        q_blocks = block_ids.view(-1, 1)
        k_blocks = block_ids.view(1, -1)
        q_types = token_types.view(-1, 1)
        k_types = token_types.view(1, -1)

        # base_mask：prefix/clean 的 query 走块级因果；同时禁止任何非 noisy query 看到 noisy K。
        # 这样 clean 段永远看不到 noisy 段，避免表征污染。
        base_mask = (q_blocks >= k_blocks) & (k_types != 2)
        # noisy query 的可见范围：全部 prefix + 严格前置 clean block + 同 block 的 noisy token。
        noisy_query_mask = q_types == 2
        noisy_visibility = (k_types == 0) | ((k_types == 1) & (k_blocks < q_blocks)) | ((k_types == 2) & (k_blocks == q_blocks))
        local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_dllm_batch(
        self,
        data_dict,
        mask_token_id=None,
        pad_token_id=None,
    ):
        if mask_token_id is None:
            mask_token_id = int(self.harmon_mask_token_id.item())
        if pad_token_id is None:
            pad_token_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer is not None
                else 151645
            )
        block_size = getattr(self, "block_size", 32)

        for key in data_dict.keys():
            if 'input_ids' not in data_dict[key] or 'labels' not in data_dict[key]:
                continue
            
            data_batch = data_dict[key]
            input_ids_list = data_batch['input_ids']
            labels_list = data_batch['labels']
            bsz = len(input_ids_list)

            batch_input_ids, batch_labels = [], []
            batch_t_types, batch_b_indices = [], []
            batch_response_mask, batch_loss_mask = [], [] # 【新增】：容器
            batch_position_ids = []  # 【新增】：per-sample position_ids，用于对齐 clean/noisy
            batch_corruption_states = []
            batch_lengths = []
            batch_t = []
            active_targets = 0
            legitimate_empty_responses = 0
            unexpected_empty_masked_nonempty = 0

            for i in range(bsz):
                i_ids = input_ids_list[i].clone()
                l_ids = labels_list[i].clone()
                response_corruption_state = None
                
                start_idx, end_idx = self.find_response_span(i_ids)
                if start_idx is None or end_idx is None:
                    if self.enforce_nonempty_dllm_targets:
                        raise ValueError(
                            'missing assistant response span in dLLM batch')
                    start_idx, end_idx = 0, 0
                    
                len_res = end_idx - start_idx
                
                if len_res > 0:
                    ar_i_ids = i_ids[:end_idx]
                    ar_l_ids = l_ids[:end_idx]
                    
                    # 补齐到 block_size 的倍数
                    remainder = len_res % block_size
                    pad_len = block_size - remainder if remainder != 0 else 0
                    
                    if pad_len > 0:
                        pad_tensor = torch.full((pad_len,), pad_token_id, dtype=ar_i_ids.dtype, device=ar_i_ids.device)
                        lbl_pad_tensor = torch.full((pad_len,), -100, dtype=ar_l_ids.dtype, device=ar_l_ids.device)
                        ar_i_ids = torch.cat([ar_i_ids, pad_tensor])
                        ar_l_ids = torch.cat([ar_l_ids, lbl_pad_tensor])
                        end_idx += pad_len
                        len_res += pad_len
                    
                    # AR 区域属性
                    ar_t_types = torch.zeros_like(ar_i_ids)
                    ar_b_indices = torch.zeros_like(ar_i_ids)
                    ar_t_types[start_idx:end_idx] = 1
                    
                    # 【新增】：AR 区域的 Mask 初始化 (全为 False)
                    ar_response_mask = torch.zeros_like(ar_i_ids, dtype=torch.bool)
                    ar_loss_mask = torch.zeros_like(ar_i_ids, dtype=torch.bool)
                    
                    clean_resp_ids = ar_i_ids[start_idx:end_idx]
                    clean_resp_labels = ar_l_ids[start_idx:end_idx]
                    
                    n_blocks = len_res // block_size
                    t_tensor = self._sample_block_times(
                        n_blocks,
                        device=ar_i_ids.device,
                        low=getattr(self, "min_mask_rate", 0.0),
                        high=getattr(self, "max_mask_rate", 1.0),
                    )
                    batch_t.append(t_tensor)

                    stable_noisy_ids = None
                    if self.enforce_nonempty_dllm_targets:
                        token_times = t_tensor.repeat_interleave(block_size)
                        response_random_values = torch.rand(
                            len_res, device=clean_resp_ids.device
                        )
                        corruptor = MaskedCorruptor(
                            mask_token_id=mask_token_id,
                            enforce_nonempty=True,
                            excluded_token_ids=(
                                IMAGE_TOKEN_INDEX,
                                int(self.harmon_image_token_id.item()),
                            ),
                        )
                        stable_noisy_ids, response_corruption_state = (
                            corruptor.corrupt(
                                clean_resp_ids,
                                clean_resp_labels,
                                t=token_times,
                                random_values=response_random_values,
                            )
                        )
                        response_corruption_state.require_valid_supervision()
                        force_flags = data_batch.get(
                            'mrare_force_changed_token')
                        valid_flags = data_batch.get('mrare_valid')
                        changed_offsets = data_batch.get(
                            'mrare_changed_response_offset')
                        force_this = (
                            self.mrare_force_changed_token
                            and torch.is_tensor(force_flags)
                            and torch.is_tensor(valid_flags)
                            and torch.is_tensor(changed_offsets)
                            and bool(force_flags[i].item())
                            and bool(valid_flags[i].item())
                        )
                        if force_this:
                            changed_offset = int(changed_offsets[i].item())
                            if (changed_offset < 0
                                    or changed_offset >= clean_resp_ids.numel()
                                    or clean_resp_labels[changed_offset].item() < 0):
                                raise ValueError(
                                    'formal changed response offset is invalid')
                            stable_noisy_ids[changed_offset] = mask_token_id
                            active_mask = (
                                response_corruption_state.active_mask.clone())
                            valid_target_mask = (
                                response_corruption_state.valid_target_mask.clone())
                            active_mask[changed_offset] = True
                            valid_target_mask[changed_offset] = True
                            response_corruption_state = replace(
                                response_corruption_state,
                                active_mask=active_mask,
                                valid_target_mask=valid_target_mask,
                            )
                            response_corruption_state.require_valid_supervision()
                        if response_corruption_state.legitimate_empty:
                            legitimate_empty_responses += 1

                    diff_i_ids_blocks, diff_l_ids_blocks = [], []
                    diff_t_types_blocks, diff_b_indices_blocks = [], []
                    diff_resp_mask_blocks, diff_loss_mask_blocks = [], [] # 【新增】：Block级Mask容器
                    
                    for b_idx in range(n_blocks):
                        b_start = b_idx * block_size
                        b_end = (b_idx + 1) * block_size
                        
                        b_clean_ids = clean_resp_ids[b_start:b_end]
                        b_clean_labels = clean_resp_labels[b_start:b_end]
                        
                        if self.enforce_nonempty_dllm_targets:
                            b_noisy_ids = stable_noisy_ids[b_start:b_end]
                            b_mask_idx = (
                                response_corruption_state.active_mask[
                                    b_start:b_end
                                ]
                            )
                        else:
                            b_rand_vals = torch.rand(
                                block_size, device=self.device
                            )
                            b_t = t_tensor[b_idx].item()
                            b_mask_prob = 1.0 - b_t
                            b_mask_idx = b_rand_vals < b_mask_prob
                            if b_idx == n_blocks - 1 and pad_len > 0:
                                b_mask_idx[-pad_len:] = False
                            b_noisy_ids = b_clean_ids.clone()
                            b_noisy_ids[b_mask_idx] = mask_token_id
                        
                        # 组装当前 Block 基础属性
                        diff_i_ids_blocks.append(b_noisy_ids)
                        diff_l_ids_blocks.append(b_clean_labels)
                        diff_t_types_blocks.append(torch.full_like(b_noisy_ids, 2))
                        diff_b_indices_blocks.append(torch.full_like(b_noisy_ids, b_idx + 1))
                        
                        # 【新增】：组装当前 Block 的 Masks
                        # response_mask：整个 Block 都是 True
                        diff_resp_mask_blocks.append(torch.ones_like(b_noisy_ids, dtype=torch.bool))
                        # loss_mask：只有被 mask_idx 选中的 token 才是 True
                        diff_loss_mask_blocks.append(b_mask_idx.clone())
                        
                        # 映射 AR 区域对应的 block_index
                        ar_b_indices[start_idx + b_start : start_idx + b_end] = b_idx + 1

                    # 拼装 Diff 区域
                    diff_i_ids = torch.cat(diff_i_ids_blocks)
                    diff_l_ids = torch.cat(diff_l_ids_blocks)
                    diff_t_types = torch.cat(diff_t_types_blocks)
                    diff_b_indices = torch.cat(diff_b_indices_blocks)
                    
                    # 【新增】：拼装 Diff 区域的 Masks
                    diff_response_mask = torch.cat(diff_resp_mask_blocks)
                    diff_loss_mask = torch.cat(diff_loss_mask_blocks)
                    
                    # 单条样本大拼接
                    sample_i_ids = torch.cat([ar_i_ids, diff_i_ids])
                    sample_l_ids = torch.cat([ar_l_ids, diff_l_ids])
                    sample_t_types = torch.cat([ar_t_types, diff_t_types])
                    sample_b_indices = torch.cat([ar_b_indices, diff_b_indices])
                    
                    # 【新增】：单条样本 Masks 拼接
                    sample_resp_mask = torch.cat([ar_response_mask, diff_response_mask])
                    sample_loss_mask = torch.cat([ar_loss_mask, diff_loss_mask])
                    batch_lengths.append((start_idx, len_res))
                    valid_active = (
                        sample_resp_mask
                        & sample_loss_mask
                        & sample_l_ids.ne(-100)
                    )
                    sample_active_count = int(valid_active.sum().item())
                    active_targets += sample_active_count
                    has_valid_target = bool(clean_resp_labels.ne(-100).any())
                    if has_valid_target and sample_active_count == 0:
                        unexpected_empty_masked_nonempty += 1
                        if self.enforce_nonempty_dllm_targets:
                            raise RuntimeError(
                                'nonempty response has zero active targets')
                else:
                    # 兜底：没有找到回复的情况
                    batch_t.append(torch.empty(0, dtype=torch.float32, device=self.device))
                    sample_i_ids = i_ids[i_ids != pad_token_id] 
                    sample_l_ids = l_ids[:len(sample_i_ids)]
                    sample_t_types = torch.zeros_like(sample_i_ids)
                    sample_b_indices = torch.zeros_like(sample_i_ids)
                    
                    # 【新增】：兜底全为 False
                    sample_resp_mask = torch.zeros_like(sample_i_ids, dtype=torch.bool)
                    sample_loss_mask = torch.zeros_like(sample_i_ids, dtype=torch.bool)
                    batch_lengths.append((len(sample_i_ids), 0))
                    legitimate_empty_responses += 1

                # 【新增】：构造 position_ids，让 noisy 段与 clean 段共享位置编码，
                # 消除 RoPE 在 clean/noisy 上的位置偏移，保证训练与推理分布一致。
                sample_pos_ids = torch.arange(sample_i_ids.shape[0], dtype=torch.long)
                if len_res > 0:
                    p_len_i, r_len_i = batch_lengths[-1]
                    sample_pos_ids[p_len_i + r_len_i : p_len_i + 2 * r_len_i] = \
                        torch.arange(p_len_i, p_len_i + r_len_i, dtype=torch.long)

                batch_input_ids.append(sample_i_ids)
                batch_labels.append(sample_l_ids)
                batch_t_types.append(sample_t_types)
                batch_b_indices.append(sample_b_indices)
                batch_response_mask.append(sample_resp_mask) # 【新增】
                batch_loss_mask.append(sample_loss_mask)     # 【新增】
                batch_position_ids.append(sample_pos_ids)    # 【新增】
                batch_corruption_states.append(
                    response_corruption_state
                )

            # ==========================================
            # Batch 级别的统一 Padding
            # ==========================================
            data_batch['input_ids'] = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
            data_batch['labels'] = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
            data_batch['token_types'] = pad_sequence(batch_t_types, batch_first=True, padding_value=-1)
            data_batch['block_indices'] = pad_sequence(batch_b_indices, batch_first=True, padding_value=-1)
            
            # 【新增】：Padding 值为 False
            data_batch['response_mask'] = pad_sequence(batch_response_mask, batch_first=True, padding_value=False)
            data_batch['loss_mask'] = pad_sequence(batch_loss_mask, batch_first=True, padding_value=False)
            data_batch['t'] = pad_sequence(batch_t, batch_first=True, padding_value=1.0)
            # 【新增】：position_ids padding 用 0，padded 位置本身不参与 attention（被 attention_mask 屏蔽）
            data_batch['position_ids'] = pad_sequence(batch_position_ids, batch_first=True, padding_value=0)
            data_batch['corruption_states'] = batch_corruption_states
            # Pack corruption state features for the state conditioner and/or
            # the GUARD RankGate (plan §3.3, §8.3).
            if (self.dllm_state_conditioner is not None
                    or getattr(self, 'guard_enabled', False)):
                state_features, state_mask = (
                    pack_corruption_state_features(
                        batch_corruption_states,
                        data_batch['token_types'],
                        fields=self.state_conditioner_fields,
                    )
                )
                data_batch['corruption_state_features'] = state_features
                data_batch['corruption_state_mask'] = state_mask
            data_batch['dllm_stats'] = {
                'active_targets': active_targets,
                'legitimate_empty_responses': legitimate_empty_responses,
                'unexpected_empty_masked_nonempty': (
                    unexpected_empty_masked_nonempty
                ),
            }
            
            batch_max_len = data_batch['input_ids'].shape[1]
            batch_attn_masks = []
            for p_len, r_len in batch_lengths:
                # 原封不动调用你现有的 self.create_attention_mask 函数
                single_mask = self.create_attention_mask(
                    prefix_len=p_len,
                    res_len=r_len,
                    block_size=block_size,
                    max_len=batch_max_len
                )
                batch_attn_masks.append(single_mask)
                
            # 沿 batch 维度拼接，挂载到 data_batch
            data_batch['attention_mask'] = torch.cat(batch_attn_masks, dim=0).to(self.device)
            
        return data_dict
        
    # ============================================================
    # Block Diffusion Inference (dLLM Generate)
    # ============================================================

    @torch.no_grad()
    def generate_dllm(
        self,
        inputs_embeds,
        max_new_tokens=512,
        block_size=None,
        denoising_steps=None,
        temperature=0.0,
        mask_token_id=None,
        eos_token_id=None,
        guard_risk_commit: Optional[bool] = None,
        tracer_router_enabled: Optional[bool] = None,
        tracer_policy_enabled: Optional[bool] = None,
        trajectory_records: Optional[list] = None,
        runtime_diagnostics=None,
    ):
        """Block Diffusion 推理解码。

        与训练对齐的解码逻辑：
          - 块间 (inter-block): AR 因果，逐块生成并更新 KV Cache。
          - 块内 (intra-block): 迭代去噪，使用全可见注意力；
            每步预测所有 mask token，将置信度最高的固定，其余 remask。

        Args:
            inputs_embeds: [batch, prefix_len, hidden_dim] 已包含图像嵌入的前缀。
            max_new_tokens: 最多生成的 token 数。
            block_size: 每个解码块的大小，默认 self.block_size。
            denoising_steps: 每块去噪步数，默认 block_size（每步固定约 1 个 token）。
            temperature: 采样温度，0 = greedy argmax。
            mask_token_id: Mask 占位符 token ID。
            eos_token_id: 终止 token ID。

        Returns:
            generated_ids: [batch, gen_len] 生成的 token ID 序列（截断到 eos）。
        """
        if block_size is None:
            block_size = getattr(self, 'block_size', 32)
        if denoising_steps is None:
            denoising_steps = block_size
        if mask_token_id is None:
            mask_token_id = int(self.harmon_mask_token_id.item())
        if eos_token_id is None:
            eos_token_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer is not None
                else 151645
            )
        if tracer_router_enabled is None:
            tracer_router_enabled = bool(getattr(
                self, 'tracer_router_enabled', True))
        if tracer_policy_enabled is None:
            tracer_policy_enabled = bool(getattr(
                self, 'tracer_policy_enabled', True))
        # Backward-compatible call sites can still supply the historical
        # flag. New TRACER scripts use the two independent switches above.
        if guard_risk_commit is not None:
            tracer_policy_enabled = bool(guard_risk_commit)

        device = self.device
        dtype = self.dtype
        batch_size = inputs_embeds.shape[0]
        prefix_len = inputs_embeds.shape[1]
        hidden_dim = inputs_embeds.shape[2]

        num_blocks = (max_new_tokens + block_size - 1) // block_size

        # GUARD §5.6/§5.7: build per-round corruption states from the
        # transfer schedule and a CommitRevisionPolicy for risk-aware
        # commit.  Both are only enabled when guard_lora_manager is
        # installed and a frozen RiskHead is available.  When the risk
        # head is missing the policy falls back to confidence top-k, so
        # behaviour matches the original generate_dllm.
        guard_mgr = getattr(self, 'guard_lora_manager', None)
        guard_has_risk = (
            guard_mgr is not None and self._guard_risk_head is not None)
        guard_trace_enabled = (
            guard_mgr is not None
            and (guard_has_risk or trajectory_records is not None)
        )
        if guard_mgr is not None:
            from src.models.dllm.guard.risk_control import (
                CommitRevisionPolicy,
                transfer_schedule_to_mask_probs,
            )
            transfer_schedule = self._transfer_schedule(
                block_size, denoising_steps)
            round_states = transfer_schedule_to_mask_probs(
                transfer_schedule, block_size
            ).to(device)  # [K, 3]
            commit_policy = CommitRevisionPolicy(block_size=block_size)
            guard_state_clean = torch.tensor(
                [[1.0, -math.log(1e-6), 0.0]] * batch_size,
                device=device, dtype=dtype,
            )
        else:
            transfer_schedule = self._transfer_schedule(
                block_size, denoising_steps)
            round_states = None
            commit_policy = None
            guard_state_clean = None

        # ---------- 1. Prefix 编码，构建初始 KV Cache ----------
        if guard_mgr is not None:
            # Prefix forward uses the clean endpoint state (no lagged
            # evidence yet, §5.6).
            guard_mgr.set_round_context(
                state=guard_state_clean,
                evidence=None,
                routing_enabled=tracer_router_enabled,
            )
            self._guard_lagged_evidence = None
            self._guard_prev_logits = None
            self._guard_prev_candidates = None
            self._guard_committed_history = None
        output = self.llm_model(
            inputs_embeds=inputs_embeds,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        kv_cache = output.past_key_values
        if guard_mgr is not None:
            guard_mgr.reset()

        all_block_ids = []
        finished = False

        # ---------- 2. 逐块生成 ----------
        for b_idx in range(num_blocks):
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_block_started()
            # 当前块的 position_ids（与训练一致，contiguous from prefix_len + offset）
            block_start_pos = prefix_len + b_idx * block_size
            block_pos_ids = torch.arange(
                block_start_pos, block_start_pos + block_size,
                device=device, dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

            # 初始化：全部为 mask token
            block_ids = torch.full(
                (batch_size, block_size), mask_token_id,
                dtype=torch.long, device=device,
            )

            # 构造块内全可见 4D attention mask（禁止 HF 默认因果）
            # shape: [batch, 1, block_size, cache_len + block_size]
            cache_len = kv_cache.get_seq_length()
            total_kv_len = cache_len + block_size
            attn_mask_4d = torch.zeros(
                (batch_size, 1, block_size, total_kv_len),
                dtype=dtype, device=device,
            )

            # Per-block lagged state. Evidence is token-position-specific and
            # is reset at every new block, so each block's first round is
            # neutral and starts without committed tokens.
            current_committed_block = torch.zeros(
                batch_size, block_size, dtype=torch.bool, device=device,
            )
            ever_committed_block = torch.zeros_like(current_committed_block)
            remask_block = torch.zeros_like(current_committed_block)
            prev_probs_block = None
            prev_candidates_block = None
            # Evidence is token-position-specific, so it never crosses a
            # block boundary. The first round of every block is neutral.
            lagged_evidence_block = None
            cumulative_target = 0

            # ---------- 3. 块内迭代去噪 ----------
            for step in range(denoising_steps):
                is_mask = (block_ids == mask_token_id)
                if not is_mask.any():
                    break
                if runtime_diagnostics is not None:
                    runtime_diagnostics.record_round_started()

                # Snapshot round k's lagged evidence before the forward
                # computes g_hat^k. Both the RankGate and commit policy for
                # this round consume exactly g_hat^(k-1).
                round_lagged_evidence = lagged_evidence_block

                # GUARD §5.6: feed the round-k state + lagged evidence to
                # the RankGate before the LLM forward.
                if guard_mgr is not None and round_states is not None:
                    state_k = round_states[step].unsqueeze(0).expand(
                        batch_size, -1).to(device=device, dtype=dtype)
                    # Lagged evidence is per-token from the previous round;
                    # every block resets it, so each block's first round sees
                    # ``None``.
                    ev = (round_lagged_evidence
                          if round_lagged_evidence is not None
                          else None)
                    guard_mgr.set_round_context(
                        state=state_k,
                        evidence=ev,
                        routing_enabled=tracer_router_enabled,
                    )

                block_embeds = self.llm.get_input_embeddings()(block_ids)
                mask_token_delta = getattr(self, 'mask_token_delta', None)
                if mask_token_delta is not None:
                    block_embeds = mask_token_delta(
                        block_embeds,
                        block_ids,
                    )

                out = self.llm_model(
                    inputs_embeds=block_embeds,
                    position_ids=block_pos_ids,
                    attention_mask=attn_mask_4d,
                    past_key_values=kv_cache,
                    use_cache=False,
                    return_dict=True,
                )

                # DynamicCache.update() 在 use_cache=False 时仍会扩展 cache。
                # 用公开 API 裁剪并同步 cache 的内部长度记账。
                kv_cache.crop(cache_len)

                if runtime_diagnostics is not None and guard_mgr is not None:
                    runtime_diagnostics.record_gate_scales(
                        getattr(guard_mgr, 'last_gate_scales', {}).values(),
                        active=(
                            bool(tracer_router_enabled)
                            and round_lagged_evidence is not None
                        ),
                    )
                if guard_mgr is not None:
                    guard_mgr.reset()

                logits = self.llm.get_output_embeddings()(
                    out.last_hidden_state
                )  # [batch, block_size, vocab]
                valid_logit_mask = getattr(
                    self, 'dllm_valid_logit_mask', None
                )
                if valid_logit_mask is not None:
                    logits = valid_logit_mask(logits)

                # Compute §5.4'.1 trajectory features for this round, so
                # the frozen RiskHead can produce g_hat for the next
                # round's RankGate and (optionally) the commit policy.
                features = None
                g_hat = None
                if guard_trace_enabled:
                    from src.models.dllm.guard.risk_control import (
                        compute_trajectory_features,
                    )
                    state_k_for_feats = (
                        round_states[step].unsqueeze(0).expand(
                            batch_size, -1
                        ).to(device)
                        if round_states is not None
                        else guard_state_clean
                    )
                    features = compute_trajectory_features(
                        hidden=out.last_hidden_state,
                        logits=logits,
                        prev_probs=prev_probs_block,
                        prev_candidates=prev_candidates_block,
                        committed_history=ever_committed_block,
                        remask=remask_block,
                        state=state_k_for_feats,
                        block_size=block_size,
                    )
                    if guard_has_risk:
                        g_hat = self._guard_risk_head(
                            features.hidden,
                            features.confidence,
                            features.entropy,
                            features.js_div,
                            features.stable,
                            features.committed_history.float(),
                            features.remask.float(),
                            features.block_commit_corr,
                            features.state,
                        )  # [batch, block_size, 1]
                        if self._guard_isotonic is not None:
                            g_hat = self._guard_isotonic.transform(
                                g_hat).view_as(g_hat)
                        g_hat = g_hat.detach()
                        if runtime_diagnostics is not None:
                            runtime_diagnostics.record_risk(g_hat)
                        # Lagged: round k evidence controls round k+1 gate.
                        lagged_evidence_block = g_hat
                        self._guard_lagged_evidence = g_hat

                # Trajectory recording hook: when a list is provided,
                # snapshot all §5.4'.1 features + metadata for offline
                # RiskHead training.  This lets the collector use the
                # EXACT inference code path (guard-enabled, RankGate
                # active, lagged evidence flowing), eliminating the
                # train-inference distribution mismatch that caused
                # §5.7 to produce <10% when enabled.
                trajectory_record = None
                if trajectory_records is not None and features is not None:
                    trajectory_record = {
                        'block_idx': b_idx,
                        'step': step,
                        'block_start': block_start_pos,
                        'block_end': block_start_pos + block_size,
                        'block_size': block_size,
                        'is_first_round': prev_probs_block is None,
                        'hidden': out.last_hidden_state[
                            :, :block_size].detach().cpu(),
                        'confidence': features.confidence.detach().cpu(),
                        'entropy': features.entropy.detach().cpu(),
                        'js_div': features.js_div.detach().cpu(),
                        'stable': features.stable.detach().cpu(),
                        'current_committed_before': (
                            current_committed_block.clone().cpu()),
                        'committed_history': (
                            ever_committed_block.clone().cpu()),
                        'remask': remask_block.clone().cpu(),
                        'block_commit_corr': (
                            features.block_commit_corr.detach().cpu()),
                        'state': state_k_for_feats.detach().cpu(),
                        'candidates': features.candidates.detach().cpu(),
                        'risk': (
                            None if g_hat is None
                            else g_hat.detach().cpu()),
                        'policy_risk': (
                            None
                            if (not tracer_policy_enabled
                                or round_lagged_evidence is None)
                            else round_lagged_evidence.detach().cpu()),
                        'tracer_router_enabled': bool(
                            tracer_router_enabled),
                        'tracer_policy_enabled': bool(
                            tracer_policy_enabled),
                    }

                # 采样
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    flat_probs = probs.view(-1, probs.shape[-1])
                    sampled_flat = torch.multinomial(flat_probs, 1)
                    sampled_ids = sampled_flat.view(batch_size, block_size)
                    sampled_conf = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
                else:
                    # Greedy: argmax
                    sampled_conf = F.softmax(logits, dim=-1).max(dim=-1).values
                    sampled_ids = logits.argmax(dim=-1)

                # 选择本轮固定的 token：risk-aware commit (§5.7) 或
                # confidence top-k fallback。
                num_to_fix = transfer_schedule[step]
                cumulative_target += int(num_to_fix)
                if features is not None and commit_policy is not None:
                    from src.models.dllm.guard.risk_control import (
                        apply_commit_revision,
                    )
                    decision = commit_policy.select_committed(
                        features=features,
                        target_committed_count=cumulative_target,
                        prev_committed=current_committed_block,
                        # The exact lagged tensor routed into this round's
                        # RankGate is also consumed by the policy. No second
                        # RiskHead call and no same-round evidence leakage.
                        risk_scores=(
                            round_lagged_evidence
                            if tracer_policy_enabled else None),
                        risk_head=None,
                    )
                    if runtime_diagnostics is not None:
                        used_risk = bool(
                            tracer_policy_enabled
                            and round_lagged_evidence is not None
                        )
                        runtime_diagnostics.record_policy_decision(
                            decision, used_risk=used_risk)
                        if not used_risk:
                            if round_lagged_evidence is None:
                                fallback_reason = 'first_round'
                            elif not tracer_policy_enabled:
                                fallback_reason = 'policy_disabled'
                            else:
                                fallback_reason = 'risk_unavailable'
                            runtime_diagnostics.record_confidence_fallback(
                                fallback_reason)
                    block_ids = apply_commit_revision(
                        block_ids=block_ids,
                        sampled_ids=sampled_ids,
                        mask_token_id=mask_token_id,
                        decision=decision,
                    )
                    current_committed_block = decision.committed_mask
                    remask_block = decision.remask_mask
                else:
                    if runtime_diagnostics is not None:
                        runtime_diagnostics.record_confidence_fallback(
                            'guard_unavailable')
                    neg_inf = torch.tensor(float('-inf'), device=device)
                    confidence = torch.where(
                        is_mask, sampled_conf, neg_inf)
                    new_commit_mask = torch.zeros_like(
                        block_ids, dtype=torch.bool)
                    for j in range(batch_size):
                        n_masked = int(is_mask[j].sum().item())
                        k = min(num_to_fix, n_masked)
                        if k > 0:
                            _, topk_idx = torch.topk(confidence[j], k)
                            new_commit_mask[j, topk_idx] = True
                    if runtime_diagnostics is not None:
                        runtime_diagnostics.record_fallback_commits(
                            new_commit_mask, current_committed_block)
                    block_ids = torch.where(
                        new_commit_mask, sampled_ids, block_ids)
                    current_committed_block = (
                        current_committed_block | new_commit_mask)
                    remask_block = torch.zeros_like(
                        current_committed_block)

                ever_committed_block = (
                    ever_committed_block | current_committed_block)

                if trajectory_record is not None:
                    trajectory_record.update({
                        'target_committed_count': cumulative_target,
                        'current_committed_after': (
                            current_committed_block.clone().cpu()),
                        'new_commit': (
                            decision.new_commit_mask.clone().cpu()
                            if features is not None and commit_policy is not None
                            else new_commit_mask.clone().cpu()),
                        'remask_after': remask_block.clone().cpu(),
                    })
                    trajectory_records.append(trajectory_record)

                # Update prev_* for the next round's JS / stability.
                if features is not None:
                    prev_probs_block = F.softmax(
                        logits.float(), dim=-1).detach()
                    prev_candidates_block = features.candidates.detach()
                else:
                    prev_probs_block = F.softmax(
                        logits.float(), dim=-1).detach()
                    prev_candidates_block = sampled_ids.detach()

            # Fixed-NFE invariant: the final cumulative budget must fill the
            # block. Silent EOS substitution would hide an invalid run.
            residual_mask = (block_ids == mask_token_id)
            if residual_mask.any():
                raise RuntimeError(
                    'TRACER fixed-NFE invariant violated: residual mask '
                    f'after {denoising_steps} steps')
            if runtime_diagnostics is not None:
                runtime_diagnostics.record_block_completed()

            all_block_ids.append(block_ids)

            # 检查是否生成了 eos
            if (block_ids == eos_token_id).any():
                finished = True
                break

            # ---------- 4. Clean forward 更新 KV Cache ----------
            clean_embeds = self.llm.get_input_embeddings()(block_ids)
            clean_out = self.llm_model(
                inputs_embeds=clean_embeds,
                position_ids=block_pos_ids,
                attention_mask=torch.zeros(
                    (batch_size, 1, block_size, total_kv_len),
                    dtype=dtype, device=device,
                ),
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
            kv_cache = clean_out.past_key_values

        # ---------- 5. 拼接并截断 ----------
        generated_ids = torch.cat(all_block_ids, dim=1)

        # 截断到第一个 eos
        for i in range(batch_size):
            eos_pos = (generated_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                generated_ids[i, eos_pos[0] + 1:] = eos_token_id

        return generated_ids

    @staticmethod
    def _transfer_schedule(block_size: int, num_steps: int) -> list:
        """线性调度：每步应固定多少个 token。

        总固定数 = block_size，均匀分配到 num_steps 步；
        最后一步兜底余数，保证总和恰好 = block_size。
        """
        if num_steps >= block_size:
            return [1] * block_size + [0] * (num_steps - block_size)
        base = block_size // num_steps
        remainder = block_size % num_steps
        schedule = []
        for s in range(num_steps):
            schedule.append(base + (1 if s < remainder else 0))
        return schedule

    def compute_loss(self, data_dict):
        if self.dllm: data_dict = self.create_dllm_batch(data_dict)
        dllm_stats = {
            'active_targets': 0,
            'legitimate_empty_responses': 0,
            'unexpected_empty_masked_nonempty': 0,
        }
        if self.dllm and self.enforce_nonempty_dllm_targets:
            for batch_data in data_dict.values():
                for name, value in batch_data.get('dllm_stats', {}).items():
                    if name in dllm_stats:
                        dllm_stats[name] += int(value)

        losses = {}
        # actual_start_idx, end_idx = self.find_response_span(data_dict['image2text']['input_ids'][0])
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'image2text' in data_type:
                if (getattr(self, 'mrare_enabled', False)
                        or getattr(self, 'mrfc_enabled', False)):
                    result = self._image2text_loss_result(batch_data)
                    losses[f'loss_{data_type}'] = (
                        result.base_loss
                        * self.loss_weights.get(data_type, 1.0))
                if getattr(self, 'mrare_enabled', False):
                    auxiliary = self._mrare_auxiliary_losses(
                        batch_data,
                        result.last_hidden_state,
                        result.response_mask,
                        result.loss_mask,
                    )
                    losses['loss_mrare_positive'] = (
                        auxiliary['weighted_positive'])
                    losses['loss_mrare_rank'] = auxiliary['weighted_rank']
                    if self.mrare_distill_enabled:
                        losses['loss_mrare_distill'] = (
                            auxiliary['weighted_distill'])
                    for name in self._mrare_diagnostic_names():
                        losses[name] = auxiliary[name]
                if getattr(self, 'mrfc_enabled', False):
                    mrfc = self._mrfc_auxiliary_losses(
                        batch_data, result.last_hidden_state)
                    losses['loss_mrfc_pair'] = mrfc['weighted_pair']
                    losses['loss_mrfc_distill'] = mrfc['weighted_distill']
                    for name in (
                        'mrfc_active_pairs',
                        'mrfc_score_gap',
                        'mrfc_pair_raw',
                        'mrfc_distill_raw',
                    ):
                        losses[name] = mrfc[name]
                if (not getattr(self, 'mrare_enabled', False)
                        and not getattr(self, 'mrfc_enabled', False)):
                    # Preserve the stable image-to-text path exactly when
                    # both auxiliary methods are disabled, including its RNG
                    # consumption.
                    loss = self.image2text_loss(batch_data)
                    losses[f'loss_{data_type}'] = (
                        loss * self.loss_weights.get(data_type, 1.0))
            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'depth' in data_type:
                # Depth reconstruction (source image -> target depth map)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'sketch' in data_type:
                # Sketch generation (source image -> target sketch)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'mask' in data_type:
                # Mask segmentation (source image -> target mask)
                loss = self.edit_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            elif 'joint' in data_type:
                loss_sft, loss_depth = self.joint_loss(batch_data)
                # Weight joint loss parts using individual weights if available, or just sum
                w_sft = self.loss_weights.get('image2text', 1.0)
                w_depth = self.loss_weights.get('edit', 1.0)
                # Use 'joint' weight as global scaler for this branch
                w_joint = self.loss_weights.get('joint', 1.0)
                
                losses[f'loss_{data_type}_sft'] = loss_sft * w_sft * w_joint
                losses[f'loss_{data_type}_depth'] = loss_depth * w_depth * w_joint
            else:
                raise NotImplementedError(f"Unknown data type: {data_type}")

        if self.dllm and self.enforce_nonempty_dllm_targets:
            reference = next(iter(losses.values()), self.harmon_mask_token_id)
            for name, value in dllm_stats.items():
                losses[f'dllm_{name}'] = reference.detach().new_tensor(
                    float(value), dtype=torch.float32
                )
        return losses
