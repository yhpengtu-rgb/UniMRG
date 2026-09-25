import inspect
import torch.nn as nn
from typing import List, Optional, Union
from mmengine.optim import DefaultOptimWrapperConstructor, OptimWrapper
from mmengine.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                               OPTIMIZERS)


def add_weight_decay(model, weight_decay=1e-5, skip_list=(),
                     guard_lr_multiplier=None, guard_weight_decay=None):
    """Build optimizer param groups with weight-decay split.

    Parameters
    ----------
    model:
        The model to optimise.  Must have ``named_parameters``.
    weight_decay:
        Weight decay applied to the "decay" group (matmul weights).
    skip_list:
        Names to force into the no-decay group.
    guard_lr_multiplier:
        If not ``None``, parameters whose name contains
        ``"guard_rank_gate"`` (the §5.6 RankGate) are split into their own
        group with ``lr = base_lr * guard_lr_multiplier``.  This lets the
        newly-initialised RankGate train faster than the LoRA without
        disturbing the LoRA schedule.  ``None`` disables the split (legacy
        behaviour).
    guard_weight_decay:
        Weight decay for the RankGate group.  Defaults to 0 (RankGate is
        small and benefits from no decay).
    """
    decay = []
    no_decay = []
    guard_decay = []
    guard_no_decay = []
    guard_kw = 'guard_rank_gate'
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        is_guard = guard_kw in name
        if is_guard and guard_lr_multiplier is not None:
            # RankGate gets its own group (no weight decay by default).
            if (len(param.shape) == 1 or name.endswith(".bias")
                    or name in skip_list):
                guard_no_decay.append(param)
            else:
                guard_decay.append(param)
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    print(f"num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters")

    groups = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

    if guard_lr_multiplier is not None and (guard_decay or guard_no_decay):
        guard_params = guard_decay + guard_no_decay
        num_guard = sum(p.numel() for p in guard_params)
        print(f"num guard_rank_gate parameter tensors: {len(guard_params)},"
              f" with {num_guard:,} parameters (lr x{guard_lr_multiplier})")
        groups.append({
            'params': guard_params,
            'weight_decay': 0.0 if guard_weight_decay is None
            else guard_weight_decay,
            # NOTE: the actual ``lr`` multiplier is applied in
            # ``MAROptimWrapperConstructor`` where the base lr is known.
            '_is_guard_group': True,
        })
    return groups


class MAROptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __call__(self, model: nn.Module) -> OptimWrapper:
        if hasattr(model, 'module'):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault('type', 'OptimWrapper')
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cls = self.optimizer_cfg['type']
        # Optimizer like HybridAdam in colossalai requires the argument name
        # `model_params` rather than `params`. Here we get the first argument
        # name and fill it with the model parameters.
        if isinstance(optimizer_cls, str):
            with OPTIMIZERS.switch_scope_and_registry(None) as registry:
                optimizer_cls = registry.get(self.optimizer_cfg['type'])
        fisrt_arg_name = next(
            iter(inspect.signature(optimizer_cls).parameters))

        # §5.6 RankGate LR multiplier (optional).  When set, RankGate
        # parameters get their own param group with
        # ``lr = base_lr * guard_lr_multiplier`` so the newly-initialised
        # gate can train faster than the LoRA schedule.
        guard_lr_multiplier = optimizer_cfg.pop(
            'guard_lr_multiplier', None)
        guard_weight_decay = optimizer_cfg.pop(
            'guard_weight_decay', None)
        weight_decay = optimizer_cfg.pop('weight_decay', 0)
        param_groups = add_weight_decay(
            model, weight_decay,
            guard_lr_multiplier=guard_lr_multiplier,
            guard_weight_decay=guard_weight_decay)
        base_lr = optimizer_cfg.get('lr', 1e-5)
        for grp in param_groups:
            if grp.pop('_is_guard_group', False):
                grp['lr'] = base_lr * guard_lr_multiplier
        optimizer_cfg[fisrt_arg_name] = param_groups
        optimizer = OPTIMIZERS.build(optimizer_cfg)

        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
