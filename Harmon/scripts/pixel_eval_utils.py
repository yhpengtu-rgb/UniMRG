"""Load full / LoRA pixel DLLM checkpoints without changing their trained structure."""

from collections import Counter
from pathlib import Path
import sys


def sample_dllm_images(model, conditions, batch_size, *, image_size, num_iter, cfg,
                       progress=False):
    """Batch text conditions without routing through the legacy MAR sampler."""
    import torch

    if not getattr(model, 'pixel_dllm', False):
        raise RuntimeError('The model must enable pixel_dllm; no AR fallback')
    sampler = getattr(model, 'sample_pixel_dllm_t2i', None)
    if not callable(sampler):
        raise RuntimeError('The model lacks the text-to-image DLLM sampler; no AR fallback')
    ids = conditions['input_ids']
    valid = conditions['attention_mask']
    if cfg != 1.0:
        if ids.shape[0] != 2:
            raise ValueError('CFG requires positive and weak text conditions')
        ids = torch.cat((ids[:1].expand(batch_size, -1), ids[1:2].expand(batch_size, -1)))
        valid = torch.cat((valid[:1].expand(batch_size, -1), valid[1:2].expand(batch_size, -1)))
    else:
        ids = ids[:1].expand(batch_size, -1)
        valid = valid[:1].expand(batch_size, -1)
    return sampler(input_ids=ids, attention_mask=valid,
                   image_shape=(image_size // 16, image_size // 16),
                   num_iter=num_iter, cfg=cfg, progress=progress)


def load_checkpoint_state(checkpoint, torch):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        model_states = checkpoint / 'mp_rank_00_model_states.pt'
        if model_states.is_file():
            # Harmon trains with ZeRO-2: model weights are complete in this file.
            # Loading optimizer shards to reconstruct them is unnecessary.
            state = torch.load(model_states, map_location='cpu', weights_only=False)
        else:
            from xtuner.model.utils import guess_load_checkpoint
            state = guess_load_checkpoint(str(checkpoint))
    else:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    elif isinstance(state, dict) and 'module' in state:
        state = state['module']
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a state dict')
    return {key.removeprefix('module.'): value for key, value in state.items()}


def load_pixel_weights(model, state, *, pretrained_loaded=False):
    matched = len(set(state).intersection(model.state_dict()))
    if matched == 0:
        raise RuntimeError('Checkpoint has no model keys in common with the config')
    info = model.load_state_dict(state, strict=False)
    print(f'Loaded {matched} keys; missing={len(info.missing_keys)}, '
          f'unexpected={len(info.unexpected_keys)}', flush=True)
    if info.missing_keys:
        print('Missing key groups:', dict(Counter(
            key.split('.', 1)[0] for key in info.missing_keys)), flush=True)
    # ZeRO can omit frozen LoRA base weights; the constructor loads those before
    # attaching adapters. Include aliases because Qwen ties embeddings and lm_head.
    frozen = {name for name, parameter in model.named_parameters(remove_duplicate=False)
              if not parameter.requires_grad} if pretrained_loaded else set()
    invalid_missing = [key for key in info.missing_keys
                       if not key.startswith('vae.') and key not in frozen]
    if info.unexpected_keys or invalid_missing:
        raise RuntimeError('Checkpoint does not match the Harmon pixel model; '
                           f'missing required keys: {invalid_missing[:10]}; '
                           f'unexpected keys: {info.unexpected_keys[:10]}')
    return info


def load_model(config_path, checkpoint, device):
    import torch
    from mmengine.config import Config

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.builder import BUILDER

    config = Config.fromfile(str(config_path))
    # Reuse the exact training config. Only turn off activation checkpointing
    # and select the inference attention implementation.
    config.model.gradient_checkpointing = False
    config.model.llm.attn_implementation = 'sdpa'
    model = BUILDER.build(config.model).eval()
    if not getattr(model, 'pixel_dllm', False):
        raise RuntimeError('The config must enable pixel_dllm')
    model = model.to(dtype=model.dtype)
    state = load_checkpoint_state(checkpoint, torch)
    load_pixel_weights(model, state,
                       pretrained_loaded=bool(config.model.get('pretrained_pth')))
    return model.to(device)
