#!/usr/bin/env python3
"""Collect self-generated trajectory evidence for RiskHead retraining.

Unlike ``collect_trajectory_evidence.py`` (which uses
``run_dllm_with_trajectory`` with ``guard_enabled=False`` and
``block_size=32``), this script calls ``model.generate_dllm`` directly with
``guard_enabled=True`` and the **same** ``block_size`` / ``denoising_steps``
as inference.  This eliminates two sources of train-inference distribution
mismatch that caused §5.7 to produce <10% when enabled:

1. **No RankGate during collection**: the old script ran without guard, so
   trajectory features never saw RankGate modulation.  At inference the
   RankGate is active, making the features OOD for the RiskHead.
2. **Wrong block size**: the old script used ``block_size=32`` while
   inference uses ``block_size=4``.  Trajectory features (especially
   ``committed_history``, ``block_commit_corr``, JS divergence) have
   completely different distributions at different block sizes.

The model generates tokens freely through the dLLM denoising loop (the
exact inference code path).  GT is only used to (1) size the generation
(``max_new_tokens = turn_len``) and (2) label wrong commits
(``Y_i^k = 1[ŷ_i^k ≠ y_i^*]``).

Usage:
    python scripts/collect_self_trajectory.py \\
        --risk-fit /nvmedata/xiexu/data/uni/guard_exclusions/risk-fit.json \\
        --checkpoint /nvmedata/xiexu/uni/work_dirs/guard_training/runs/merged_stable_traj_p2/iter_20000.pth \\
        --config configs/examples/UniMRG_dllm_lora_guard_trajectory_infer.py \\
        --output-dir /nvmedata/xiexu/data/uni/guard_trajectory_self_v1 \\
        --block-size 4 --denoising-steps 4 --gpu 0
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HARMON_ROOT = Path(__file__).resolve().parents[1]
if str(HARMON_ROOT) not in sys.path:
    sys.path.insert(0, str(HARMON_ROOT))

# Reuse helpers from the existing collector.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from collect_trajectory_evidence import (  # noqa: E402
    load_state_dict_agnostic,
    normalize_conversations,
    build_inputs_embeds,
    encode_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--risk-fit', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--block-size', type=int, default=4,
                        help='Block size (must match inference, default 4)')
    parser.add_argument('--denoising-steps', type=int, default=4,
                        help='Denoising steps (must match inference, default 4)')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--image-root', type=str,
                        default=os.path.join(
                            os.environ.get(
                                'HARMON_DATA_ROOT',
                                '/nvmedata/xiexu/data/'
                                'LLaVA-Instruct-150K-UniMRG'),
                            'tuning_data'))
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_guarded_model(config_path, checkpoint_path, device):
    """Load HarmonDev with guard_enabled=True (RankGate + RiskHead active)."""
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    model_cfg = dict(cfg.model)
    # dLLM inference mode.
    model_cfg['dllm'] = True
    model_cfg['gradient_checkpointing'] = False
    # Guard must be enabled so the RankGate is installed and the
    # trajectory-features code path inside generate_dllm activates.
    model_cfg['guard_enabled'] = True

    from src.models.harmon_dev import HarmonDev
    model_cfg = {k: v for k, v in model_cfg.items() if k != 'type'}
    model = HarmonDev(**model_cfg)
    model = model.to(device)
    model.eval()

    from src.models.harmon_dev import _load_pretrained_weights
    _load_pretrained_weights(model, checkpoint_path,
                             required_key_groups=('llm.', 'proj_in.'))
    print(f'Guard enabled: {model.guard_enabled}, '
          f'RiskHead loaded: {model._guard_risk_head is not None}')
    return model, cfg


def collect_turn_trajectory(model, prefix_embeds, gt_response_ids,
                            block_size, denoising_steps, temperature,
                            mask_token_id, eos_token_id):
    """Run generate_dllm for one turn and label wrong commits with GT.

    Returns a list of per-round record dicts in the same format as
    ``collect_trajectory_evidence.run_dllm_with_trajectory``, including
    the ``wrong_commit`` field.
    """
    turn_len = gt_response_ids.shape[1]
    records = []
    # DEBUG: verify guard components are active so the trajectory hook fires.
    _gm = getattr(model, 'guard_lora_manager', None)
    _rh = getattr(model, '_guard_risk_head', None)
    _guard_has_risk = _gm is not None and _rh is not None
    if _gm is None:
        print('  WARN: guard manager is missing; trajectory hook will not fire')
    elif not _guard_has_risk:
        print('  INFO: collecting bootstrap trajectory without a RiskHead; '
              'RankGate remains neutral and features are still recorded')
    # DEBUG: print device of key tensors + risk head params
    _dev = prefix_embeds.device
    if _rh is not None:
        _rh_dev = next(_rh.parameters()).device
        print(f'  DEBUG: prefix_embeds on {_dev}, risk_head params on {_rh_dev}')
    import traceback as _tb
    try:
        model.generate_dllm(
            inputs_embeds=prefix_embeds,
            max_new_tokens=turn_len,
            block_size=block_size,
            denoising_steps=denoising_steps,
            temperature=temperature,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            tracer_router_enabled=True,
            tracer_policy_enabled=False,  # router-only audit collection cell
            trajectory_records=records,
        )
    except Exception as _e:
        import traceback as _tb2
        _full = _tb2.format_exc()
        print(f'  EXCEPTION in generate_dllm: {_e}')
        print(_full)
        debug_path = Path(os.environ.get(
            'TRACER_DEBUG_LOG', 'guard_self_debug.log'))
        with debug_path.open('w', encoding='utf-8') as _f:
            _f.write(f'EXCEPTION: {_e}\n{_full}\n')
            _f.write(f'prefix_embeds device: {_dev}\n')
            if _rh is not None:
                _f.write(f'risk_head param device: {next(_rh.parameters()).device}\n')
        raise
    if not records:
        print(f'  WARN: generate_dllm produced 0 records for turn '
              f'(turn_len={turn_len}, guard_has_risk={_guard_has_risk})')

    # Label wrong_commit and truncate to GT length.
    labeled = []
    for r in records:
        b_idx = r['block_idx']
        resp_start = b_idx * block_size
        resp_end = min(resp_start + block_size, turn_len)
        cur_len = resp_end - resp_start
        if cur_len <= 0:
            continue
        gt_block = gt_response_ids[0, resp_start:resp_end].cpu()  # [cur_len]
        candidates = r['candidates'][0, :cur_len]  # [cur_len] (already CPU from hook)
        r['wrong_commit'] = (
            candidates != gt_block).float().unsqueeze(0)  # [1, cur_len]
        # Truncate all per-token tensors to cur_len.
        for key in ('hidden', 'confidence', 'entropy', 'js_div', 'stable',
                    'committed_history', 'remask', 'block_commit_corr',
                    'candidates'):
            r[key] = r[key][:, :cur_len] if r[key].dim() == 3 \
                else r[key][:, :cur_len]
        # state is [1, 3] — no truncation.
        labeled.append(r)
    return labeled


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.risk_fit, 'r') as f:
        risk_fit = json.load(f)
    if args.max_samples:
        risk_fit = risk_fit[:args.max_samples]
    print(f'Loaded {len(risk_fit)} risk-fit samples')

    print(f'Loading guarded model from {args.checkpoint}...')
    model, cfg = load_guarded_model(args.config, args.checkpoint, device)
    print(f'hidden_size={model.llm.config.hidden_size}')

    from transformers import AutoTokenizer
    tokenizer_path = cfg.model.llm.get(
        'pretrained_model_name_or_path',
        '/nvmedata/xiexu/data/uni/Qwen2.5-1.5B-Instruct',
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    HARMON_MASK_TOKEN = '<|dllm_mask|>'
    if HARMON_MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([HARMON_MASK_TOKEN])
    mask_token_id = int(tokenizer.convert_tokens_to_ids(HARMON_MASK_TOKEN))
    eos_token_id = tokenizer.eos_token_id
    print(f'Mask token id: {mask_token_id}, EOS id: {eos_token_id}')

    from xtuner.utils import PROMPT_TEMPLATE
    template = PROMPT_TEMPLATE['qwen_chat']

    from src.datasets.understanding.llava_datasets import MARProcessor
    image_processor = MARProcessor(image_size=512)

    all_samples = []
    n_processed = 0
    n_skipped = 0
    total_turns = 0

    for idx, sample in enumerate(risk_fit):
        if idx % 10 == 0:
            print(f'Processing sample {idx}/{len(risk_fit)}...')

        try:
            image_path = os.path.join(args.image_root, sample['image'])
            if not os.path.exists(image_path):
                print(f'  SKIP: image not found: {image_path}')
                n_skipped += 1
                continue
            image = Image.open(image_path).convert('RGB')
            pixel_values = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]

            input_ids_list, labels_list, gpt_turns = normalize_conversations(
                sample, tokenizer, template)
            if not gpt_turns:
                n_skipped += 1
                continue

            input_ids = torch.tensor(
                [input_ids_list], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            labels = torch.tensor(
                [labels_list], dtype=torch.long, device=device)
            vocab_size = model.llm.config.vocab_size
            labels[labels >= vocab_size] = -100

            z_enc = encode_image(model, pixel_values, device)
            inputs_embeds, input_ids_expanded, attn_expanded = (
                build_inputs_embeds(
                    model, input_ids, z_enc, attention_mask, device))

            n_image_tokens = z_enc.shape[1]
            shift = n_image_tokens - 1

            sample_records = []
            sample_turn_lens = []

            for turn_idx, (resp_start, resp_end) in enumerate(gpt_turns):
                if resp_end <= resp_start:
                    continue
                resp_start_exp = resp_start + shift
                prefix_embeds = inputs_embeds[:, :resp_start_exp, :]

                gt_response_ids = (
                    labels[0, resp_start:resp_end]
                    .clamp(min=0).unsqueeze(0))

                turn_records = collect_turn_trajectory(
                    model=model,
                    prefix_embeds=prefix_embeds,
                    gt_response_ids=gt_response_ids,
                    block_size=args.block_size,
                    denoising_steps=args.denoising_steps,
                    temperature=args.temperature,
                    mask_token_id=mask_token_id,
                    eos_token_id=eos_token_id,
                )
                if not turn_records:
                    continue
                for r in turn_records:
                    r['turn_idx'] = turn_idx
                sample_records.extend(turn_records)
                sample_turn_lens.append(int(resp_end - resp_start))
                total_turns += 1

            if not sample_records:
                n_skipped += 1
                continue

            all_samples.append({
                'sample_idx': idx,
                'sample_id': sample.get('id', str(idx)),
                'image': sample.get('image', ''),
                'n_turns': len(sample_turn_lens),
                'turn_lens': sample_turn_lens,
                'records': sample_records,
            })
            n_processed += 1

        except Exception as e:
            import traceback as _tb_full
            _full_tb = _tb_full.format_exc()
            print(f'  ERROR on sample {idx}: {e}')
            print(_full_tb)
            with open('/home/xiexu/code/UniMRG/guard_self_debug.log', 'a') as _f:
                _f.write(f'=== sample {idx} ===\n')
                _f.write(f'ERROR: {e}\n{_full_tb}\n\n')
            n_skipped += 1
            continue

    print(f'\nProcessed: {n_processed} samples, {total_turns} turns, '
          f'skipped: {n_skipped}')

    out_path = output_dir / 'trajectory_evidence.pt'
    torch.save({
        'samples': all_samples,
        'config': {
            'block_size': args.block_size,
            'denoising_steps': args.denoising_steps,
            'temperature': args.temperature,
            'seed': args.seed,
            'checkpoint': args.checkpoint,
            'risk_fit': args.risk_fit,
            'collection_mode': 'self_generated_guarded',
        },
    }, out_path)
    print(f'Saved {len(all_samples)} samples to {out_path}')

    # Summary statistics.
    total_rounds = sum(len(s['records']) for s in all_samples)
    total_tokens = sum(
        r['wrong_commit'].numel() for s in all_samples for r in s['records'])
    wrong_commit_rate = float(np.mean([
        r['wrong_commit'].mean().item()
        for s in all_samples for r in s['records']
    ]))
    turn_rate_by_len_bucket = collections.defaultdict(list)
    for s in all_samples:
        for r in s['records']:
            bucket = min(r['block_size'], 32)
            turn_rate_by_len_bucket[bucket].append(
                r['wrong_commit'].mean().item())

    summary = {
        'n_samples': len(all_samples),
        'n_skipped': n_skipped,
        'total_turns': total_turns,
        'total_rounds': total_rounds,
        'total_tokens': total_tokens,
        'wrong_commit_rate': wrong_commit_rate,
        'wrong_commit_rate_by_block_size': {
            str(k): float(np.mean(v))
            for k, v in sorted(turn_rate_by_len_bucket.items())
        },
        'block_size': args.block_size,
        'denoising_steps': args.denoising_steps,
    }
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrong-commit rate: {wrong_commit_rate:.4f} '
          f'({total_tokens} tokens, {total_rounds} rounds)')
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
