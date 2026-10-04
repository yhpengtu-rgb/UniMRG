#!/usr/bin/env python3
"""Generate GenEval / DPG-Bench images with the Harmon pixel branch.

Run from any directory. Prompt files live beside the existing training data.
This script generates images in the layouts required by the bundled scorers;
it does not turn generated images into benchmark scores.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


HARMON_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('/nvmedata/xiexu/data')
DEFAULT_CHECKPOINT = DATA_ROOT / 'uni/harmon_1.5b.pth'
PROMPTS = {
    'geneval': DATA_ROOT / 'Benchmark/geneval/evaluation_metadata.jsonl',
    'dpgbench': DATA_ROOT / 'Benchmark/dpg_bench/prompts.json',
}
DEFAULT_SAMPLES = {'geneval': 12, 'dpgbench': 4}


def load_prompts(benchmark, path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {benchmark} prompts: {path}')
    if benchmark == 'geneval':
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows or any(not isinstance(row.get('prompt'), str) or not row['prompt'] for row in rows):
            raise ValueError(f'Invalid GenEval prompts: {path}')
        return rows
    with path.open(encoding='utf-8') as stream:
        rows = json.load(stream)
    if not isinstance(rows, dict) or not rows or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in rows.items()):
        raise ValueError(f'Invalid DPG-Bench prompts: {path}')
    if len({Path(key).stem for key in rows}) != len(rows):
        raise ValueError('DPG-Bench prompt stems collide')
    return list(rows.items())


def output_paths(benchmark, output_dir, index, row, count):
    if benchmark == 'geneval':
        prompt_dir = output_dir / f'{index:05d}'
        return [prompt_dir / 'samples' / f'{sample:05d}.png' for sample in range(count)]
    key, _ = row
    stem = Path(key).stem
    return [output_dir / f'{stem}_{sample}.jpg' for sample in range(count)]


def complete_image(path, image_size):
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.size != (image_size, image_size):
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=PROMPTS, default='geneval')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--config', type=Path,
                        default=HARMON_ROOT / 'configs/models/qwen2_5_1_5b_kl16_mar_h.py')
    parser.add_argument('--prompts', type=Path, help='Override the prompt file')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Run only the first N prompts (smoke test)')
    parser.add_argument('--samples-per-prompt', type=int)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Images generated together on each GPU (default: 1)')
    parser.add_argument('--resume-with-new-batch-size', action='store_true',
                        help='Resume an existing output directory with a different batch size')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-iter', type=int, default=64)
    parser.add_argument('--cfg', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without loading the model')
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.image_size < 16 or args.image_size % 16:
        parser.error('--image-size must be a positive multiple of 16')
    if args.num_iter < 1:
        parser.error('--num-iter must be positive')
    if args.cfg <= 0:
        parser.error('--cfg must be positive')
    count = (DEFAULT_SAMPLES[args.benchmark] if args.samples_per_prompt is None
             else args.samples_per_prompt)
    if count < 1:
        parser.error('--samples-per-prompt must be positive')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    prompt_path = args.prompts or PROMPTS[args.benchmark]
    rows = load_prompts(args.benchmark, prompt_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    output_dir = args.output_dir or (DATA_ROOT / 'Benchmark/pixel_results' /
                                     f'harmon_1.5b_{args.benchmark}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {args.checkpoint}')
    if not args.config.is_file():
        raise FileNotFoundError(f'Missing model config: {args.config}')
    print(f'Benchmark: {args.benchmark}; prompts: {len(rows)}; samples/prompt: {count}', flush=True)
    print(f'Prompt file: {prompt_path}\nCheckpoint: {args.checkpoint}\nOutput: {output_dir}', flush=True)
    if args.dry_run:
        print(f'First output: {output_paths(args.benchmark, output_dir, 0, rows[0], count)[0]}')
        return

    import torch
    import torch.distributed as dist
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available():
        raise RuntimeError('Harmon pixel evaluation requires CUDA')
    visible_gpus = torch.cuda.device_count()
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f'LOCAL_RANK={local_rank} but only {visible_gpus} CUDA GPU(s) are visible. '
            'Set --nproc_per_node to the number of GPUs in CUDA_VISIBLE_DEVICES.')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world_size > 1:
        dist.init_process_group('nccl')

    run_spec = {
        'benchmark': args.benchmark,
        'checkpoint': str(args.checkpoint.resolve()),
        'prompts_sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        'limit': args.limit,
        'samples_per_prompt': count,
        'image_size': args.image_size,
        'num_iter': args.num_iter,
        'cfg': args.cfg,
        'seed': args.seed,
        'world_size': world_size,
    }
    # Keep batch-size 1 manifests compatible with runs already in progress.
    if args.batch_size != 1:
        run_spec['batch_size'] = args.batch_size
    # Keep the manifest beside the image directory: DPG-Bench's scorer tries
    # every file inside the image directory as an image.
    manifest = output_dir.parent / f'{output_dir.name}.pixel_run.json'
    if rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not manifest.is_file():
                raise RuntimeError(f'Output directory has a different run: {output_dir}. '
                                   'Choose another --output-dir.')
            previous = json.loads(manifest.read_text(encoding='utf-8'))
            previous_core = {key: value for key, value in previous.items()
                             if key not in ('batch_size', 'batch_size_history')}
            current_core = {key: value for key, value in run_spec.items()
                            if key not in ('batch_size', 'batch_size_history')}
            previous_batch = previous.get('batch_size', 1)
            if previous_core != current_core or (previous_batch != args.batch_size and
                                                 not args.resume_with_new_batch_size):
                raise RuntimeError(f'Output directory has a different run: {output_dir}. '
                                   'Choose another --output-dir, or use '
                                   '--resume-with-new-batch-size for a batch change.')
            history = previous.get('batch_size_history', [previous_batch])
            if previous_batch != args.batch_size:
                run_spec['batch_size_history'] = history + [args.batch_size]
                print('Resuming with a new batch size; existing images are kept. '
                      'Random samples for newly generated images will differ.', flush=True)
            elif 'batch_size_history' in previous:
                run_spec['batch_size_history'] = history
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(run_spec, indent=2) + '\n', encoding='utf-8')
    if world_size > 1:
        dist.barrier()

    from mmengine.config import Config
    import sys
    sys.path.insert(0, str(HARMON_ROOT))
    from src.builder import BUILDER
    from geneval import sample_images

    config = Config.fromfile(str(args.config))
    # The installed flash_attn unpad_input returns five values, while this
    # Transformers release expects four. Match the existing inference configs.
    config.model.llm.attn_implementation = 'sdpa'
    model = BUILDER.build(config.model).eval().to(device)
    model = model.to(model.dtype)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a state dict')
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    matched = len(set(state).intersection(model.state_dict()))
    if matched == 0:
        raise RuntimeError('Checkpoint has no model keys in common with the config')
    info = model.load_state_dict(state, strict=False)
    print(f'Loaded {matched} keys; missing={len(info.missing_keys)}, '
          f'unexpected={len(info.unexpected_keys)}', flush=True)
    if info.missing_keys:
        print('Missing key groups:', dict(Counter(
            key.split('.', 1)[0] for key in info.missing_keys)), flush=True)
    if info.unexpected_keys:
        print('Unexpected key groups:', dict(Counter(
            key.split('.', 1)[0] for key in info.unexpected_keys)), flush=True)
    if info.unexpected_keys or any(not key.startswith('vae.') for key in info.missing_keys):
        raise RuntimeError('Checkpoint does not match the Harmon pixel model; '
                           'only VAE keys may be absent (VAE uses its own checkpoint)')
    del state

    # Divide only unfinished prompts. A resumed run may have work left on
    # just one parity; static index sharding would leave the other GPU idle.
    unfinished = [index for index, row in enumerate(rows)
                  if any(not complete_image(path, args.image_size) for path in
                         output_paths(args.benchmark, output_dir, index, row, count))]
    print(f'[GPU {local_rank}] {len(unfinished[rank::world_size])} unfinished prompts',
          flush=True)
    for index in unfinished[rank::world_size]:
        row = rows[index]
        paths = output_paths(args.benchmark, output_dir, index, row, count)
        if args.benchmark == 'geneval':
            metadata_path = output_dir / f'{index:05d}' / 'metadata.jsonl'
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            prompt = row['prompt']
        else:
            prompt = row[1]
        full_prompt = f'Generate an image: {prompt}'
        pending = [(sample_index, path) for sample_index, path in enumerate(paths)
                   if not complete_image(path, args.image_size)]
        if not pending:
            continue
        conditions = model.prepare_text_conditions(full_prompt, 'Generate an image.')
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            batch_size = len(batch)
            seed_all(args.seed + index * 1000 + batch[0][0], torch)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                results = sample_images(
                    model, conditions, batch_size, image_size=args.image_size,
                    num_iter=args.num_iter, cfg=args.cfg, progress=False)
            for (_, path), result in zip(batch, results):
                image = (result.float().clamp(-1, 1).add(1).mul(127.5)
                         .to(torch.uint8).cpu().permute(1, 2, 0).numpy())
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
            print(f'[GPU {local_rank}: {index + 1}/{len(rows)}] '
                  f'{batch_size} images in {time.perf_counter() - started:.1f}s '
                  f'peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB '
                  f'({batch[0][1].parent})', flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print('Image generation complete.', flush=True)


if __name__ == '__main__':
    main()
