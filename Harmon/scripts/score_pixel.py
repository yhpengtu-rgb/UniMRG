#!/usr/bin/env python3
"""Score generated pixel images and update the experiment record."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from eval_pixel import PROMPTS, complete_image, load_prompts, output_paths

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / 'docs/experiments/PIXEL_EVAL_RESULTS_20261002.md'
DATA = Path('/nvmedata/xiexu/data/Benchmark')


def _record_score_row(path, experiment, benchmark=None):
    row_names = {experiment}
    inferred_benchmark = None
    if experiment == 'base-dpg-b1':
        row_names.add('base')
        inferred_benchmark = 'dpgbench'
    else:
        for name in ('geneval', 'dpgbench'):
            suffix = '-' + name
            if experiment.endswith(suffix):
                row_names.add(experiment[:-len(suffix)])
                inferred_benchmark = name
                break
    if benchmark and inferred_benchmark and benchmark != inferred_benchmark:
        raise ValueError(f'Benchmark differs from experiment {experiment}')
    benchmark = benchmark or inferred_benchmark
    lines = path.read_text(encoding='utf-8').splitlines()
    columns = {}
    for i, line in enumerate(lines):
        if not line.startswith('|'):
            columns = {}
            continue
        cells = [x.strip() for x in line.split('|')[1:-1]]
        if any(name in cells for name in ('分数（%）', 'Score (%)',
                                         'GenEval（%）', 'DPGBench（%）')):
            columns = {name: index for index, name in enumerate(cells)}
            continue
        if cells and cells[0] in row_names:
            score_column = columns.get('分数（%）', columns.get('Score (%)'))
            if score_column is None:
                name = {'geneval': 'GenEval（%）', 'dpgbench': 'DPGBench（%）'}.get(benchmark)
                score_column = columns.get(name)
            if score_column is None or len(cells) != len(columns):
                raise ValueError(f'Invalid score table for {experiment} in {path}')
            return lines, i, cells, columns, score_column
    raise ValueError(f'Experiment {experiment} is missing from {path}')


def _update_record(path, experiment, summary):
    lines, index, cells, columns, score_column = _record_score_row(
        path, experiment, summary.get('benchmark'))
    for name in ('评分图片/预期', 'Scored/expected'):
        if name in columns:
            cells[columns[name]] = f"{summary['scored_images']}/{summary['expected_images']}"
    for name in ('状态', 'Status'):
        if name in columns:
            cells[columns[name]] = '完整' if summary['complete'] else '部分样本'
    cells[score_column] = f"{summary['score_percent']:.2f}" + (
        '（部分样本）' if not summary['complete'] else '')
    lines[index] = '| ' + ' | '.join(cells) + ' |'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def update_record(path, experiment, summary):
    lock_path = DATA / 'pixel_scores/record.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _update_record(path, experiment, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True, choices=['geneval', 'dpgbench'])
    parser.add_argument('--image-dir', required=True, type=Path)
    parser.add_argument('--experiment', required=True,
                        help='Score run name, e.g. base-geneval / base-dpg-b1 / EXPERIMENT-dpgbench')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--record', type=Path, default=RECORD)
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly score available images as partial results')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--geneval-model-path', type=Path, default=Path('/nvmedata/xiexu/data/models/geneval'))
    parser.add_argument('--geneval-config', type=Path)
    parser.add_argument('--dpg-model', default='damo/mplug_visual-question-answering_coco_large_en', help='ModelScope model ID or local model directory')
    parser.add_argument('--csv', type=Path, default=DATA / 'dpg_bench/dpg_bench.csv')
    args = parser.parse_args()
    if '/' in args.experiment or args.experiment in {'.', '..'}:
        parser.error('--experiment must be a directory name')
    args.image_dir = args.image_dir.resolve()
    manifest = json.loads(args.image_dir.with_name(args.image_dir.name + '.pixel_run.json').read_text())
    if manifest['benchmark'] != args.benchmark:
        parser.error('Benchmark differs from the generation manifest')
    prompt_path = PROMPTS[args.benchmark]
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != manifest['prompts_sha256']:
        parser.error('Prompt file differs from the generation manifest')
    rows = load_prompts(args.benchmark, prompt_path)
    full_prompt_count = len(rows)
    if manifest.get('limit') is not None:
        rows = rows[:manifest['limit']]
    images = []
    complete_prompts = 0
    for index, row in enumerate(rows):
        paths = output_paths(args.benchmark, args.image_dir, index, row, manifest['samples_per_prompt'])
        valid = [p for p in paths if complete_image(p, manifest['image_size'])]
        if len(valid) == len(paths):
            complete_prompts += 1
        if args.benchmark == 'geneval' and valid:
            metadata = json.loads((args.image_dir / f'{index:05d}' / 'metadata.jsonl').read_text())
            if metadata != row:
                parser.error(f'Incorrect metadata for prompt {index}')
        images.extend(valid)
    expected = len(rows) * manifest['samples_per_prompt']
    if not images:
        parser.error('No valid images')
    if len(images) != expected and not args.allow_incomplete:
        parser.error(f'{len(images)}/{expected} valid images; use --allow-incomplete for a partial score')
    output = args.output_dir or DATA / 'pixel_scores' / args.experiment
    print(f'{args.experiment}: {len(images)}/{expected} images; scores -> {output}', flush=True)
    if args.dry_run:
        return
    try:
        _record_score_row(args.record, args.experiment, args.benchmark)
    except ValueError as error:
        parser.error(str(error))
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run one scoring process; select its GPU with CUDA_VISIBLE_DEVICES')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA is required for scoring')
    torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=True)
    # A second launch waits and then reuses completed per-image results.
    run_lock = (output / 'score.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    config = args.geneval_config
    if args.benchmark == 'geneval' and config is None:
        import mmdet
        config = Path(mmdet.__file__).parent / '.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py'
    spec = {
        'benchmark': args.benchmark, 'image_dir': str(args.image_dir),
        'generation': manifest,
        'method': 'geneval-bundled-mm3-strict-v2' if args.benchmark == 'geneval' else 'dpg-independent-images-all-csv-rows-v1',
        'model': str(args.geneval_model_path.resolve()) if args.benchmark == 'geneval' else args.dpg_model,
        'config': str(config.resolve()) if config else None,
        'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest() if args.benchmark == 'dpgbench' else None,
    }
    spec_path = output / 'score_run.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise RuntimeError('Scoring configuration changed; choose a new --output-dir')
    spec_path.write_text(json.dumps(spec, indent=2) + '\n')
    details_path = output / 'details.jsonl'
    module_path = ROOT / ('Benchmark/geneval/evaluation/summary_scores.py'
                          if args.benchmark == 'geneval' else 'Benchmark/dpg_bench/compute_dpg_bench.py')
    module_spec = importlib.util.spec_from_file_location('pixel_evaluator', module_path)
    evaluator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(evaluator)
    if args.benchmark == 'geneval':
        if not config.is_file():
            raise FileNotFoundError(f'Missing Mask2Former config: {config}; use --geneval-config')
        command = [sys.executable, str(ROOT / 'Benchmark/geneval/evaluation/evaluate_images.py'),
                   '--imagedir', str(args.image_dir), '--outfile', str(details_path),
                   '--model-path', str(args.geneval_model_path), '--model-config', str(config)]
        subprocess.run(command, check=True, cwd=ROOT)
        results = [json.loads(line) for line in details_path.read_text().splitlines() if line.strip()]
        if {str(Path(x['filename']).resolve()) for x in results} != {str(x) for x in images} or len(results) != len(images):
            raise RuntimeError('GenEval evaluator output does not match validated input images')
    else:
        results = evaluator.score_images(images, args.csv, details_path, args.dpg_model)
    score, metrics = evaluator.summarize_results(results)
    summary = dict(experiment=args.experiment, benchmark=args.benchmark,
                   scored_images=len(results), expected_images=expected,
                   complete_prompts=complete_prompts, total_prompts=len(rows),
                   complete=len(images) == expected and len(rows) == full_prompt_count,
                   sampler=manifest.get('sampler', 'sample'),
                   num_iter=manifest['num_iter'], cfg=manifest['cfg'],
                   batch_size=manifest.get('batch_size', 1), score_percent=score,
                   category_scores_percent=metrics)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    update_record(args.record, args.experiment, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
