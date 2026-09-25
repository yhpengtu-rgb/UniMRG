#!/usr/bin/env python3
"""Create a canonical single-test summary from one VLMEvalKit run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLMEVAL_ROOT = REPO_ROOT / 'Benchmark' / 'VLMEvalKit'
if str(VLMEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(VLMEVAL_ROOT))

from vlmeval.utils import can_infer  # noqa: E402
from vlmeval.utils import matching_util  # noqa: E402


DATASET_ORDER = (
    'MMBench_DEV_EN',
    'RealWorldQA',
    'MMVP',
    'HallusionBench',
    'VSR-zeroshot',
)
EXPECTED_SAMPLE_COUNTS = {
    'MMBench_DEV_EN': 4329,
    'RealWorldQA': 765,
    'MMVP': 300,
    'HallusionBench': 951,
    'VSR-zeroshot': 1222,
}


class SummaryError(RuntimeError):
    """Raised when a run cannot be summarized without ambiguity."""


def _load_expected_counts_from_environment() -> None:
    raw = os.environ.get('UNIMRG_EXPECTED_SAMPLE_COUNTS')
    if not raw:
        return
    try:
        parsed = json.loads(raw)
        counts = {name: int(parsed[name]) for name in DATASET_ORDER}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SummaryError(
            'UNIMRG_EXPECTED_SAMPLE_COUNTS must be a JSON object containing '
            f'{list(DATASET_ORDER)}'
        ) from exc
    EXPECTED_SAMPLE_COUNTS.clear()
    EXPECTED_SAMPLE_COUNTS.update(counts)


_load_expected_counts_from_environment()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        chunk = handle.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(1024 * 1024)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_one(run_dir: Path, model_name: str, suffix: str, dataset: str) -> Path:
    expected_name = f'{model_name}_{suffix}'
    discovered = sorted(
        path for path in run_dir.rglob(expected_name)
        if path.is_file()
    )
    matches = sorted({path.resolve() for path in discovered})
    if not matches:
        raise SummaryError(
            f'{dataset}: missing required result file {expected_name} under {run_dir}'
        )
    if len(matches) != 1:
        rendered = ', '.join(str(path) for path in discovered)
        raise SummaryError(
            f'{dataset}: multiple result files named {expected_name}: {rendered}'
        )
    return matches[0]


def _load_raw_dataset(
    run_dir: Path,
    model_name: str,
    dataset: str,
) -> tuple[Path, pd.DataFrame]:
    path = _resolve_one(
        run_dir,
        model_name,
        f'{dataset}.xlsx',
        dataset,
    )
    try:
        data = pd.read_excel(path)
    except Exception as exc:
        raise SummaryError(f'{dataset}: failed to read {path}: {exc}') from exc
    expected = EXPECTED_SAMPLE_COUNTS[dataset]
    found = len(data)
    if found != expected:
        raise SummaryError(
            f'{dataset}: expected {expected} raw samples, found {found} in {path}'
        )
    return path, data


def _load_csv(path: Path, dataset: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise SummaryError(f'{dataset}: failed to read {path}: {exc}') from exc


def _score(value: Any, dataset: str, path: Path, field: str, scale: float = 1.0) -> float:
    try:
        score = float(value) * scale
    except (TypeError, ValueError) as exc:
        raise SummaryError(
            f'{dataset}: field {field} in {path} is not numeric: {value!r}'
        ) from exc
    if not math.isfinite(score):
        raise SummaryError(
            f'{dataset}: field {field} in {path} must be finite, got {value!r}'
        )
    if not 0.0 <= score <= 100.0:
        raise SummaryError(
            f'{dataset}: field {field} in {path} is outside [0, 100]: {score}'
        )
    return score


def _file_record(path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        'path': _relative_path(path, run_dir),
        'sha256': sha256_file(path),
        'size_bytes': path.stat().st_size,
    }


def _score_mmbench(data: pd.DataFrame, path: Path) -> float:
    required = {'answer', 'prediction'}
    missing = required.difference(data.columns)
    if missing:
        raise SummaryError(
            f'MMBench_DEV_EN: missing columns {sorted(missing)} in {path}'
        )
    hits = 0
    for _, row in data.iterrows():
        choices = {
            key: row[key]
            for key in 'ABCD'
            if key in data.columns and pd.notna(row[key])
        }
        predicted = can_infer(row['prediction'], choices)
        hits += predicted == row['answer']
    return 100.0 * hits / len(data)


def summarize_run(run_dir: Path, model_name: str) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise SummaryError(f'run directory does not exist: {run_dir}')
    if not model_name or '/' in model_name:
        raise SummaryError(f'invalid model name: {model_name!r}')

    raw_files: dict[str, Path] = {}
    raw_data: dict[str, pd.DataFrame] = {}
    for dataset in DATASET_ORDER:
        path, data = _load_raw_dataset(run_dir, model_name, dataset)
        raw_files[dataset] = path
        raw_data[dataset] = data

    datasets: dict[str, dict[str, Any]] = {}

    mmbench_score = _score_mmbench(
        raw_data['MMBench_DEV_EN'],
        raw_files['MMBench_DEV_EN'],
    )
    datasets['MMBench_DEV_EN'] = {
        'score': mmbench_score,
        'display_score': round(mmbench_score, 2),
        'sample_count': len(raw_data['MMBench_DEV_EN']),
        'source_field': 'official can_infer over vanilla_all',
        'inputs': [_file_record(raw_files['MMBench_DEV_EN'], run_dir)],
    }

    real_path = _resolve_one(
        run_dir,
        model_name,
        'RealWorldQA_acc.csv',
        'RealWorldQA',
    )
    real = _load_csv(real_path, 'RealWorldQA')
    if len(real) != 1 or 'Overall' not in real.columns:
        raise SummaryError(
            f'RealWorldQA: expected one row with Overall in {real_path}'
        )
    real_score = _score(
        real.iloc[0]['Overall'],
        'RealWorldQA',
        real_path,
        'Overall',
        100.0,
    )
    datasets['RealWorldQA'] = {
        'score': real_score,
        'display_score': round(real_score, 2),
        'sample_count': len(raw_data['RealWorldQA']),
        'source_field': 'Overall',
        'inputs': [
            _file_record(raw_files['RealWorldQA'], run_dir),
            _file_record(real_path, run_dir),
        ],
    }

    mmvp_path = _resolve_one(
        run_dir,
        model_name,
        'MMVP_acc.csv',
        'MMVP',
    )
    mmvp = _load_csv(mmvp_path, 'MMVP')
    if len(mmvp) != 1 or 'Average' not in mmvp.columns:
        raise SummaryError(f'MMVP: expected one row with Average in {mmvp_path}')
    mmvp_score = _score(
        mmvp.iloc[0]['Average'],
        'MMVP',
        mmvp_path,
        'Average',
        100.0,
    )
    datasets['MMVP'] = {
        'score': mmvp_score,
        'display_score': round(mmvp_score, 2),
        'sample_count': len(raw_data['MMVP']),
        'source_field': 'Average',
        'inputs': [
            _file_record(raw_files['MMVP'], run_dir),
            _file_record(mmvp_path, run_dir),
        ],
    }

    hall_path = _resolve_one(
        run_dir,
        model_name,
        'HallusionBench_score.csv',
        'HallusionBench',
    )
    hall = _load_csv(hall_path, 'HallusionBench')
    if 'split' not in hall.columns or 'aAcc' not in hall.columns:
        raise SummaryError(
            f'HallusionBench: expected split and aAcc columns in {hall_path}'
        )
    overall = hall[hall['split'] == 'Overall']
    if len(overall) != 1:
        raise SummaryError(
            f'HallusionBench: expected exactly one split=Overall row in {hall_path}'
        )
    hall_score = _score(
        overall.iloc[0]['aAcc'],
        'HallusionBench',
        hall_path,
        'split=Overall|aAcc',
    )
    datasets['HallusionBench'] = {
        'score': hall_score,
        'display_score': round(hall_score, 2),
        'sample_count': len(raw_data['HallusionBench']),
        'source_field': 'split=Overall|aAcc',
        'inputs': [
            _file_record(raw_files['HallusionBench'], run_dir),
            _file_record(hall_path, run_dir),
        ],
    }

    vsr_path = _resolve_one(
        run_dir,
        model_name,
        'VSR-zeroshot_score.csv',
        'VSR-zeroshot',
    )
    vsr = _load_csv(vsr_path, 'VSR-zeroshot')
    if len(vsr) != 1 or 'acc' not in vsr.columns:
        raise SummaryError(
            f'VSR-zeroshot: expected one row with acc in {vsr_path}'
        )
    vsr_score = _score(
        vsr.iloc[0]['acc'],
        'VSR-zeroshot',
        vsr_path,
        'acc',
    )
    datasets['VSR-zeroshot'] = {
        'score': vsr_score,
        'display_score': round(vsr_score, 2),
        'sample_count': len(raw_data['VSR-zeroshot']),
        'source_field': 'acc',
        'inputs': [
            _file_record(raw_files['VSR-zeroshot'], run_dir),
            _file_record(vsr_path, run_dir),
        ],
    }

    average = sum(datasets[name]['score'] for name in DATASET_ORDER) / 5.0
    parser_path = Path(matching_util.__file__).resolve()
    return {
        'schema_version': 1,
        'judge': 'exact_matching',
        'aggregation': 'single_test',
        'model_name': model_name,
        'run_dir': str(run_dir),
        'dataset_order': list(DATASET_ORDER),
        'datasets': datasets,
        'average': average,
        'display_average': round(average, 2),
        'parser': {
            'callable': 'vlmeval.utils.can_infer',
            'path': str(parser_path),
            'sha256': sha256_file(parser_path),
        },
    }


def _unique_inputs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for dataset in DATASET_ORDER:
        for record in summary['datasets'][dataset]['inputs']:
            by_path[record['path']] = record
    return [by_path[path] for path in sorted(by_path)]


def write_outputs(summary: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'single_test_results.json'
    csv_path = output_dir / 'single_test_results.csv'
    markdown_path = output_dir / 'single_test_results.md'
    hashes_path = output_dir / 'result_inputs.sha256'

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    fieldnames = ['model_name', *DATASET_ORDER, 'Avg']
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'model_name': summary['model_name'],
            **{
                dataset: repr(summary['datasets'][dataset]['score'])
                for dataset in DATASET_ORDER
            },
            'Avg': repr(summary['average']),
        })

    display = [
        summary['datasets'][dataset]['display_score']
        for dataset in DATASET_ORDER
    ]
    markdown_path.write_text(
        '| Model | MMBench_DEV_EN | RealWorldQA | MMVP | '
        'HallusionBench | VSR-zeroshot | Avg |\n'
        '|---|---:|---:|---:|---:|---:|---:|\n'
        f"| {summary['model_name']} | "
        + ' | '.join(f'{value:.2f}' for value in display)
        + f" | {summary['display_average']:.2f} |\n",
        encoding='utf-8',
    )

    hashes_path.write_text(
        ''.join(
            f"{record['sha256']}  {record['path']}\n"
            for record in _unique_inputs(summary)
        ),
        encoding='utf-8',
    )
    return [json_path, csv_path, markdown_path, hashes_path]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Summarize one UniMRG VLMEvalKit run with canonical single-test metrics.'
    )
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = summarize_run(args.run_dir, args.model_name)
        paths = write_outputs(summary, args.output_dir or args.run_dir)
    except SummaryError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
