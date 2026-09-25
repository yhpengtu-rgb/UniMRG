#!/usr/bin/env python3
"""Aggregate per-sample TRACER runtime diagnostics into one JSON report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COUNTERS = (
    'blocks_started',
    'blocks_completed',
    'denoising_rounds',
    'risk_head_calls',
    'risk_policy_rounds',
    'confidence_fallback_rounds',
    'revision_count',
    'new_commit_count',
    'retained_count',
)


def _distribution(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {'count': 0, 'mean': None, 'p50': None, 'p95': None,
                'min': None, 'max': None}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'p50': percentile(0.50),
        'p95': percentile(0.95),
        'min': values[0],
        'max': values[-1],
    }


def _merge_moments(records, key):
    entries = [record.get(key, {}) for record in records]
    total_count = sum(int(entry.get('count', 0)) for entry in entries)
    if total_count == 0:
        return {
            'count': 0, 'mean': None, 'std': None,
            'min': None, 'max': None,
            'nonneutral_count': 0, 'nonneutral_fraction': None,
            'neutral_value': None,
        }
    count = 0
    mean = 0.0
    m2 = 0.0
    for entry in entries:
        entry_count = int(entry.get('count', 0))
        if entry_count == 0:
            continue
        entry_mean = float(entry['mean'])
        combined_count = count + entry_count
        delta = entry_mean - mean
        m2 += (
            float(entry.get('m2', 0.0))
            + delta * delta * count * entry_count / combined_count
        )
        mean += delta * entry_count / combined_count
        count = combined_count
    minimum = min(
        float(entry['min']) for entry in entries
        if int(entry.get('count', 0)) > 0)
    maximum = max(
        float(entry['max']) for entry in entries
        if int(entry.get('count', 0)) > 0)
    nonneutral = sum(
        int(entry.get('nonneutral_count', 0)) for entry in entries)
    neutral_values = {
        entry.get('neutral_value') for entry in entries
        if int(entry.get('count', 0)) > 0
    }
    neutral_value = (
        next(iter(neutral_values))
        if len(neutral_values) == 1 and None not in neutral_values
        else None
    )
    variance = max(m2 / count, 0.0)
    return {
        'count': count,
        'mean': mean,
        'std': math.sqrt(variance),
        'min': minimum,
        'max': maximum,
        'nonneutral_count': nonneutral,
        'nonneutral_fraction': (
            nonneutral / count if neutral_value is not None else None),
        'neutral_value': neutral_value,
    }


def _summarize(records):
    eos = [bool(value) for row in records for value in row['eos_emitted']]
    exhausted = [
        bool(value) for row in records
        for value in row['max_token_exhausted']
    ]
    output_tokens = [
        int(value) for row in records for value in row['output_tokens']
    ]
    payload = {
        'sample_count': len(records),
        'sequence_count': len(eos),
        'eos_count': sum(eos),
        'eos_rate': sum(eos) / len(eos),
        'max_token_exhaustion_count': sum(exhausted),
        'max_token_exhaustion_rate': sum(exhausted) / len(exhausted),
        'output_tokens': _distribution(output_tokens),
        'response_chars': _distribution(
            int(row.get('response_chars', 0)) for row in records),
        'gate_scale': _merge_moments(records, 'gate_scale'),
        'active_gate_scale': _merge_moments(records, 'active_gate_scale'),
        'risk': _merge_moments(records, 'risk'),
    }
    for key in COUNTERS:
        payload[key] = sum(int(row.get(key, 0)) for row in records)
    fallback_reasons = {}
    for row in records:
        for reason, count in row.get('fallback_reasons', {}).items():
            fallback_reasons[reason] = (
                fallback_reasons.get(reason, 0) + int(count))
    payload['fallback_reasons'] = dict(sorted(fallback_reasons.items()))
    return payload


def summarize(
    input_path: Path,
    output_path: Path,
    *,
    expected_results_path: Path | None = None,
) -> dict:
    records = []
    for number, line in enumerate(
            input_path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('schema_version') != 1:
            raise ValueError(
                f'unsupported schema on line {number}: '
                f"{record.get('schema_version')!r}")
        records.append(record)
    if not records:
        raise ValueError(f'no runtime records found in {input_path}')

    datasets = {}
    for dataset in sorted({row.get('dataset', 'unknown') for row in records}):
        datasets[dataset] = _summarize([
            row for row in records
            if row.get('dataset', 'unknown') == dataset
        ])
    expected_sample_counts = None
    if expected_results_path is not None:
        expected_results = json.loads(
            expected_results_path.read_text(encoding='utf-8'))
        expected_sample_counts = {
            dataset: int(result['sample_count'])
            for dataset, result in expected_results['datasets'].items()
        }
        actual_sample_counts = {
            dataset: int(summary['sequence_count'])
            for dataset, summary in datasets.items()
        }
        if actual_sample_counts != expected_sample_counts:
            raise ValueError(
                'runtime diagnostics sample-count mismatch: '
                f'actual={actual_sample_counts}, '
                f'expected={expected_sample_counts}')
    payload = {
        'schema_version': 1,
        'source': str(input_path),
        'overall': _summarize(records),
        'datasets': datasets,
        'expected_sample_counts': expected_sample_counts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-results', type=Path)
    args = parser.parse_args()
    summarize(
        args.input,
        args.output,
        expected_results_path=args.expected_results,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
