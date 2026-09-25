#!/usr/bin/env python3
"""Build or verify a relocatable content manifest for TRACER training data."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root, excluded):
    for path in sorted(root.rglob('*')):
        if path in excluded:
            continue
        if path.is_symlink():
            raise RuntimeError(
                'dataset manifests reject symlinks; materialize the file: '
                f'{path}')
        if path.is_file():
            yield path


def build_manifest(data_root, output):
    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f'data root is not a directory: {root}')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    excluded = {output, temporary}
    count = 0
    total_bytes = 0
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps({
            'record': 'header',
            'schema_version': SCHEMA_VERSION,
            'kind': 'tracer_training_data',
            'path_semantics': 'relative_to_data_root',
            'hash': 'sha256',
        }, sort_keys=True) + '\n')
        for path in _iter_files(root, excluded):
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            handle.write(json.dumps({
                'record': 'file',
                'path': relative,
                'size_bytes': size,
                'sha256': _sha256(path),
            }, sort_keys=True) + '\n')
            count += 1
            total_bytes += size
        handle.write(json.dumps({
            'record': 'summary',
            'file_count': count,
            'total_bytes': total_bytes,
        }, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {'file_count': count, 'total_bytes': total_bytes}


def _safe_relative_path(raw):
    path = PurePosixPath(raw)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise RuntimeError(f'unsafe manifest path: {raw!r}')
    return Path(*path.parts)


def verify_manifest(data_root, manifest):
    root = data_root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    seen = set()
    count = 0
    total_bytes = 0
    summary = None
    with manifest.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f'invalid JSON at {manifest}:{line_number}: {error}')
            kind = record.get('record')
            if line_number == 1:
                expected = {
                    'record': 'header',
                    'schema_version': SCHEMA_VERSION,
                    'kind': 'tracer_training_data',
                    'path_semantics': 'relative_to_data_root',
                    'hash': 'sha256',
                }
                if record != expected:
                    raise RuntimeError('unsupported dataset manifest header')
                continue
            if kind == 'summary':
                summary = record
                if any(remaining.strip() for remaining in handle):
                    raise RuntimeError('summary must be the final manifest record')
                break
            if kind != 'file':
                raise RuntimeError(
                    f'expected file record at line {line_number}')
            relative = _safe_relative_path(record.get('path'))
            if relative in seen:
                raise RuntimeError(f'duplicate manifest path: {relative}')
            seen.add(relative)
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f'manifest file missing or unsafe: {path}')
            size = path.stat().st_size
            if size != record.get('size_bytes'):
                raise RuntimeError(f'manifest size mismatch: {path}')
            if _sha256(path) != record.get('sha256'):
                raise RuntimeError(f'manifest SHA256 mismatch: {path}')
            count += 1
            total_bytes += size
    if summary is None:
        raise RuntimeError('dataset manifest has no summary record')
    if summary != {
        'record': 'summary',
        'file_count': count,
        'total_bytes': total_bytes,
    }:
        raise RuntimeError('dataset manifest summary does not match records')
    return {'file_count': count, 'total_bytes': total_bytes}


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'verify'):
        child = subparsers.add_parser(command)
        child.add_argument('--data-root', type=Path, required=True)
        child.add_argument('--manifest', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == 'build':
        result = build_manifest(args.data_root, args.manifest)
    else:
        result = verify_manifest(args.data_root, args.manifest)
    print(json.dumps({'state': 'PASS', **result}, sort_keys=True))


if __name__ == '__main__':
    main()
