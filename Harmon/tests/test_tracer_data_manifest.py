"""Contracts for relocatable TRACER training-data manifests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HARMON_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = HARMON_ROOT / 'scripts' / 'tracer_data_manifest.py'


class TracerDataManifestTest(unittest.TestCase):
    def run_script(self, *arguments):
        return subprocess.run(
            ['python', str(SCRIPT), *map(str, arguments)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_manifest_is_relocatable_and_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / 'source'
            relocated = temp / 'relocated'
            manifest = temp / 'dataset.jsonl'
            (source / 'images').mkdir(parents=True)
            (source / 'samples.json').write_text('[]\n', encoding='utf-8')
            (source / 'images' / 'one.bin').write_bytes(b'one')

            built = self.run_script(
                'build', '--data-root', source, '--manifest', manifest)
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(json.loads(built.stdout)['file_count'], 2)

            relocated.mkdir()
            (relocated / 'images').mkdir()
            (relocated / 'samples.json').write_text('[]\n', encoding='utf-8')
            (relocated / 'images' / 'one.bin').write_bytes(b'one')
            verified = self.run_script(
                'verify', '--data-root', relocated, '--manifest', manifest)
            self.assertEqual(verified.returncode, 0, verified.stderr)

            (relocated / 'images' / 'one.bin').write_bytes(b'two')
            drifted = self.run_script(
                'verify', '--data-root', relocated, '--manifest', manifest)
            self.assertNotEqual(drifted.returncode, 0)
            self.assertIn('mismatch', drifted.stderr)

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            root = temp / 'data'
            root.mkdir()
            manifest = temp / 'unsafe.jsonl'
            records = [
                {
                    'record': 'header',
                    'schema_version': 1,
                    'kind': 'tracer_training_data',
                    'path_semantics': 'relative_to_data_root',
                    'hash': 'sha256',
                },
                {
                    'record': 'file',
                    'path': '../outside',
                    'size_bytes': 0,
                    'sha256': '0' * 64,
                },
                {'record': 'summary', 'file_count': 1, 'total_bytes': 0},
            ]
            manifest.write_text(
                ''.join(json.dumps(item) + '\n' for item in records),
                encoding='utf-8',
            )
            completed = self.run_script(
                'verify', '--data-root', root, '--manifest', manifest)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('unsafe manifest path', completed.stderr)


if __name__ == '__main__':
    unittest.main()
