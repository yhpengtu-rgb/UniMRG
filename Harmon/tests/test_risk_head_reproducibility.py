"""Regression tests for reproducible TRACER RiskHead fitting."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


HARMON_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = HARMON_ROOT / 'scripts' / 'train_risk_head.py'


def _load_training_module():
    spec = importlib.util.spec_from_file_location(
        'tracer_train_risk_head', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tiny_trajectory(path: Path) -> None:
    generator = torch.Generator().manual_seed(907)
    records = []
    for record_idx in range(4):
        block = 8
        confidence = torch.rand(1, block, generator=generator)
        records.append({
            'hidden': torch.randn(1, block, 8, generator=generator),
            'confidence': confidence,
            'entropy': 1.0 - confidence,
            'js_div': torch.rand(1, block, generator=generator),
            'stable': torch.tensor(
                [[(record_idx + i) % 2 for i in range(block)]],
                dtype=torch.float32,
            ),
            'committed_history': torch.zeros(1, block),
            'remask': torch.zeros(1, block),
            'block_commit_corr': torch.linspace(0.0, 1.0, block).view(1, -1),
            'state': torch.tensor([[0.25, 0.50, 0.75]]),
            'wrong_commit': torch.tensor(
                [[(record_idx + i) % 2 for i in range(block)]],
                dtype=torch.float32,
            ),
        })
    torch.save({'samples': [{'records': records}]}, path)


def _fit_args(trajectory: Path, output_dir: Path, seed: int):
    return argparse.Namespace(
        trajectory=str(trajectory),
        output_dir=str(output_dir),
        epochs=3,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=8,
        proj_hidden=4,
        patience=3,
        seed=seed,
    )


class RiskHeadReproducibilityTest(unittest.TestCase):
    def test_two_fits_with_the_same_seed_produce_identical_state(self):
        """Catches unseeded parameter initialisation or DataLoader shuffle."""
        module = _load_training_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = root / 'trajectory.pt'
            _write_tiny_trajectory(trajectory)

            with patch.object(module.torch.cuda, 'is_available', return_value=False):
                module.fit(_fit_args(trajectory, root / 'run_a', seed=42))
                module.fit(_fit_args(trajectory, root / 'run_b', seed=42))

            first = torch.load(
                root / 'run_a' / 'risk_head.pt', map_location='cpu')
            second = torch.load(
                root / 'run_b' / 'risk_head.pt', map_location='cpu')
            self.assertEqual(first['state_dict'].keys(), second['state_dict'].keys())
            for key in first['state_dict']:
                torch.testing.assert_close(
                    first['state_dict'][key], second['state_dict'][key],
                    rtol=0.0, atol=0.0,
                )
            self.assertEqual(first['temperature'], second['temperature'])
            torch.testing.assert_close(first['iso_xs'], second['iso_xs'])
            torch.testing.assert_close(first['iso_ys'], second['iso_ys'])


if __name__ == '__main__':
    unittest.main()
