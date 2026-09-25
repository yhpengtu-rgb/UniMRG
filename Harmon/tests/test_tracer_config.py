"""Configuration and launcher contracts for TRACER-LoRA experiments."""

import json
import os
import copy
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmengine.config import Config


HARMON_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG = (
    HARMON_ROOT / 'configs' / 'examples' /
    'UniMRG_dllm_lora_tracer_full.py'
)
INFER_CONFIG = (
    HARMON_ROOT / 'configs' / 'examples' /
    'UniMRG_dllm_lora_tracer_full_infer.py'
)
ORCHESTRATOR = (
    HARMON_ROOT / 'scripts' /
    'run_tracer_lora_experiments.sh'
)
RISK_PREPARE = HARMON_ROOT / 'scripts' / 'prepare_tracer_risk.sh'
EXP5_TRAIN_ENTRY = HARMON_ROOT.parent / 'run_exp5_train.sh'
EXP5_EVAL_ENTRY = HARMON_ROOT.parent / 'run_exp5_eval.sh'
EVAL_INSTANCE_CONFIG = (
    HARMON_ROOT / 'configs' / 'eval' /
    'tracer_lora_exp5_ddpfix_coupled.env'
)
INVALID_EXP4_CONFIG = (
    HARMON_ROOT / 'configs' / 'eval' /
    'tracer_lora_exp4_coupled.env'
)


class TracerConfigTest(unittest.TestCase):
    def test_eval_manifest_records_resolved_checkpoint_and_diagnostics(self):
        """Catches a plan whose manifest hides the actual eval artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            orchestration_root = temp / 'orchestration'
            checkpoint = temp / 'exp7-s41.pth'
            env = os.environ.copy()
            env.update({
                'EXPERIMENT_ID': 'manifest-provenance-v1',
                'STAGE': 'eval',
                'PLAN_ONLY': '1',
                'DRY_RUN': '0',
                'OFFICIAL_EVAL_FROZEN': '1',
                'SEEDS': '41',
                'TRACER_CELLS': 'plain coupled',
                'TRACER_CHECKPOINT': str(checkpoint),
                'EVAL_RUN_ID_PREFIX': 'manifest-eval',
                'ORCHESTRATION_ROOT': str(orchestration_root),
            })

            completed = subprocess.run(
                ['bash', str(ORCHESTRATOR)],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            run_dir = (
                orchestration_root / 'runs' /
                'manifest-provenance-v1-eval-plan'
            )
            manifest = json.loads(
                (run_dir / 'manifest.json').read_text(encoding='utf-8'))
            commands = (run_dir / 'commands.sh').read_bytes()

        self.assertIn('checkpoint_source', manifest)
        self.assertEqual(manifest['checkpoint_source'], 'override')
        self.assertEqual(
            manifest['resolved_checkpoints'], {'41': str(checkpoint)})
        self.assertEqual(
            manifest['eval_run_ids'],
            {
                '41': {
                    'plain': 'manifest-eval-s41-plain',
                    'coupled': 'manifest-eval-s41-coupled',
                },
            },
        )
        self.assertEqual(
            manifest['runtime_diagnostics'],
            {
                'enabled': True,
                'schema_version': 1,
                'samples_file': 'tracer_runtime_samples.jsonl',
                'summary_file': 'tracer_runtime_stats.json',
                'completeness_check': 'single_test_results.sample_count',
            },
        )
        self.assertEqual(manifest['commands_file'], 'commands.sh')
        self.assertEqual(
            manifest['commands_sha256'],
            hashlib.sha256(commands).hexdigest(),
        )

    def test_risk_dry_run_records_deterministic_fit_contract(self):
        """Catches a risk manifest that cannot reproduce its fitted head."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / 'risk'
            env = os.environ.copy()
            env.update({
                'RUN_ID': 'deterministic-risk-contract',
                'OUTPUT_ROOT': str(output_root),
                'DRY_RUN': '1',
                'SEED': '73',
            })
            completed = subprocess.run(
                ['bash', str(RISK_PREPARE)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            run_dir = output_root / 'runs' / 'deterministic-risk-contract'
            manifest = json.loads(
                (run_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(
                manifest['risk_fit_reproducibility'],
                {
                    'seed': 73,
                    'pythonhashseed': '73',
                    'cublas_workspace_config': ':4096:8',
                    'torch_deterministic_algorithms': True,
                    'dataloader_workers': 0,
                    'dataloader_generator_seed': 73,
                },
            )
            inputs = (run_dir / 'inputs.sha256').read_text(encoding='utf-8')
            self.assertIn(str(TRAIN_CONFIG.with_name(
                'UniMRG_dllm_lora_tracer_full_infer.py')), inputs)

    def test_exp5_eval_entry_defaults_to_plan_without_launching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan_root = temp / 'plans'
            capture = temp / 'launched'
            launcher = temp / 'fake_eval.sh'
            launcher.write_text(
                '#!/usr/bin/env bash\n'
                'touch "$CAPTURE"\n',
                encoding='utf-8',
            )
            launcher.chmod(0o755)
            checkpoints = [temp / 'iter_41.pth']
            risk_head = temp / 'risk_head.pt'
            audit = temp / 'risk_audit.json'
            for path in (*checkpoints, risk_head, audit):
                path.write_bytes(b'fixture')

            env = os.environ.copy()
            env.update({
                'PLAN_ROOT': str(plan_root),
                'EVAL_LAUNCHER': str(launcher),
                'CAPTURE': str(capture),
                'CHECKPOINT_TEMPLATE': str(temp / 'iter_{seed}.pth'),
                'WORK_ROOT': str(temp / 'work'),
                'TRACER_RISK_HEAD_PATH': str(risk_head),
                'TRACER_RISK_AUDIT_PATH': str(audit),
                'RUN_DATE': '20990101',
            })
            env.pop('EXECUTE', None)

            completed = subprocess.run(
                ['bash', str(EXP5_EVAL_ENTRY)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertFalse(capture.exists())
            manifest = json.loads(
                (plan_root / 'manifest.json').read_text(encoding='utf-8')
            )
            self.assertEqual(manifest['state'], 'planned')
            self.assertEqual(manifest['seeds'], [41])
            self.assertEqual(manifest['tracer_cell'], 'coupled')
            self.assertEqual(manifest['max_new_tokens'], 512)
            self.assertEqual(manifest['preflight_samples'], 8)
            self.assertEqual(
                manifest['revision_budget'],
                {
                    'fraction': 0.10,
                    'round_cap_rule': 'ceil(fraction * block_size)',
                    'max_revisions_per_round': 1,
                    'effective_block_fraction_cap': 0.25,
                },
            )
            self.assertEqual(len(manifest['commands']), 1)
            self.assertEqual(
                manifest['checkpoints'], [str(path) for path in checkpoints])
            self.assertTrue(
                (plan_root / 'commands.sh').read_text(encoding='utf-8')
                .startswith('#!/usr/bin/env bash\nset -euo pipefail\n')
            )
            self.assertIn(
                f'RUN_DATE=20990101 PLAN_ROOT={plan_root} EXECUTE=1 '
                f'bash {EXP5_EVAL_ENTRY}',
                completed.stdout,
            )

    def test_exp5_eval_execute_mode_forwards_reproducible_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            plan_root = temp / 'plans'
            capture = temp / 'capture.json'
            launcher = temp / 'fake_eval.sh'
            launcher.write_text(
                '#!/usr/bin/env bash\n'
                'python - <<\'PY\'\n'
                'import json, os\n'
                'keys = ["CHECKPOINT", "RUN_ID", "TRACER_CELL", '
                '"TRACER_ROUTER_ENABLED", "TRACER_POLICY_ENABLED", '
                '"BLOCK_SIZE", "DENOISING_STEPS", "MAX_NEW_TOKENS", '
                '"DRY_RUN", "OFFICIAL_EVAL_FROZEN"]\n'
                'json.dump({key: os.environ[key] for key in keys}, '
                'open(os.environ["CAPTURE"], "w"))\n'
                'PY\n',
                encoding='utf-8',
            )
            launcher.chmod(0o755)
            checkpoint = temp / 'iter_20000.pth'
            risk_head = temp / 'risk_head.pt'
            audit = temp / 'risk_audit.json'
            for path in (checkpoint, risk_head, audit):
                path.write_bytes(b'fixture')

            env = os.environ.copy()
            env.update({
                'PLAN_ROOT': str(plan_root),
                'EVAL_LAUNCHER': str(launcher),
                'CAPTURE': str(capture),
                'CHECKPOINT_ROOT': str(temp),
                'WORK_ROOT': str(temp / 'work'),
                'TRACER_RISK_HEAD_PATH': str(risk_head),
                'TRACER_RISK_AUDIT_PATH': str(audit),
                'SEEDS': '42',
                'RUN_DATE': '20990101',
            })

            planned = subprocess.run(
                ['bash', str(EXP5_EVAL_ENTRY)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(
                planned.returncode, 0,
                msg=planned.stdout + planned.stderr,
            )
            self.assertFalse(capture.exists())

            env['EXECUTE'] = '1'
            completed = subprocess.run(
                ['bash', str(EXP5_EVAL_ENTRY)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            captured = json.loads(capture.read_text(encoding='utf-8'))
            self.assertEqual(captured['CHECKPOINT'], str(checkpoint))
            self.assertEqual(
                captured['RUN_ID'],
                'tracer-v2-exp5-hotstartfix-coupled-s42-single-20990101',
            )
            self.assertEqual(captured['TRACER_CELL'], 'coupled')
            self.assertEqual(captured['TRACER_ROUTER_ENABLED'], 'true')
            self.assertEqual(captured['TRACER_POLICY_ENABLED'], 'true')
            self.assertEqual(
                [captured['BLOCK_SIZE'], captured['DENOISING_STEPS'],
                 captured['MAX_NEW_TOKENS']],
                ['4', '4', '512'],
            )
            self.assertEqual(captured['DRY_RUN'], '0')
            self.assertEqual(captured['OFFICIAL_EVAL_FROZEN'], '1')
            manifest = json.loads(
                (plan_root / 'manifest.json').read_text(encoding='utf-8')
            )
            self.assertEqual(manifest['state'], 'completed')

    def test_exp5_train_entry_defaults_to_plan_and_single_seed41(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture = temp / 'capture.json'
            fake_orchestrator = temp / 'orchestrator.sh'
            fake_orchestrator.write_text(
                '#!/usr/bin/env bash\n'
                'python - "$CAPTURE" <<\'PY\'\n'
                'import json, os, sys\n'
                'keys = ["STAGE", "PLAN_ONLY", "DRY_RUN", "SEEDS", '
                '"EXPERIMENT_ID", "TRAIN_RUN_ID_PREFIX"]\n'
                'open(sys.argv[1], "w").write(json.dumps('
                '{key: os.environ[key] for key in keys}))\n'
                'PY\n',
                encoding='utf-8',
            )
            fake_orchestrator.chmod(0o755)
            required = []
            for name in ('base.pth', 'risk.pt', 'audit.json', 'exclude.json'):
                path = temp / name
                path.write_bytes(b'fixture')
                required.append(path)
            required[3].with_name('exclude.json.sha256').write_text(
                'fixture\n', encoding='utf-8')
            env = os.environ.copy()
            env.update({
                'ORCHESTRATOR': str(fake_orchestrator),
                'CAPTURE': str(capture),
                'TRACER_BASE_CHECKPOINT': str(required[0]),
                'TRACER_RISK_HEAD_PATH': str(required[1]),
                'TRACER_RISK_AUDIT_PATH': str(required[2]),
                'HARMON_GUARD_EXCLUSION_MANIFEST': str(required[3]),
                'MIN_ROOT_FREE_GB': '0',
            })
            for key in (
                'EXECUTE', 'STAGE', 'PLAN_ONLY', 'DRY_RUN', 'SEEDS',
                'EXPERIMENT_ID', 'TRAIN_RUN_ID_PREFIX',
            ):
                env.pop(key, None)

            completed = subprocess.run(
                ['bash', str(EXP5_TRAIN_ENTRY)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            captured = json.loads(capture.read_text(encoding='utf-8'))
            self.assertEqual(captured['STAGE'], 'train')
            self.assertEqual(captured['PLAN_ONLY'], '1')
            self.assertEqual(captured['DRY_RUN'], '0')
            self.assertEqual(captured['SEEDS'], '41')
            self.assertEqual(
                captured['EXPERIMENT_ID'], 'tracer-v2-exp5-hotstartfix')
            self.assertEqual(
                captured['TRAIN_RUN_ID_PREFIX'],
                'tracer-v2-exp5-hotstartfix-train',
            )

    def test_exp5_train_entry_execute_mode_launches_real_training_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture = temp / 'capture.json'
            fake_orchestrator = temp / 'orchestrator.sh'
            fake_orchestrator.write_text(
                '#!/usr/bin/env bash\n'
                'printf \'{"plan":"%s","dry":"%s"}\\n\' '
                '"$PLAN_ONLY" "$DRY_RUN" > "$CAPTURE"\n',
                encoding='utf-8',
            )
            fake_orchestrator.chmod(0o755)
            paths = {}
            for key, name in (
                ('TRACER_BASE_CHECKPOINT', 'base.pth'),
                ('TRACER_RISK_HEAD_PATH', 'risk.pt'),
                ('TRACER_RISK_AUDIT_PATH', 'audit.json'),
                ('HARMON_GUARD_EXCLUSION_MANIFEST', 'exclude.json'),
            ):
                path = temp / name
                path.write_bytes(b'fixture')
                paths[key] = str(path)
            (temp / 'exclude.json.sha256').write_text(
                'fixture\n', encoding='utf-8')
            env = os.environ.copy()
            env.update(paths)
            env.update({
                'ORCHESTRATOR': str(fake_orchestrator),
                'CAPTURE': str(capture),
                'EXECUTE': '1',
                'MIN_ROOT_FREE_GB': '0',
            })

            completed = subprocess.run(
                ['bash', str(EXP5_TRAIN_ENTRY)],
                cwd=HARMON_ROOT.parent,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertEqual(
                json.loads(capture.read_text(encoding='utf-8')),
                {'plan': '0', 'dry': '0'},
            )

    def test_training_and_inference_share_the_same_risk_head_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            risk = temp / 'risk_head.pt'
            base = temp / 'stable20k.pth'
            exclusion = temp / 'exclusions.json'
            for path in (risk, base, exclusion):
                path.write_bytes(b'contract')
            exclusion.with_name(exclusion.name + '.sha256').write_text(
                'placeholder\n', encoding='utf-8')
            env = {
                'TRACER_RISK_HEAD_PATH': str(risk),
                'TRACER_BASE_CHECKPOINT': str(base),
                'HARMON_GUARD_EXCLUSION_MANIFEST': str(exclusion),
            }
            with patch.dict(os.environ, env, clear=False):
                train = Config.fromfile(TRAIN_CONFIG)
                infer = Config.fromfile(INFER_CONFIG)

        self.assertEqual(train.model.guard_risk_head_path, str(risk))
        self.assertEqual(infer.model.guard_risk_head_path, str(risk))
        self.assertEqual(
            tuple(infer.checkpoint_required_key_groups),
            (
                'proj_in.',
                'lora_A',
                'lora_B',
                'guard_rank_gate.',
                'mask_token_delta.',
            ),
        )
        self.assertEqual(train.model.pretrained_pth, str(base))
        self.assertEqual(
            tuple(train.model.pretrained_required_key_groups),
            ('proj_in.', 'lora_A', 'lora_B', 'mask_token_delta.'),
        )
        self.assertTrue(train.model.tracer_router_enabled)
        self.assertTrue(train.model.tracer_policy_enabled)
        self.assertEqual(train.model.guard_train_rounds, 2)
        self.assertTrue(train.randomness.deterministic)
        self.assertTrue(
            train.tracer_method_contract.deterministic_training)
        self.assertEqual(
            train.tracer_method_contract.rank_gate_initial_beta, 0.10)
        self.assertEqual(
            tuple(train.tracer_method_contract.factorial_cells),
            ('plain', 'router', 'policy', 'coupled'),
        )

    def test_training_config_can_be_deepcopied_by_mmengine_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            risk = temp / 'risk_head.pt'
            base = temp / 'stable20k.pth'
            exclusion = temp / 'exclusions.json'
            for path in (risk, base, exclusion):
                path.write_bytes(b'contract')
            exclusion.with_name(exclusion.name + '.sha256').write_text(
                'placeholder\n', encoding='utf-8')
            env = {
                'TRACER_RISK_HEAD_PATH': str(risk),
                'TRACER_BASE_CHECKPOINT': str(base),
                'HARMON_GUARD_EXCLUSION_MANIFEST': str(exclusion),
            }
            with patch.dict(os.environ, env, clear=False):
                config = Config.fromfile(TRAIN_CONFIG)

        copy.deepcopy(config)

    def test_eval_launcher_registers_all_four_cells_and_single_test_runner(self):
        launcher = (
            HARMON_ROOT / 'scripts' /
            'eval_unimrg_tracer_lora_full.sh'
        ).read_text(encoding='utf-8')
        for cell in ('plain)', 'router)', 'policy)', 'coupled)'):
            self.assertIn(cell, launcher)
        self.assertIn('eval_unimrg_common.sh', launcher)
        self.assertIn('HarmonTRACERLoRAFull_', launcher)
        self.assertNotIn('vanilla_all=True', launcher)

    def test_exp5_eval_instance_config_is_sourceable_and_seed_resolved(self):
        self.assertTrue(EVAL_INSTANCE_CONFIG.is_file())
        command = (
            'set -euo pipefail; '
            f'source {EVAL_INSTANCE_CONFIG}; '
            'printf "%s\\n" "$CHECKPOINT" "$MODEL_CONFIG" '
            '"$TRACER_CELL" "$TRACER_ROUTER_ENABLED" '
            '"$TRACER_POLICY_ENABLED" "$BLOCK_SIZE" '
            '"$DENOISING_STEPS" "$MAX_NEW_TOKENS" "$USE_DLLM" '
            '"$TRACER_RISK_HEAD_PATH" "$TRACER_RISK_AUDIT_PATH" '
            '"$EVAL_PREFLIGHT_SAMPLES" '
            '"$EVAL_PREFLIGHT_REQUIRE_EOS" '
            '"$EVAL_PREFLIGHT_REQUIRE_NONEMPTY"'
        )
        env = os.environ.copy()
        env['TRACER_SEED'] = '41'
        completed = subprocess.run(
            ['bash', '-c', command],
            cwd=HARMON_ROOT.parent,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(
            completed.returncode, 0,
            msg=completed.stdout + completed.stderr,
        )
        values = completed.stdout.splitlines()
        self.assertEqual(len(values), 14)
        self.assertTrue(values[0].endswith(
            'tracer-v2-exp5-hotstartfix-train-s41/work/iter_20000.pth'))
        self.assertTrue(values[1].endswith(
            'Harmon/configs/examples/UniMRG_dllm_lora_tracer_full_infer.py'))
        self.assertEqual(values[2:5], ['coupled', 'true', 'true'])
        self.assertEqual(values[5:9], ['4', '4', '512', 'true'])
        self.assertTrue(values[9].endswith(
            'tracer-v2-exp3-risk/risk/risk_head.pt'))
        self.assertTrue(values[10].endswith(
            'tracer-v2-exp3-risk/audit/risk_audit_report.json'))
        self.assertEqual(values[11:14], ['8', 'true', 'true'])

    def test_exp4_eval_config_rejects_known_invalid_checkpoints(self):
        completed = subprocess.run(
            ['bash', '-c', f'source {INVALID_EXP4_CONFIG}'],
            cwd=HARMON_ROOT.parent,
            env=os.environ.copy(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('known-invalid', completed.stderr)

    def test_orchestrator_plan_records_multiseed_four_cell_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            orchestration_root = temp / 'orchestration'
            env = os.environ.copy()
            env.update({
                'EXPERIMENT_ID': 'tracer-contract-v1',
                'STAGE': 'all',
                'PLAN_ONLY': '1',
                'DRY_RUN': '0',
                'OFFICIAL_EVAL_FROZEN': '1',
                'SEEDS': '41 42',
                'TRACER_CELLS': 'plain router policy coupled',
                'ORCHESTRATION_ROOT': str(orchestration_root),
                'RISK_OUTPUT_ROOT': str(temp / 'risk'),
                'TRAIN_WORK_ROOT': str(temp / 'train'),
                'EVAL_WORK_ROOT': str(temp / 'eval'),
                'TRACER_BASE_CHECKPOINT': str(temp / 'stable20k.pth'),
            })
            completed = subprocess.run(
                ['bash', str(ORCHESTRATOR)],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            run_dir = (
                orchestration_root / 'runs' /
                'tracer-contract-v1-all-plan'
            )
            manifest = json.loads(
                (run_dir / 'manifest.json').read_text(encoding='utf-8')
            )
            self.assertEqual(manifest['method'], 'TRACER-LoRA')
            self.assertEqual(manifest['stage'], 'all')
            self.assertEqual(manifest['seeds'], [41, 42])
            self.assertEqual(
                manifest['tracer_cells'],
                ['plain', 'router', 'policy', 'coupled'],
            )
            self.assertEqual(
                manifest['evaluation_protocol'],
                'canonical_single_test',
            )
            self.assertEqual(manifest['judge'], 'exact_matching')
            self.assertEqual(manifest['preflight_samples'], 8)
            self.assertEqual(
                manifest['revision_budget']['max_revisions_per_round'],
                1,
            )
            self.assertTrue(
                manifest['training_reproducibility']['strict'])
            self.assertEqual(
                manifest['training_reproducibility']['pythonhashseed'],
                'per_training_seed',
            )
            self.assertEqual(
                manifest['provenance']['source_snapshot'],
                'source_snapshot.tar.gz',
            )
            self.assertTrue((run_dir / 'source_snapshot.tar.gz').is_file())
            self.assertEqual(
                manifest['risk_head'],
                str(
                    temp / 'risk' / 'runs' /
                    'tracer-contract-v1-risk' / 'risk' / 'risk_head.pt'
                ),
            )

            command_script = (
                run_dir / 'commands.sh'
            ).read_text(encoding='utf-8')
            self.assertTrue(command_script.startswith(
                '#!/usr/bin/env bash\nset -euo pipefail\n'
            ))
            commands = [
                line for line in command_script.splitlines()
                if line.startswith('env ')
            ]
            self.assertEqual(len(commands), 11)
            eval_commands = [
                line for line in commands
                if 'eval_unimrg_tracer_lora_full.sh' in line
            ]
            self.assertTrue(all(
                'EVAL_PREFLIGHT_SAMPLES=8' in line
                and 'EVAL_PREFLIGHT_REQUIRE_EOS=true' in line
                and 'EVAL_PREFLIGHT_REQUIRE_NONEMPTY=true' in line
                and 'OFFICIAL_EVAL_FROZEN=1' in line
                for line in eval_commands
            ))
            self.assertEqual(
                sum('prepare_tracer_risk.sh' in line for line in commands),
                1,
            )
            self.assertEqual(
                sum('train_tracer_lora.sh' in line for line in commands),
                2,
            )
            self.assertEqual(
                sum(
                    'eval_unimrg_tracer_lora_full.sh' in line
                    for line in commands
                ),
                8,
            )
            self.assertTrue(all('DRY_RUN=0' in line for line in commands))
            self.assertIn(
                str(
                    temp / 'train' / 'runs' /
                    'tracer-contract-v1-train-s41' / 'work' /
                    'iter_20000.pth'
                ),
                '\n'.join(commands),
            )
            status = json.loads(
                (run_dir / 'status.json').read_text(encoding='utf-8')
            )
            self.assertEqual(status['state'], 'planned')
            self.assertFalse((temp / 'risk' / 'runs').exists())

    def test_orchestrator_rejects_unfrozen_formal_eval_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            orchestration_root = temp / 'orchestration'
            env = os.environ.copy()
            env.update({
                'EXPERIMENT_ID': 'unfrozen-eval',
                'STAGE': 'eval',
                'PLAN_ONLY': '1',
                'DRY_RUN': '0',
                'OFFICIAL_EVAL_FROZEN': '0',
                'ORCHESTRATION_ROOT': str(orchestration_root),
            })
            completed = subprocess.run(
                ['bash', str(ORCHESTRATOR)],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                'OFFICIAL_EVAL_FROZEN=1',
                completed.stderr,
            )
            self.assertFalse(orchestration_root.exists())

    def test_orchestrator_strict_run_rejects_missing_data_manifest_early(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestration_root = Path(temp_dir) / 'orchestration'
            env = os.environ.copy()
            env.update({
                'EXPERIMENT_ID': 'missing-data-manifest',
                'STAGE': 'risk',
                'PLAN_ONLY': '0',
                'DRY_RUN': '0',
                'TRACER_STRICT_REPRODUCIBILITY': '1',
                'TRACER_DATA_MANIFEST': '',
                'ORCHESTRATION_ROOT': str(orchestration_root),
            })
            completed = subprocess.run(
                ['bash', str(ORCHESTRATOR)],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn('TRACER_DATA_MANIFEST', completed.stderr)
            self.assertFalse(orchestration_root.exists())

    def test_direct_formal_eval_rejects_unfrozen_checkpoint(self):
        launcher = HARMON_ROOT / 'scripts' / 'eval_unimrg_tracer_lora_full.sh'
        env = os.environ.copy()
        env.update({'DRY_RUN': '0', 'OFFICIAL_EVAL_FROZEN': '0'})
        completed = subprocess.run(
            ['bash', str(launcher)],
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('OFFICIAL_EVAL_FROZEN=1', completed.stderr)

    def test_training_dry_run_records_cli_seed_in_config_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            env = os.environ.copy()
            env.update({
                'STAGE': 'tracer_full',
                'DRY_RUN': '1',
                'SEED': '41',
                'RUN_ID': 'seed-record-contract',
                'WORK_ROOT': str(temp / 'train'),
                'CONDA_ENV': 'harmon',
                'CONDA_SH': '/home/xiexu/anaconda3/etc/profile.d/conda.sh',
                'HARMON_GUARD_EXCLUSION_MANIFEST': (
                    '/nvmedata/xiexu/data/uni/guard_exclusions/'
                    'train_exclusions.json'
                ),
                'TRACER_RISK_HEAD_PATH': str(temp / 'risk_head.pt'),
                'TRACER_RISK_AUDIT_PATH': str(temp / 'risk_audit.json'),
                'TRACER_BASE_CHECKPOINT': str(temp / 'stable20k.pth'),
            })
            completed = subprocess.run(
                ['bash', str(HARMON_ROOT / 'scripts' / 'train_tracer_lora.sh')],
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(
                completed.returncode, 0,
                msg=completed.stdout + completed.stderr,
            )
            summary = json.loads(
                (
                    temp / 'train' / 'runs' /
                    'seed-record-contract' / 'config_summary.json'
                ).read_text(encoding='utf-8')
            )
            self.assertEqual(summary['randomness_seed'], 41)
            self.assertTrue(summary['randomness_deterministic'])


if __name__ == '__main__':
    unittest.main()
