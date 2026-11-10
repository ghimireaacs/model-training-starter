import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from training_lab import train


class BootstrapTests(unittest.TestCase):
    def test_nvidia_smi_cuda_header_formats(self):
        headers = [
            ('NVIDIA-SMI 580.178.04  Driver Version: 580.178.04  CUDA Version: 13.0', (13, 0)),
            ('NVIDIA-SMI 620.70.03  KMD Version: 621.14  CUDA UMD Version: 13.5', (13, 5)),
            ('CUDA  UMD  Version : 13.5', (13, 5)),
        ]
        for header, expected in headers:
            with self.subTest(header=header), \
                 patch.object(bootstrap.shutil, 'which', return_value='nvidia-smi'), \
                 patch.object(bootstrap.subprocess, 'run', return_value=bootstrap.subprocess.CompletedProcess([], 0, header, '')):
                self.assertEqual(bootstrap.gpu_info(), expected)

    def test_unavailable_cuda_version_is_not_guessed(self):
        with patch.object(bootstrap.shutil, 'which', return_value='nvidia-smi'), \
             patch.object(bootstrap.subprocess, 'run', return_value=bootstrap.subprocess.CompletedProcess([], 0, 'CUDA Version: N/A', '')):
            with self.assertRaisesRegex(RuntimeError, 'Cannot identify'):
                bootstrap.gpu_info()

    def test_cpu_override_never_probes_gpu(self):
        with patch.object(bootstrap, 'gpu_info', side_effect=AssertionError('Must not probe')):
            self.assertEqual(bootstrap.choose_backend('cpu'), 'cpu')

    def test_auto_selects_cpu_without_driver(self):
        with patch.object(bootstrap, 'gpu_info', return_value=None):
            self.assertEqual(bootstrap.choose_backend('auto'), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'No working NVIDIA'):
                bootstrap.choose_backend('cuda')

    def test_cuda_driver_gate(self):
        with patch.object(bootstrap.platform, 'system', return_value='Linux'), \
             patch.object(bootstrap.platform, 'machine', return_value='x86_64'):
            with patch.object(bootstrap, 'gpu_info', return_value=(12, 6)):
                self.assertEqual(bootstrap.choose_backend('auto'), 'cuda')
            with patch.object(bootstrap, 'gpu_info', return_value=(12, 5)):
                with self.assertRaisesRegex(RuntimeError, 'newer NVIDIA driver'):
                    bootstrap.choose_backend('auto')

    def test_supported_wheel_targets(self):
        with patch.object(bootstrap.platform, 'system', return_value='Linux'), \
             patch.object(bootstrap.platform, 'machine', return_value='x86_64'):
            with patch.object(bootstrap.platform, 'libc_ver', return_value=('glibc', '2.31')):
                self.assertIn('manylinux_2_17_x86_64', bootstrap.host_target())
            with patch.object(bootstrap.platform, 'libc_ver', return_value=('musl', '1.2')):
                with self.assertRaisesRegex(RuntimeError, 'Alpine'):
                    bootstrap.host_target()
        with patch.object(bootstrap.platform, 'system', return_value='Windows'), \
             patch.object(bootstrap.platform, 'machine', return_value='AMD64'):
            self.assertEqual(bootstrap.host_target(), 'win_amd64.whl')
        with patch.object(bootstrap.platform, 'system', return_value='Darwin'), \
             patch.object(bootstrap.platform, 'machine', return_value='arm64'):
            self.assertIn('macosx', bootstrap.host_target())

    def test_dependency_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'uv.lock').write_text('original')
            (root / 'pyproject.toml').write_text('project')
            old = bootstrap.fingerprint(root)
            (root / 'uv.lock').write_text('changed')
            self.assertNotEqual(old, bootstrap.fingerprint(root))


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['return_dict'] is False
        return [1, 2]

    def encode(self, text, **kwargs):
        return [3, 4]


class DataTests(unittest.TestCase):
    def test_only_answer_and_eos_are_supervised(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'data.jsonl'
            file.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'prompt'},
                                                     {'role': 'assistant', 'content': 'answer'}]}))
            row = train.read_rows(file, Tokenizer(), 8)[0]
            self.assertEqual(row['labels'], [-100, -100, 3, 4, 9, -100, -100, -100])
            self.assertEqual(row['attention_mask'], [1, 1, 1, 1, 1, 0, 0, 0])
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                train.read_rows(file, Tokenizer(), 4)

    def test_wrong_roles_and_empty_answers_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'data.jsonl'
            for messages in ([{'role': 'assistant', 'content': 'answer'}],
                             [{'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': ' '} ]):
                file.write_text(json.dumps({'messages': messages}))
                with self.assertRaises(ValueError):
                    train.read_rows(file, Tokenizer(), 8)

    def test_smoke_splits_are_distinct(self):
        a = [json.loads(line)['messages'][0]['content'] for line in (ROOT / 'examples/train.jsonl').read_text().splitlines()]
        b = [json.loads(line)['messages'][0]['content'] for line in (ROOT / 'examples/valid.jsonl').read_text().splitlines()]
        self.assertEqual((len(a), len(b)), (16, 4))
        self.assertFalse(set(a) & set(b))


class ResumeTests(unittest.TestCase):
    def test_resume_restores_budget_and_drops_old_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint-10'
            checkpoint.mkdir()
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json', 'adapter_model.safetensors'):
                (checkpoint / name).touch()
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 10}))
            (root / 'run.json').write_text(json.dumps({'settings': train.DEFAULTS, 'revision': 'a' * 40}))
            args, _ = train.parse_args(['--resume', str(checkpoint)])
            self.assertEqual(args.steps, 20)
            self.assertEqual(args.revision, 'a' * 40)
            self.assertIsNone(args.pause_after)
            for extra in (['--steps', '30'], ['--pause-after', '10']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(['--resume', str(checkpoint)] + extra)
            (checkpoint / 'optimizer.pt').unlink()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--resume', str(checkpoint)])

    def test_pause_marker_and_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'PAUSE'
            self.assertFalse(train.pause_requested(2, 20, None, marker))
            self.assertTrue(train.pause_requested(10, 20, 10, marker))
            marker.touch()
            self.assertTrue(train.pause_requested(2, 20, None, marker))
            self.assertFalse(train.pause_requested(20, 20, None, marker))


class PublicationTests(unittest.TestCase):
    def test_package_uses_only_source_trees(self):
        spec = importlib.util.spec_from_file_location('package_source', ROOT / 'tools/package.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        paths = module.source_files()
        self.assertTrue(paths)
        excluded = {'.venv', '.bootstrap', '.cache', '__pycache__', 'runs', 'data', 'models', 'exports', 'dist', '.git'}
        self.assertFalse(any(excluded & set(p.relative_to(ROOT).parts) for p in paths))
        self.assertTrue(all(p.is_file() for p in paths))

class LocalRuntimeTests(unittest.TestCase):
    def test_default_uses_home_not_source(self):
        from training_lab import paths
        with patch.dict('os.environ', {}, clear=True), patch.object(paths.os, 'name', 'posix'), \
             patch.object(paths.Path, 'home', return_value=Path('/tmp/example-user')):
            self.assertEqual(paths.runtime_root(), Path('/tmp/example-user/.local/share/model-training-starter'))

    def test_override_caches_and_staging(self):
        from training_lab import paths
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'MODEL_TRAINING_HOME': directory}):
            root = paths.prepare_runtime()
            environment = paths.runtime_environment(root)
            for key in ('UV_PROJECT_ENVIRONMENT', 'UV_PYTHON_INSTALL_DIR', 'UV_CACHE_DIR',
                        'HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TORCH_HOME', 'TMPDIR'):
                self.assertTrue(Path(environment[key]).is_relative_to(root), key)
            paths.stage_examples(root)
            sample = root / 'examples/train.jsonl'
            self.assertEqual(sample.read_bytes(), (ROOT / 'examples/train.jsonl').read_bytes())
            sample.write_text('keep unfinished experiment input')
            paths.stage_examples(root)
            self.assertEqual(sample.read_text(), 'keep unfinished experiment input')
            with patch.dict('os.environ', {'MODEL_TRAINING_HOME': 'relative-folder'}):
                with self.assertRaisesRegex(ValueError, 'absolute'):
                    paths.runtime_root()


if __name__ == '__main__':
    unittest.main()
