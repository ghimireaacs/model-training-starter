"""Shared source, machine-local environments and training data."""
import os
from pathlib import Path
import shutil

SOURCE = Path(__file__).resolve().parents[1]


def runtime_root():
    override = os.environ.get('MODEL_TRAINING_HOME')
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError('MODEL_TRAINING_HOME must be an absolute path on local disk')
        return path.resolve()
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'model-training-starter'
    return Path.home() / '.local/share/model-training-starter'


def prepare_runtime():
    root = runtime_root()
    for name in ('.bootstrap', 'cache/uv', 'cache/huggingface', 'cache/datasets',
                 'cache/torch', 'tmp', 'data', 'runs', 'exports', 'examples'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def runtime_environment(root):
    env = os.environ.copy()
    env.update(UV_PYTHON_INSTALL_DIR=str(root / '.bootstrap/python'),
               UV_PROJECT_ENVIRONMENT=str(root / '.venv'), UV_PYTHON_PREFERENCE='only-managed',
               UV_CACHE_DIR=str(root / 'cache/uv'), HF_HOME=str(root / 'cache/huggingface'),
               HF_HUB_CACHE=str(root / 'cache/huggingface/hub'),
               HF_DATASETS_CACHE=str(root / 'cache/datasets'), TORCH_HOME=str(root / 'cache/torch'),
               TMPDIR=str(root / 'tmp'), TEMP=str(root / 'tmp'), TMP=str(root / 'tmp'),
               PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE='1')
    return env


def stage_examples(root):
    for source in (SOURCE / 'examples').glob('*.jsonl'):
        target = root / 'examples' / source.name
        if not target.exists():
            shutil.copyfile(source, target)
