#!/usr/bin/env python3
"""Prepare the project with Python 3.10+ and the standard library only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

sys.dont_write_bytecode = True
from training_lab.paths import prepare_runtime, runtime_root, runtime_environment, stage_examples

ROOT = Path(__file__).resolve().parent
UV_VERSION = '0.13.7'


def python_path(root=None):
    root = runtime_root() if root is None else root
    return root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def fingerprint(root=ROOT):
    return hashlib.sha256((root / 'uv.lock').read_bytes() + (root / 'pyproject.toml').read_bytes()).hexdigest()


def host_target():
    system, machine = platform.system(), platform.machine().lower()
    if system == 'Linux':
        libc, version = platform.libc_ver()
        if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
            raise RuntimeError('Use a glibc 2.28+ distribution (for example Ubuntu 20.04+). Alpine/musl is not supported by these PyTorch wheels.')
        if machine == 'x86_64':
            return 'manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
        if machine in ('aarch64', 'arm64'):
            return 'manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl'
    if system == 'Windows' and machine in ('amd64', 'x86_64'):
        return 'win_amd64.whl'
    if system == 'Darwin' and machine == 'arm64':
        return 'macosx_11_0_arm64.whl'
    raise RuntimeError(f'Unsupported platform: {system} {machine}. See the README support table.')


def gpu_info():
    executable = shutil.which('nvidia-smi')
    if not executable and Path('/usr/lib/wsl/lib/nvidia-smi').exists():
        executable = '/usr/lib/wsl/lib/nvidia-smi'
    if not executable:
        return None
    result = subprocess.run([executable], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('nvidia-smi exists but cannot use the driver. Fix the host driver, or explicitly select --backend cpu for the self-test.')
    match = re.search(r'\bCUDA(?:\s+UMD)?\s+Version\s*:\s*(\d+)\.(\d+)\b', result.stdout)
    if not match:
        raise RuntimeError('Cannot identify driver CUDA support from nvidia-smi. Check the host driver.')
    return tuple(map(int, match.groups()))


def choose_backend(requested):
    if requested == 'cpu':
        return 'cpu'
    capability = gpu_info()
    if capability is None:
        if requested == 'cuda':
            raise RuntimeError('No working NVIDIA driver detected. Install the host driver first; in WSL install it on Windows.')
        return 'cpu'
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('This CUDA recipe supports x86-64 Linux/Windows. Use --backend cpu for the portable self-test.')
    if capability < (12, 6):
        raise RuntimeError('This pinned CUDA 12.6 recipe needs a newer NVIDIA driver. Update the host driver or use --backend cpu.')
    return 'cuda'


def install_uv(target, root=None):
    root = runtime_root() if root is None else root
    destination = root / '.bootstrap' / UV_VERSION / ('uv.exe' if os.name == 'nt' else 'uv')
    if destination.is_file():
        return destination
    print(f'Installing machine-local uv {UV_VERSION} from PyPI...', flush=True)
    with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{UV_VERSION}/json', timeout=60) as response:
        metadata = json.load(response)
    artifact = next(item for item in metadata['urls'] if item['filename'].endswith(target))
    with urllib.request.urlopen(artifact['url'], timeout=120) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != artifact['digests']['sha256']:
        raise RuntimeError('uv wheel checksum mismatch; nothing was executed.')
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        names = [name for name in archive.namelist() if name.endswith('/' + destination.name)]
        if len(names) != 1:
            raise RuntimeError('Unexpected uv wheel layout.')
        executable = archive.read(names[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    temporary.write_bytes(executable)
    temporary.chmod(0o755)
    temporary.replace(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        raise RuntimeError('Launch bootstrap.py with Python 3.10 or newer. It installs the project Python 3.12 itself.')
    target = host_target()
    runtime = prepare_runtime()
    environment = runtime_environment(runtime)
    print(f'Source: {ROOT}\nLocal runtime: {runtime}', flush=True)
    state_path = runtime / '.bootstrap' / 'ready.json'
    existing = json.loads(state_path.read_text()) if state_path.exists() else None
    if existing:
        if args.backend != 'auto' and existing['backend'] != args.backend:
            raise RuntimeError('This folder already uses another backend. Set MODEL_TRAINING_HOME to a new local directory for a different backend; unfinished runs depend on the original environment.')
        backend = existing['backend']
        if existing['fingerprint'] != fingerprint():
            raise RuntimeError('Dependency files changed. Keep this environment for unfinished runs; use a new local MODEL_TRAINING_HOME for the new lockfile.')
        if not python_path().is_file():
            raise RuntimeError('The recorded environment is missing. Restore the local .venv or choose a new MODEL_TRAINING_HOME.')
        print('Environment already prepared; checking it without changing packages.', flush=True)
    else:
        backend = choose_backend(args.backend)
        print(f'Preparing {backend.upper()} environment. No sudo, model download or training job.', flush=True)
        uv = install_uv(target)
        subprocess.run([str(uv), 'sync', '--frozen', '--extra', backend, '--python', '3.12'],
                       cwd=ROOT, env=environment, check=True)
    subprocess.run([str(python_path()), '-m', 'training_lab.doctor', '--backend', backend], cwd=runtime, env=environment, check=True)
    stage_examples(runtime)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({'backend': backend, 'fingerprint': fingerprint(), 'uv': UV_VERSION}, indent=2) + '\n')
    print('\nREADY: ' + ('CUDA training environment' if backend == 'cuda' else 'CPU self-test environment; 4B/8B QLoRA needs a supported NVIDIA GPU'))
    print('Next: python lab.py self-test')
    if backend == 'cuda':
        print('Then: python lab.py train configs/4b-smoke.json --pause-after 10')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError, StopIteration) as error:
        print(f'Bootstrap stopped: {error}', file=sys.stderr)
        sys.exit(1)
