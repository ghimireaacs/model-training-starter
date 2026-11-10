"""Make a source-only archive from an explicit allowlist."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'LICENSE', 'AGENTS.md', '.gitignore', '.python-version',
         'pyproject.toml', 'uv.lock', 'bootstrap.py', 'lab.py']
TREES = {'.github/workflows': {'.yml', '.yaml'}, 'configs': {'.json'},
         'examples': {'.jsonl'}, 'training_lab': {'.py'}, 'tests': {'.py'}, 'tools': {'.py'}}


def source_files():
    paths = [ROOT / name for name in FILES]
    for directory, suffixes in TREES.items():
        paths.extend(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts)
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Public source must be a regular in-tree file: {path.name}')
    return sorted(paths)


def main():
    destination = ROOT / 'dist/model-training-starter.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, 'model-training-starter/' + path.relative_to(ROOT).as_posix())
    print(destination)


if __name__ == '__main__':
    main()
