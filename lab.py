#!/usr/bin/env python3
"""Run training tasks in the project environment without activating it."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Check the installed environment and hardware')
    sub.add_parser('self-test', help='Test pause/resume on a tiny random CPU model, without model downloads')
    train = sub.add_parser('train', help='Start a new experiment from a JSON configuration')
    train.add_argument('config')
    train.add_argument('--pause-after', type=int)
    train.add_argument('--check-data', action='store_true')
    resume = sub.add_parser('resume', help='Restore a complete training checkpoint')
    resume.add_argument('checkpoint')
    resume.add_argument('--pause-after', type=int)
    pause = sub.add_parser('pause', help='Ask a running job to save and exit after its next optimizer step')
    pause.add_argument('run')
    generate = sub.add_parser('generate', help='Compare a finished adapter with its base')
    generate.add_argument('run')
    generate.add_argument('prompt')
    generate.add_argument('--base-only', action='store_true')
    merge = sub.add_parser('merge', help='Merge a completed adapter on CPU for later conversion')
    merge.add_argument('run')
    merge.add_argument('output')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'pause':
        run = Path(args.run)
        if not (run / 'run.json').is_file():
            parser.error('Run manifest not found; start the run before requesting a pause')
        if (run / 'adapter').exists():
            parser.error('This run already completed')
        (run / 'PAUSE').touch()
        print('Pause requested. Wait for PAUSED and process exit in the training terminal.')
        return
    python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        if not python.is_file():
            parser.error('Run python bootstrap.py first')
        raise SystemExit(subprocess.call([str(python), str(ROOT / 'lab.py'), *sys.argv[1:]], cwd=ROOT))
    if args.command == 'doctor':
        from training_lab.doctor import main as doctor
        state = json.loads((ROOT / '.bootstrap' / 'ready.json').read_text())
        doctor(['--backend', state['backend']])
    elif args.command == 'self-test':
        from training_lab.selftest import main as selftest
        selftest()
    elif args.command in ('train', 'resume'):
        from training_lab.train import DEFAULTS, main as train_model
        if args.command == 'resume':
            command = ['--resume', args.checkpoint]
        else:
            config = json.loads(Path(args.config).read_text())
            unknown = set(config) - set(DEFAULTS)
            if unknown:
                parser.error('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
            command = []
            for key, value in config.items():
                command.extend(['--' + key.replace('_', '-'), str(value)])
            if args.check_data:
                command.append('--check-data')
        if args.pause_after is not None:
            command.extend(['--pause-after', str(args.pause_after)])
        train_model(command)
    else:
        from training_lab.artifacts import generate, merge
        if args.command == 'generate':
            generate(args.run, args.prompt, args.base_only)
        else:
            merge(args.run, args.output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Stopped: {error}', file=sys.stderr)
        sys.exit(1)
