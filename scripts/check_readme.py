"""Execute the documented workflow with short training and optional full evaluation."""
from checkpoint_rl.runtime import ROOT, config, new_output, read, sha, write
import argparse
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/readme-check')
    p.add_argument('--full-evaluation', action='store_true', help='Also reproduce all saved-agent results; never full training')
    a = p.parse_args()
    output = new_output(a.output)
    commands = [
        ['-m', 'pip', 'check'],
        ['-m', 'scripts.verify'],
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['train.py', '--agent', 'ppo', '--smoke', '--output', str(output / 'ppo-smoke')],
        ['train.py', '--agent', 'dqn', '--smoke', '--output', str(output / 'dqn-smoke')],
        ['evaluate.py', '--selection', str(output / 'ppo-smoke/selection.json'), '--limit', '81', '--output', str(output / 'ppo-smoke-evaluation')],
        ['evaluate.py', '--selection', str(output / 'dqn-smoke/selection.json'), '--limit', '81', '--output', str(output / 'dqn-smoke-evaluation')],
        ['evaluate.py', '--limit', '81', '--output', str(output / 'evaluation-smoke')],
        ['-m', 'scripts.verify', '--evaluation', str(output / 'evaluation-smoke')],
    ]
    if a.full_evaluation:
        commands.append(['run_all.py', '--output', str(output / 'reproduction')])
    results = []
    for index, command in enumerate(commands):
        print(f'[{index+1}/{len(commands)}] {" ".join(command)}', flush=True)
        start = time.perf_counter()
        log = output / f'check{index+1:02d}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        results.append({'command': ['python', *[s.replace(str(output), '<CHECK_OUTPUT>') for s in command]],
                        'exit_code': result.returncode, 'seconds': time.perf_counter()-start,
                        'log': log.name, 'log_sha256': sha(log)})
        write(output / 'checks.json', {'checks': results, 'full_training_performed': False})
        if result.returncode:
            print(log.read_text(encoding='utf-8'), flush=True)
            raise SystemExit(result.returncode)
        print(f'PASS ({results[-1]["seconds"]:.1f}s)', flush=True)
    smoke = {}
    for arm in ('ppo', 'dqn'):
        selection = read(output / f'{arm}-smoke/selection.json')['selections'][0]
        curve = read(output / f'{arm}-smoke/results/{arm}/seed4401/learning_curve.json')
        if not selection['smoke'] or not curve[-1]['optimizer_steps'] > 0:
            raise ValueError('Smoke run did not perform optimizer updates')
        smoke[arm] = {'decisions': selection['trained_decisions'],
                      'optimizer_steps': curve[-1]['optimizer_steps'],
                      'selected_model_sha256': selection['sha256'],
                      'validation_worlds': selection['validation']['worlds']}
    receipt = {'checks': results, 'smoke_training': smoke, 'full_training_performed': False,
               'runtime': {'python': sys.version, 'platform': platform.platform(),
               **{name: importlib.metadata.version(name) for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib','matplotlib')}},
               'data_manifest_sha256': sha(ROOT / 'data/manifest.json'),
               'model_manifest_sha256': sha(ROOT / 'models/manifest.json')}
    if a.full_evaluation:
        receipt['full_evaluation'] = read(output / 'reproduction/evaluation/verification.json')
    write(output / 'checks.json', receipt)
    print('All README workflow checks passed:', output / 'checks.json')


if __name__ == '__main__':
    main()
