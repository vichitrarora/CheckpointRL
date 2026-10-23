"""Paths, configuration and portable artifact I/O."""
import os

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import hashlib
import json
from pathlib import Path
import numpy as np
from .core import load_worlds

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('ppo', 'dqn', 'young', 'daly', 'fixed1200')
LABELS = {'ppo': 'PPO (grid + fractions)', 'dqn': 'Double DQN (trained grid)',
          'young': 'Young', 'daly': 'Daly', 'fixed1200': 'Fixed 1,200 s'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def config():
    settings = read(ROOT / 'configs/training.json')
    for algorithm in ('dqn', 'ppo'):
        settings[algorithm] = read(ROOT / 'configs' / (algorithm + '.json'))
    return settings


def canonical_config():
    return read(ROOT / 'configs/environment.json')


def corpus(split):
    if split not in ('train', 'validation', 'test', 'ood_test'):
        raise ValueError('Unknown split: ' + split)
    return load_worlds(ROOT / 'data' / (split + '.jsonl'))


def verify_assets(include_models=True):
    manifest = read(ROOT / 'data/manifest.json')
    for name, record in manifest['files'].items():
        path = ROOT / 'data' / name
        if sha(path) != record['sha256']:
            raise ValueError('Dataset checksum mismatch: ' + name)
        if len(load_worlds(path)) != record['worlds']:
            raise ValueError('Dataset row count mismatch: ' + name)
    if include_models:
        for model in read(ROOT / 'models/manifest.json')['selections']:
            if sha(ROOT / model['model']) != model['sha256']:
                raise ValueError('Model checksum mismatch: ' + model['model'])
    return manifest


def write_ledger(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def new_output(path):
    path = Path(path).resolve()
    for protected in ('data', 'models', 'results', 'configs', 'checkpoint_rl', 'agents', 'tests', 'figures'):
        fixed = ROOT / protected
        if path == fixed or fixed in path.parents or path in fixed.parents:
            raise ValueError('Choose a new output directory, e.g. runs/check1')
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output already contains files; choose a new --output directory')
    path.mkdir(parents=True, exist_ok=True)
    return path
