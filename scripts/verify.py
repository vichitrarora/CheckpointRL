"""Verify packaged assets and optionally compare a fresh evaluation to the reference."""
from checkpoint_rl.runtime import *
from checkpoint_rl.metrics import summarize
import argparse


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-7))
    return actual == expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, help='A fresh evaluate.py output directory')
    a = p.parse_args()
    verify_assets()
    ids = set()
    for split in ('train', 'validation', 'test', 'ood_test'):
        worlds = corpus(split)
        new_ids = {w['id'] for w in worlds}
        if len(new_ids) != len(worlds) or ids & new_ids or any(w['split'] != split for w in worlds):
            raise ValueError('Duplicate IDs or invalid split membership')
        ids |= new_ids
        for w in worlds:
            if w['horizon'] != 8*w['work'] or w['failures'] != sorted(w['failures']):
                raise ValueError('Invalid episode horizon or failure sequence')
    refs = ROOT / 'results/reference'
    manifest = read(refs / 'manifest.json')
    total = 0
    for name, digest in manifest['files'].items():
        path = refs / name
        if sha(path) != digest:
            raise ValueError('Changed reference: ' + name)
        rows = load_worlds(path)
        split, policy = rows[0]['split'], rows[0]['policy']
        if policy not in POLICIES:
            raise ValueError('Unexpected strategy')
        summarize(rows, {w['id'] for w in corpus(split)}, {rows[0]['seed']})
        total += len(rows)
    if total != manifest['runs']:
        raise ValueError('Reference run count mismatch')
    compared = 0
    if a.evaluation:
        metadata = read(a.evaluation / 'evaluation.json')
        if metadata['custom_selection']:
            raise ValueError('Newly trained agents are not the shipped reference agents')
        actual_files = {p.name for p in (a.evaluation / 'raw').glob('*.jsonl')}
        if actual_files != {Path(p['path']).name for p in metadata['files']}:
            raise ValueError('Evaluation inventory mismatch')
        full = not metadata['partial'] and set(metadata['policies']) == set(POLICIES) and set(metadata['splits']) == {'test', 'ood_test'}
        if full and actual_files != set(manifest['files']):
            raise ValueError('Incomplete full evaluation')
        for record in metadata['files']:
            path = a.evaluation / record['path']
            if sha(path) != record['sha256']:
                raise ValueError('Changed evaluation panel')
            expected = {r['world_id']: r for r in load_worlds(refs / path.name)}
            rows = load_worlds(path)
            if len(rows) != record['runs'] or len({r['world_id'] for r in rows}) != len(rows):
                raise ValueError('Evaluation count/duplicate mismatch')
            if not metadata['partial'] and {r['world_id'] for r in rows} != set(expected):
                raise ValueError('Evaluation missing worlds')
            for row in rows:
                if not equivalent(row, expected[row['world_id']]):
                    raise ValueError('Reproduction mismatch: ' + path.name + ' / ' + row['world_id'])
                compared += 1
        if full and compared != total:
            raise ValueError('Incomplete reproduction')
        write(a.evaluation / 'verification.json', {'reference_runs': total, 'compared_runs': compared,
              'full_reproduction': full, 'float_rtol': 1e-10, 'float_atol': 1e-7})
    print(f'PASS: 8,100 worlds, six models, {total:,} reference runs; {compared:,} fresh runs matched.')


if __name__ == '__main__':
    main()
