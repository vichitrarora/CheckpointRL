"""Evaluate the five strategies on identical saved worlds."""
from checkpoint_rl.runtime import *
from checkpoint_rl.rollout import rollouts
from checkpoint_rl.metrics import summarize
from agents.actors import load_actor
import argparse
import platform
import sys
import importlib.metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='New output directory, e.g. runs/evaluation')
    p.add_argument('--split', choices=['test', 'ood_test', 'both'], default='both')
    p.add_argument('--policies', choices=POLICIES, nargs='+', default=list(POLICIES))
    p.add_argument('--limit', type=int, help='Smoke only: first N worlds in each split, not a final benchmark')
    p.add_argument('--selection', type=Path, help='Evaluate one newly trained strategy using its selection.json')
    a = p.parse_args()
    if a.limit is not None and a.limit <= 0:
        p.error('--limit must be positive')
    if len(set(a.policies)) != len(a.policies):
        p.error('Duplicate policies')
    verify_assets()
    output = new_output(a.output)
    model_base = a.selection.resolve().parent if a.selection else ROOT
    manifest = read(a.selection or ROOT / 'models/manifest.json')
    selections = manifest['selections']
    if a.selection:
        names = {s['arm'] for s in selections}
        if len(names) != 1:
            p.error('Custom selection must contain exactly one RL strategy')
        policies = sorted(names)
    else:
        policies = a.policies
    splits = ['test', 'ood_test'] if a.split == 'both' else [a.split]
    records = []
    for split in splits:
        worlds = corpus(split)
        if a.limit:
            worlds = worlds[:a.limit]
        for policy in policies:
            panels = [s for s in selections if s['arm'] == policy] if policy in ('ppo', 'dqn') else [None]
            if not panels:
                raise ValueError('Missing selected agent: ' + policy)
            for selection in panels:
                seed = selection['seed'] if selection else 0
                kwargs = dict(actor=load_actor(selection, model_base), augmented=selection['augmented']) if selection else dict(baseline=policy)
                rows = rollouts(worlds, canonical_config(), settings=config(), **kwargs)
                for row in rows:
                    row.update(policy=policy, seed=seed, split=split)
                summary = summarize(rows, {w['id'] for w in worlds}, {seed})
                path = output / 'raw' / f'{split}_{policy}_seed{seed}.jsonl'
                write_ledger(path, rows)
                records.append({'path': path.relative_to(output).as_posix(), 'sha256': sha(path), 'runs': len(rows)})
                print(f'{split} {policy} seed {seed}: excess={summary["per_job_excess_pct"]:.4f}%, completion={summary["completion_pct"]:.2f}%', flush=True)
    write(output / 'evaluation.json', {'files': records, 'policies': policies, 'splits': splits,
          'partial': a.limit is not None, 'custom_selection': a.selection is not None,
          'data_manifest_sha256': sha(ROOT / 'data/manifest.json'),
          'models': selections, 'runtime': {'python': sys.version, 'platform': platform.platform(),
          **{k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib')}}})
    print('Evaluation complete:', output)


if __name__ == '__main__':
    main()
