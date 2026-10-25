"""Recompute all metrics, paired comparisons and tables from per-run ledgers."""
from checkpoint_rl.runtime import *
from checkpoint_rl.metrics import summarize, table, f
from collections import defaultdict
import argparse
import csv


def export(path, rows):
    write(path.with_suffix('.json'), rows)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def bootstrap_mean(values, rng, replicates):
    means = []
    for start in range(0, replicates, 100):
        indices = rng.integers(len(values), size=(min(100, replicates-start), len(values)))
        means.extend(values[indices].mean(axis=1))
    return np.quantile(means, [.025, .975]).tolist()


def build(source, output, partial=False):
    source, output = Path(source), Path(output)
    files = sorted(source.glob('*.jsonl'))
    if not files:
        raise ValueError('No ledgers at ' + str(source))
    if (source / 'manifest.json').exists():
        receipt = read(source / 'manifest.json')
        if set(receipt['files']) != {p.name for p in files}:
            raise ValueError('Reference panel inventory differs')
        for path in files:
            if sha(path) != receipt['files'][path.name]:
                raise ValueError('Reference ledger checksum mismatch')
    raw = [row for path in files for row in load_worlds(path)]
    grouped = defaultdict(list)
    for row in raw:
        if row['policy'] not in POLICIES or row['split'] not in ('test', 'ood_test'):
            raise ValueError('Unexpected strategy/split in report')
        grouped[row['split'], row['policy']].append(row)
    expected = {(split, policy) for split in ('test', 'ood_test') for policy in POLICIES}
    if not partial and set(grouped) != expected:
        raise ValueError('Incomplete strategy/split inventory; --partial is for smoke reports only')
    settings = config()
    rng = np.random.default_rng(settings['bootstrap_seed'])
    all_metrics, individual, strata, cells, paired = [], [], [], [], []
    maps = defaultdict(dict)
    for (split, policy), rows in sorted(grouped.items()):
        ids = {r['world_id'] for r in rows}
        seeds = {r['seed'] for r in rows}
        if not partial:
            expected_ids = {w['id'] for w in corpus(split)}
            expected_seeds = set(settings['seeds']) if policy in ('ppo', 'dqn') else {0}
        else:
            expected_ids, expected_seeds = ids, seeds
        stats = summarize(rows, expected_ids, expected_seeds)
        values = np.array([np.mean([r['normalized_time'] for r in rows if r['world_id'] == wid])
                           for wid in sorted(ids)])
        maps[split][policy] = dict(zip(sorted(ids), values))
        lo, hi = bootstrap_mean(100*(values-1), rng, settings['bootstrap_replicates'])
        all_metrics.append(dict(split=split, policy=policy, **stats,
                                excess_ci95_low_pct=lo, excess_ci95_high_pct=hi))
        for seed in sorted(seeds):
            individual.append(dict(split=split, policy=policy, seed=seed,
                **summarize([r for r in rows if r['seed'] == seed], ids, {seed})))
        for dimension in ('process', 'work_band', 'risk_band', 'cost_band'):
            for value in sorted({r[dimension] for r in rows}):
                subset = [r for r in rows if r[dimension] == value]
                strata.append(dict(split=split, policy=policy, dimension=dimension, value=value,
                    **summarize(subset, {r['world_id'] for r in subset}, seeds)))
        cell_keys = ('process', 'work_band', 'risk_band', 'cost_band')
        for key in sorted({tuple(r[k] for k in cell_keys) for r in rows}):
            subset = [r for r in rows if tuple(r[k] for k in cell_keys) == key]
            cells.append(dict(split=split, policy=policy, **dict(zip(cell_keys, key)),
                **summarize(subset, {r['world_id'] for r in subset}, seeds)))
    for split, policies in sorted(maps.items()):
        for policy in ('ppo', 'dqn'):
            if policy not in policies:
                continue
            for reference in POLICIES:
                if reference == policy or reference not in policies or (policy == 'dqn' and reference == 'ppo'):
                    continue
                ids = sorted(policies[policy])
                if set(ids) != set(policies[reference]):
                    raise ValueError('Paired comparison requires identical worlds')
                delta = np.array([100*(policies[policy][w] - policies[reference][w]) for w in ids])
                lo, hi = bootstrap_mean(delta, rng, settings['bootstrap_replicates'])
                paired.append(dict(split=split, policy=policy, reference=reference,
                    difference_excess_pp=float(delta.mean()), ci95_low_pp=lo, ci95_high_pp=hi,
                    world_win_pct=float(100*np.mean(delta < -1e-10)),
                    world_tie_pct=float(100*np.mean(abs(delta) <= 1e-10))))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('all_metrics', all_metrics), ('individual_seeds', individual),
                       ('stratum_metrics', strata), ('cell_metrics', cells), ('paired_comparisons', paired)]:
        export(output / name, rows)
    write(output / 'timeouts.json', [r for r in raw if not r['completed']])
    write(output / 'report_manifest.json', {'partial': partial, 'runs': len(raw),
          'input_sha256': {p.name: sha(p) for p in files},
          'bootstrap_replicates': settings['bootstrap_replicates'],
          'bootstrap_seed': settings['bootstrap_seed']})
    lines = ['# Checkpoint scheduling results', '',
        '**Partial smoke report; not final benchmark evidence.**' if partial else
        'All three saved RL seeds; each baseline is evaluated once on the same worlds.', '',
        'Rank by mean per-job excess within each split. Lower is better. Completed P=T; '
        'unfinished jobs stop at 8W and receive P=10W. E=100(P/W-1).', '',
        'These test suites were previously evaluated. This is a measured benchmark comparison, '
        'not an unseen confirmation. RL seed pooling is not an ensemble.', '']
    for split in ('test', 'ood_test'):
        rows = sorted([r for r in all_metrics if r['split'] == split], key=lambda r: r['per_job_excess_pct'])
        if not rows:
            continue
        lines += [f'## {split}', '', table(
            ['Policy', 'Excess %', 'Complete %', 'Timeouts / runs', 'P95 excess %', 'CVaR95 excess %', 'Penalized s'],
            [[LABELS[r['policy']], f(r['per_job_excess_pct']), f(r['completion_pct']),
              f"{r['timeouts']} / {r['runs']}", f(r['p95_excess_pct']), f(r['cvar95_excess_pct']),
              f(r['mean_penalized_seconds'])] for r in rows]), '', '### Cost and checkpoint diagnostics', '', table(
            ['Policy', 'Attempts/job', 'Writing s/job', 'Lost work s/job', 'Recovery s/job', 'Efficiency %', 'Usable %', 'Protected %'],
            [[LABELS[r['policy']], *[f(r[k]) for k in ('attempts_per_job', 'writing_seconds_per_job',
              'lost_work_seconds_per_job', 'recovery_seconds_per_job', 'efficiency_pct', 'usable_pct', 'protected_pct')]] for r in rows]),
            '', '### Seed stability and paired comparisons', '', table(
            ['Policy', 'Seed SD pp', 'Seed min excess %', 'Seed max excess %'],
            [[LABELS[r['policy']], *[f(r[k]) for k in ('seed_excess_std_pp', 'seed_excess_min_pct', 'seed_excess_max_pct')]] for r in rows]), '', table(
            ['Policy', 'Reference', 'Difference pp', '95% interval pp', 'World wins %'],
            [[LABELS[r['policy']], LABELS[r['reference']], f(r['difference_excess_pp']),
              f"{f(r['ci95_low_pp'])} to {f(r['ci95_high_pp'])}", f(r['world_win_pct'])]
             for r in paired if r['split'] == split]), '']
    lines += ['Intervals resample worlds after averaging RL seeds within each world; '
              'they are pointwise and conditional on the saved models, with no multiple-comparison adjustment.', '',
              'Full metrics, individual seeds, every process/work/risk/cost slice and all 81 cells '
              'are included in the adjacent JSON/CSV files. Counts and protection are diagnostics.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Reported {len(raw):,} runs in {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'results/reference')
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    build(args.input, args.output, args.partial)
