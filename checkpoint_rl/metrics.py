"""Frozen benchmark formulas plus validated, balanced-panel reporting.

aggregate() preserves historical names and formulas. Use summarize() for new
reports: it rejects missing/duplicate world-seed runs and incompatible panels.
"""
from collections import defaultdict
import math
import numpy as np

def aggregate(rows):
    mean=lambda k:float(np.mean([r[k] for r in rows]))
    total=lambda k:sum(r[k] for r in rows)
    p=np.array([r['penalized_seconds'] for r in rows]);n=np.array([r['normalized_time'] for r in rows])
    gaps=[g for r in rows for g in r['conditional_gaps_seconds']]
    count=total('active_failures');tail=max(1,math.ceil(.05*len(rows)))
    out={'runs':len(rows),'worlds':len({r['world_id'] for r in rows}),
         'mean_normalized_time':float(n.mean()),'per_job_excess_pct':float(100*(n.mean()-1)),
         'mean_elapsed_seconds':mean('elapsed_seconds'),'mean_penalized_seconds':float(p.mean()),
         'completion_pct':100*mean('completed'),'timeouts':len(rows)-total('completed'),
         'median_seconds':float(np.median(p)),'p95_seconds':float(np.quantile(p,.95)),
         'cvar95_seconds':float(np.sort(p)[-tail:].mean()),
         'p95_excess_pct':float(100*(np.quantile(n,.95)-1)),
         'cvar95_excess_pct':float(100*(np.sort(n)[-tail:].mean()-1)),
         'efficiency_pct':100*total('progress_seconds')/total('elapsed_seconds'),
         'active_failures':count,'recovery_failures':total('recovery_failures'),
         'usable_pct':100*total('usable_failures')/count if count else None,
         'protected_pct':100*total('protected_failures')/count if count else None,
         'lost_seconds_per_active_failure':total('lost_work_seconds')/count if count else None,
         'mean_gap_seconds':float(np.mean(gaps)) if gaps else None,
         'median_gap_seconds':float(np.median(gaps)) if gaps else None,
         'p95_gap_seconds':float(np.quantile(gaps,.95)) if gaps else None,
         'max_accounting_residual':max(abs(r['accounting_residual']) for r in rows),
         'max_reward_identity_error':max(abs(r['episode_return']-(1-r['normalized_time'])) for r in rows)}
    for key in ['attempts','successful_writes','failed_writes','unfinished_writes','writing_seconds',
                'failed_write_seconds','lost_work_seconds','recovery_seconds','used_writes',
                'observed_unused_writes','compute_actions','checkpoint_actions']:
        out[key+'_per_job']=mean(key)
    return out

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

def f(x):return 'N/A' if x is None else f'{x:,.2f}'


def validate_run(row):
    """Check measured time, timeout scoring, counters and reward identity."""
    numeric = ['work_seconds', 'elapsed_seconds', 'penalized_seconds',
               'normalized_time', 'progress_seconds', 'writing_seconds',
               'failed_write_seconds', 'lost_work_seconds', 'recovery_seconds']
    if any(not np.isfinite(row[k]) or row[k] < 0 for k in numeric):
        raise ValueError('Nonfinite or negative time/progress')
    w, t, completed = row['work_seconds'], row['elapsed_seconds'], row['completed']
    if w <= 0 or completed not in (0, 1):
        raise ValueError('Invalid work or completion flag')
    if not 0 <= row['progress_seconds'] <= w + 1e-7 or t > 8*w + 1e-7:
        raise ValueError('Progress or elapsed time outside environment bounds')
    if completed and abs(row['progress_seconds'] - w) > 1e-6:
        raise ValueError('Completed job has unfinished work')
    if not completed and (abs(t - 8*w) > 1e-6 or row['progress_seconds'] >= w - 1e-7):
        raise ValueError('Timeout must be an unfinished job at the time cap')
    score = t if completed else 10*w
    if abs(row['penalized_seconds'] - score) > 1e-6:
        raise ValueError('Incorrect timeout score')
    if abs(row['normalized_time'] - score/w) > 1e-10:
        raise ValueError('Incorrect normalized time')
    residual = t - sum(row[k] for k in ['progress_seconds', 'lost_work_seconds',
                                        'writing_seconds', 'recovery_seconds'])
    if (not np.isfinite(row['accounting_residual']) or abs(residual) > 1e-6
            or abs(row['accounting_residual']-residual) > 1e-6):
        raise ValueError('Time conservation failed')
    if not np.isfinite(row['episode_return']) or abs(row['episode_return']-(1-score/w)) > 1e-8:
        raise ValueError('Reward identity failed')
    counters = ['attempts', 'successful_writes', 'failed_writes', 'unfinished_writes',
                'active_failures', 'recovery_failures', 'usable_failures',
                'protected_failures', 'used_writes', 'observed_unused_writes',
                'compute_actions', 'checkpoint_actions']
    if any(not np.isfinite(row[k]) or row[k] < 0 or int(row[k]) != row[k] for k in counters):
        raise ValueError('Invalid counter')
    if row['attempts'] != sum(row[k] for k in ['successful_writes','failed_writes','unfinished_writes']):
        raise ValueError('Write counts do not reconcile')
    if row['checkpoint_actions'] != row['attempts']:
        raise ValueError('Checkpoint action count differs from attempts')
    if not 0 <= row['protected_failures'] <= row['usable_failures'] <= row['active_failures']:
        raise ValueError('Failure counters do not reconcile')
    if row['used_writes'] + row['observed_unused_writes'] != row['successful_writes']:
        raise ValueError('Successful-write usage does not reconcile')
    if row['used_writes'] > row['usable_failures']:
        raise ValueError('More distinct recovered checkpoints than usable failures')
    if row['failed_write_seconds'] > row['writing_seconds'] + 1e-7:
        raise ValueError('Failed-write time exceeds all write time')
    gaps = row['conditional_gaps_seconds']
    if len(gaps) != row['usable_failures'] or any(not np.isfinite(g) or g < 0 for g in gaps):
        raise ValueError('Invalid conditional gaps')


def summarize(rows, expected_world_ids, expected_seeds):
    """One policy and split; each requested seed must cover every world exactly once."""
    if not rows or not expected_world_ids or not expected_seeds:
        raise ValueError('Empty evaluation panel')
    if len({r['policy'] for r in rows}) != 1 or len({r['split'] for r in rows}) != 1:
        raise ValueError('Do not pool policies or splits')
    expected = {(w, s) for w in expected_world_ids for s in expected_seeds}
    keys = [(r['world_id'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicate or unexpected world/seed run')
    by_world = defaultdict(list)
    for row in rows:
        validate_run(row)
        by_world[row['world_id']].append(row)
    for panel in by_world.values():
        for key in ['work_seconds', 'process', 'risk_band', 'cost_band', 'work_band']:
            if len({r[key] for r in panel}) != 1:
                raise ValueError('World definition differs across seeds')
    out = aggregate(rows)
    elapsed = sum(r['elapsed_seconds'] for r in rows)
    out.update(training_seed_count=len(expected_seeds),
               failures_without_usable_checkpoint=sum(r['active_failures']-r['usable_failures'] for r in rows),
               usable_failures=sum(r['usable_failures'] for r in rows),
               protected_failures=sum(r['protected_failures'] for r in rows))
    for key in ['writing_seconds', 'lost_work_seconds', 'recovery_seconds']:
        out[key.replace('_seconds', '_fraction_pct')] = 100*sum(r[key] for r in rows)/elapsed
    seed_means = [np.mean([r['normalized_time'] for r in rows if r['seed']==s]) for s in expected_seeds]
    out['seed_excess_std_pp'] = float(100*np.std(seed_means, ddof=1)) if len(seed_means)>1 else None
    out['seed_excess_min_pct'] = float(100*(min(seed_means)-1))
    out['seed_excess_max_pct'] = float(100*(max(seed_means)-1))
    # Seed/world pooling here is balanced; raw event totals remain descriptive.
    return out
