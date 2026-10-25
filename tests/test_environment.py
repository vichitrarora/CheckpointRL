"""Independent edge cases for environment accounting and report definitions."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from checkpoint_rl.environment import Environment
from baselines.policies import baseline_interval
from checkpoint_rl.metrics import summarize, validate_run

C = json.loads((Path(__file__).resolve().parents[1]/'configs/environment.json').read_text())


def world(work=180., cost=10., failures=()):
    n=int(8*work/60)+2
    return dict(id='fixture',split='fixture',work=work,horizon=8*work,
        costs=[cost]*n,current_mtbf_estimates=[3600.]*n,nominal_mtbf=3600.,public_mtbf=3600.,
        block=60.,recovery=20.,failures=list(failures),telemetry=[0.]*n,aging_telemetry=False,
        cell=dict(work_band=0,risk_band=0,cost_band=0,process='stationary'))


def finish(e):
    while not e.done:
        e.step(len(e.grid))
    r=e.metrics();r.update(policy='fixture',seed=1,split='fixture')
    validate_run(r)
    return r


class Contract(unittest.TestCase):
    def test_no_failure_finish_without_write(self):
        r=finish(Environment(world(),C))
        self.assertEqual((r['elapsed_seconds'],r['attempts'],r['episode_return']),(180.,0,0.))

    def test_write_commit_and_no_final_write(self):
        e=Environment(world(),C);e.step(0)
        self.assertEqual((e.core.saved,e.core.t),(60.,70.))
        r=finish(e)
        self.assertEqual((r['writing_seconds'],r['elapsed_seconds'],r['successful_writes']),(10.,190.,1))

    def test_failed_write_keeps_previous_durable_checkpoint(self):
        e=Environment(world(failures=[135.]),C);e.step(0);e.step(0)
        self.assertEqual((e.core.saved,e.core.progress,e.core.failed),(60.,60.,1))
        r=finish(e)
        self.assertEqual((r['lost_work_seconds'],r['failed_write_seconds'],r['usable_failures']),(60.,5.,1))
        self.assertEqual(r['conditional_gaps_seconds'],[65.])

    def test_recovery_failure_restarts_recovery(self):
        e=Environment(world(failures=[65.,75.]),C);e.step(0)
        self.assertEqual((e.core.t,e.core.recovery,e.core.recovery_failures),(95.,30.,1))
        finish(e)

    def test_failure_at_completion_wins_tie(self):
        e=Environment(world(failures=[180.]),C);e.step(len(e.grid))
        self.assertFalse(e.done)
        self.assertEqual((e.core.progress,e.core.t),(0.,200.))
        finish(e)

    def test_timeout_during_write_is_not_a_success_or_failure(self):
        e=Environment(world(cost=2000.),C);e.step(0);r=e.metrics()
        self.assertEqual((r['unfinished_writes'],r['failed_writes'],r['successful_writes']),(1,0,0))
        self.assertEqual((r['elapsed_seconds'],r['penalized_seconds'],r['episode_return']),(1440.,1800.,-9.))
        validate_run(r)

    def test_zero_cost_write_ties_are_not_penalized(self):
        e=Environment(world(cost=0.),C);e.step(0);r=finish(e)
        self.assertEqual((r['elapsed_seconds'],r['attempts'],r['writing_seconds']),(180.,1,0.))

    def test_no_usable_checkpoint_has_no_fake_zero_gap(self):
        e=Environment(world(failures=[30.]),C);r=finish(e)
        s=summarize([r],{'fixture'},{1})
        self.assertIsNone(s['mean_gap_seconds'])
        self.assertEqual((s['usable_pct'],s['protected_pct'],s['failures_without_usable_checkpoint']),(0.,0.,1))

    def test_no_failure_denominators_are_null(self):
        s=summarize([finish(Environment(world(),C))],{'fixture'},{1})
        self.assertIsNone(s['usable_pct']);self.assertIsNone(s['protected_pct'])
        self.assertIsNone(s['lost_seconds_per_active_failure'])

    def test_future_tapes_and_metadata_do_not_change_observation(self):
        a=world();b=deepcopy(a)
        b['failures']=[80.,100.];b['costs'][2]=999.;b['current_mtbf_estimates'][2]=1.
        b['nominal_mtbf']=1.;b['id']='other';b['cell']['risk_band']=2
        ea,eb=Environment(a,C),Environment(b,C)
        ea.step(0);eb.step(0)
        np.testing.assert_array_equal(ea.observation(),eb.observation())
        np.testing.assert_array_equal(ea.mask(),eb.mask())

    def test_invalid_actions_rejected(self):
        for action in [-1,18,1.5,True]:
            with self.assertRaises(ValueError):Environment(world(),C).step(action)
        e=Environment(world(),C)
        with self.assertRaises(ValueError):e.step(3)  # 180 duplicates finish.
        finish(e)
        with self.assertRaises(ValueError):e.step(0)

    def test_formula_baseline_has_same_current_sensor(self):
        e=Environment(world(),C)
        self.assertEqual(baseline_interval('young',e),np.sqrt(2*e.core.cost()*e.core.w['public_mtbf']))

    def test_primary_is_mean_per_job_not_ratio_of_totals(self):
        a=Environment(world(work=100.,cost=100.),C);a.step(0);r1=finish(a)
        r2=finish(Environment(world(work=1000.),C));r2['world_id']='other'
        s=summarize([r1,r2],{'fixture','other'},{1})
        self.assertAlmostEqual(s['per_job_excess_pct'],50.)
        self.assertAlmostEqual(s['p95_excess_pct'],95.)
        self.assertAlmostEqual(s['cvar95_excess_pct'],100.)

    def test_missing_duplicate_and_mixed_panels_rejected(self):
        r=finish(Environment(world(),C))
        for rows,ids,seeds in [([r,r],{'fixture'},{1}),([r],{'fixture','other'},{1}),([r],{'fixture'},{1,2})]:
            with self.assertRaises(ValueError):summarize(rows,ids,seeds)
        b=deepcopy(r);b['policy']='other';b['seed']=2
        with self.assertRaises(ValueError):summarize([r,b],{'fixture'},{1,2})

    def test_incorrect_timeout_scoring_and_nonfinite_rejected(self):
        e=Environment(world(cost=2000.),C);e.step(0);r=e.metrics()
        r['penalized_seconds']=r['elapsed_seconds']
        with self.assertRaises(ValueError):validate_run(r)
        r=finish(Environment(world(),C));r['writing_seconds']=float('nan')
        with self.assertRaises(ValueError):validate_run(r)


if __name__=='__main__':unittest.main()
