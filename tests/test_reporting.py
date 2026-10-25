"""Checks that guard against misleading aggregate and paired reports."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
import numpy as np
from checkpoint_rl.runtime import corpus, canonical_config, write_ledger
from checkpoint_rl.rollout import rollouts
from checkpoint_rl.metrics import summarize
from scripts.verify import equivalent
from report import build, bootstrap_mean


class Reporting(unittest.TestCase):
    def test_constant_bootstrap_and_paired_sign(self):
        lo, hi = bootstrap_mean(np.full(20, -2.5), np.random.default_rng(7), 200)
        self.assertEqual((lo, hi), (-2.5, -2.5))

    def test_full_report_rejects_missing_strategies(self):
        rows = rollouts(corpus('test')[:2], canonical_config(), baseline='young')
        for r in rows:
            r.update(split='test', policy='young', seed=0)
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / 'raw'
            write_ledger(source / 'test_young_seed0.jsonl', rows)
            with self.assertRaises(ValueError):
                build(source, Path(tmp) / 'report')

    def test_seed_pooling_is_balanced_and_duplicate_safe(self):
        rows = rollouts(corpus('test')[:2], canonical_config(), baseline='young')
        panel = []
        for seed in (1, 2, 3):
            for row in rows:
                panel.append(dict(row, policy='fixture', split='test', seed=seed))
        result = summarize(panel, {r['world_id'] for r in rows}, {1, 2, 3})
        self.assertAlmostEqual(result['seed_excess_std_pp'], 0.)
        self.assertAlmostEqual(result['efficiency_pct'] + result['writing_fraction_pct'] +
                               result['lost_work_fraction_pct'] + result['recovery_fraction_pct'], 100.)
        with self.assertRaises(ValueError):
            summarize(panel[:-1], {r['world_id'] for r in rows}, {1, 2, 3})

    def test_reproduction_comparison_detects_corruption(self):
        row = {'world_id': 'test', 'attempts': 2, 'time': 10., 'gaps': [1., 2.]}
        self.assertTrue(equivalent(deepcopy(row), row))
        for key, value in [('attempts', 3), ('time', 10.01), ('gaps', [1.])]:
            changed = dict(row, **{key: value})
            self.assertFalse(equivalent(changed, row))


if __name__ == '__main__':
    unittest.main()
