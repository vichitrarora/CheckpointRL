"""Causal observations and numeric computation-interval actions."""
import math
import numpy as np
from .core import Environment as Core

FEATURES = ['remaining_fraction', 'log_remaining', 'log_work', 'log_cost',
            'log_nominal_mtbf', 'log_current_mtbf_estimate', 'cost_over_work',
            'remaining_over_nominal_mtbf', 'remaining_over_current_mtbf',
            'recovery_over_work', 'elapsed_over_cap', 'age_over_nominal_mtbf',
            'observed_failure_rate', 'block_phase', 'log_block_seconds', 'has_checkpoint']


class Environment:
    def __init__(self, world, config):
        self.core = Core(world, dict(config))
        self.grid = np.array(config['interval_actions_seconds'], dtype=float)
        self.decisions = 0
        self.action_counts = np.zeros(len(self.grid)+1, int)

    @property
    def done(self): return self.core.done

    @property
    def completed(self): return self.core.completed

    def current_mtbf(self):
        e = self.core
        return e.w['current_mtbf_estimates'][min(int(e.t/e.w['block']),len(e.w['costs'])-1)]

    def mask(self):
        # Durations >= remaining work are duplicates of the explicit finish action.
        remaining = self.core.work-self.core.progress
        return np.r_[self.grid < remaining-1e-7, True]

    def observation(self):
        e = self.core
        rem = max(0.,e.work-e.progress)
        m = e.w['public_mtbf']; current = self.current_mtbf(); cost=e.cost()
        return np.array([rem/e.work, math.log1p(rem)/12., math.log1p(e.work)/12.,
            math.log1p(cost)/9., math.log1p(m)/14., math.log1p(current)/14.,
            min(cost/e.work,1.), min(rem/m,8.)/4., min(rem/current,16.)/4.,
            e.w['recovery']/e.work, e.t/e.cap, min((e.t-e.last_failure)/m,8.)/4.,
            min((e.active_failures+e.recovery_failures)*e.work/(e.t+e.work),8.)/4.,
            (e.t % e.w['block'])/e.w['block'], math.log1p(e.w['block'])/12.,
            float(e.saved>0.)], dtype=np.float32)

    def advance(self, duration):
        """Compute for duration, then write unless finished or interrupted by failure.

        A failure ends the chosen interval after recovery; there is no formula override.
        Checkpoint cost is latched when the write starts, and failures during it roll back.
        """
        e = self.core
        before_failures=e.active_failures
        e.config['compute_quantum_seconds']=float(duration)
        _, r, _, _ = e.step(0)
        if not e.done and e.active_failures == before_failures:
            _, write_r, _, _ = e.step(1)
            r += write_r
        self.decisions += 1
        return self.observation(), float(r), e.done, self.mask()

    def step(self, action):
        if (not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_))
                or action < 0 or action > len(self.grid)
                or self.done or not self.mask()[action]):
            raise ValueError('Illegal interval action')
        self.action_counts[action] += 1
        duration=self.grid[action] if action<len(self.grid) else self.core.work-self.core.progress
        return self.advance(duration)

    def metrics(self):
        return {**self.core.metrics(), 'interval_decisions':self.decisions,
                'interval_action_counts':self.action_counts.tolist()}
