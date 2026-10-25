"""Three evaluation baselines; their outputs are never RL training targets."""
import math


def baseline_interval(name, env):
    if name == 'fixed1200':
        return 1200.
    m = env.core.w['public_mtbf']
    cost = env.core.cost()
    if name == 'young':
        return math.sqrt(2 * cost * m)
    if name == 'daly':
        z = cost / (2 * m)
        return m if cost >= 2 * m else math.sqrt(2 * cost * m) * (1 + math.sqrt(z) / 3 + z / 9) - cost
    raise ValueError('Unknown baseline: ' + name)
