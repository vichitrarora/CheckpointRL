"""Shared rollout and Gymnasium adapters for the retained RL strategies."""
from .runtime import *
from .environment import Environment, FEATURES
from .metrics import validate_run
from baselines.policies import baseline_interval

class ActionEnv(Environment):
    """Append known remaining-work fractions; preserve canonical action indices."""
    def __init__(self, world, canonical, augmented=False, settings=None):
        super().__init__(world, canonical)
        settings = settings or config()
        self.fractions = np.array(settings['fractions'] if augmented else [], dtype=float)
        self.minimum = settings['minimum_fraction_duration']
        self.action_counts = np.zeros(len(self.grid) + 1 + len(self.fractions), dtype=int)

    def mask(self):
        base = super().mask()
        rem = self.core.work - self.core.progress
        return np.r_[base, self.fractions * rem >= self.minimum]

    def step(self, action):
        if (not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_))
                or action < 0 or action >= len(self.action_counts) or self.done or not self.mask()[action]):
            raise ValueError('Illegal action')
        if action <= len(self.grid):
            return super().step(action)
        self.action_counts[action] += 1
        rem = self.core.work - self.core.progress
        return self.advance(float(rem * self.fractions[action - len(self.grid) - 1]))


def rollouts(worlds, canonical, actor=None, baseline=None, augmented=False, settings=None):
    if (actor is None) == (baseline is None):
        raise ValueError('Exactly one actor or baseline required')
    out = []; cursor = 0; active = []
    while cursor < len(worlds) or active:
        while len(active) < 64 and cursor < len(worlds):
            active.append(ActionEnv(worlds[cursor], canonical, augmented, settings)); cursor += 1
        if actor is not None:
            actions = actor.actions(np.array([e.observation() for e in active]),
                                    np.array([e.mask() for e in active]))
        live = []
        for i, env in enumerate(active):
            if actor is None:
                env.advance(baseline_interval(baseline, env))
            else:
                env.step(int(actions[i]))
            if env.done:
                row = env.metrics(); validate_run(row); out.append(row)
            else:
                live.append(env)
        active = live
    return sorted(out, key=lambda r: r['world_id'])


def gym_environment(worlds, canonical, settings, augmented, seed):
    import gymnasium as gym
    from gymnasium import spaces
    class CheckpointGym(gym.Env):
        def __init__(self):
            self.rng = np.random.default_rng(seed)
            self.action_space = spaces.Discrete(len(canonical['interval_actions_seconds']) + 1 +
                                               (len(settings['fractions']) if augmented else 0))
            self.observation_space = spaces.Box(0., np.inf, shape=(len(FEATURES),), dtype=np.float32)
            self.env = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.env = ActionEnv(worlds[int(self.rng.integers(len(worlds)))], canonical, augmented, settings)
            return self.env.observation(), {}

        def action_masks(self):
            return self.env.mask()

        def step(self, action):
            obs, reward, done, _ = self.env.step(int(action))
            info = {'world_id': self.env.core.w['id'], 'cell': self.env.core.w['cell'],
                    'chosen_action': int(action)}
            if done:
                info['results'] = self.env.metrics()
            return obs, reward * settings['reward_scale'], done, False, info
    return CheckpointGym()
