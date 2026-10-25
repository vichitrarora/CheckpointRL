"""Meaningful action/physics/learning-contract checks before source freeze."""
import unittest
from copy import deepcopy
from checkpoint_rl.runtime import *
from checkpoint_rl.environment import Environment, FEATURES
from checkpoint_rl.rollout import ActionEnv, rollouts, gym_environment
from checkpoint_rl.metrics import validate_run


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = corpus('validation')
        cls.canonical = canonical_config(); cls.settings = config()

    def test_canonical_action_trajectories_preserved(self):
        rng = np.random.default_rng(902)
        for world in self.worlds[:81]:
            a = Environment(world, self.canonical)
            b = ActionEnv(world, self.canonical, False, self.settings)
            c = ActionEnv(world, self.canonical, True, self.settings)
            while not a.done:
                np.testing.assert_array_equal(a.observation(), b.observation())
                np.testing.assert_array_equal(a.observation(), c.observation())
                np.testing.assert_array_equal(a.mask(), b.mask())
                np.testing.assert_array_equal(a.mask(), c.mask()[:18])
                action = int(rng.choice(np.flatnonzero(a.mask())))
                out = a.step(action); other = b.step(action); extended = c.step(action)
                self.assertEqual(out[1:3], other[1:3]); self.assertEqual(out[1:3], extended[1:3])
            self.assertEqual(a.metrics(), b.metrics())
            extended = c.metrics(); extended['interval_action_counts'] = extended['interval_action_counts'][:18]
            self.assertEqual(a.metrics(), extended)

    def test_fraction_duration_exact_and_causal(self):
        w = deepcopy(self.worlds[0]); w['failures'] = []; w['costs'] = [7.] * len(w['costs'])
        env = ActionEnv(w, self.canonical, True, self.settings)
        _, reward, done, _ = env.step(21)  # one quarter of remaining work
        self.assertFalse(done)
        self.assertAlmostEqual(env.core.progress, w['work'] * .25)
        self.assertAlmostEqual(env.core.t, w['work'] * .25 + 7.)
        self.assertAlmostEqual(reward, -7. / w['work'])
        other = deepcopy(w); other['failures'] = [100000.]
        other['costs'][1:] = [999.] * (len(w['costs'])-1)
        other['current_mtbf_estimates'][1:] = [1.] * (len(w['costs'])-1)
        a = ActionEnv(w, self.canonical, True, self.settings)
        b = ActionEnv(other, self.canonical, True, self.settings)
        np.testing.assert_array_equal(a.observation(), b.observation())
        np.testing.assert_array_equal(a.mask(), b.mask())

    def test_fraction_random_trajectories_accounting(self):
        rng = np.random.default_rng(912)
        for w in self.worlds[:81]:
            e = ActionEnv(w, self.canonical, True, self.settings)
            while not e.done:
                _, _, _, mask = e.step(int(rng.choice(np.flatnonzero(e.mask()))))
                self.assertTrue(mask.any())
            validate_run(e.metrics())

    def test_fraction_mask_and_invalid_inputs(self):
        w = deepcopy(self.worlds[0]); w['work'] = 100.; w['horizon'] = 800.
        e = ActionEnv(w, self.canonical, True, self.settings)
        self.assertEqual(e.mask()[18:].tolist(), [False, False, False, False, True, True])
        for action in [-1, 24, True, 1.5, 18]:
            with self.assertRaises(ValueError):
                e.step(action)

    def test_gym_timeout_is_absorbing_and_reward_aligned(self):
        w = deepcopy(self.worlds[0]); w['work'] = 100.; w['horizon'] = 800.
        w['failures'] = list(np.arange(50., 800., 50.)); w['recovery'] = 0.
        gym = gym_environment([w], self.canonical, self.settings, False, 9)
        gym.reset(); done = False; total = 0.
        while not done:
            _, r, done, truncated, info = gym.step(17); total += r
            self.assertFalse(truncated)
        self.assertFalse(info['results']['completed'])
        self.assertAlmostEqual(total, -90.)
        validate_run(info['results'])





if __name__ == '__main__':
    unittest.main()
