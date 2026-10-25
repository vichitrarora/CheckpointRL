"""DQN learning, action and checkpoint contract tests."""
import unittest
from checkpoint_rl.runtime import *
from checkpoint_rl.environment import FEATURES
from checkpoint_rl.rollout import ActionEnv, rollouts, gym_environment
from agents import dqn as learner
from agents.dqn import Network, Replay, Optimizer


class DQNContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = corpus('validation')
        cls.canonical = canonical_config(); cls.settings = config()


    def test_parallel_and_serial_network_rollouts(self):
        for augmented in (False, True):
            net = Network(9, len(FEATURES), 24 if augmented else 18)
            expected = []
            for w in self.worlds[:32]:
                e = ActionEnv(w, self.canonical, augmented, self.settings)
                while not e.done:
                    a = net.actions(e.observation()[None], e.mask()[None])[0]
                    e.step(int(a))
                expected.append(e.metrics())
            actual = rollouts(self.worlds[:32], self.canonical, actor=net, augmented=augmented, settings=self.settings)
            self.assertEqual(sorted(expected, key=lambda r: r['world_id']), actual)


    def test_replay_and_targets_follow_registered_double_dqn(self):
        online = Network(9, 16, 18); target = Network(99, 16, 18)
        replay = Replay(100, 16, 18, .6)
        x = np.zeros((2, 16), np.float32); nx = np.ones_like(x)
        r = np.array([2., 3.], np.float32); done = np.array([True, False])
        mask = np.ones((2, 18), bool); mask[:, :2] = False
        selected = online.actions(nx, mask)
        y = learner.double_targets(online, target, r, nx, done, mask)
        self.assertEqual(y[0], 2.)
        self.assertAlmostEqual(y[1], 3. + target.forward(nx)[1, selected[1]])
        replay.add(x, np.array([2, 3]), r, nx, done, mask)
        batch, ids = replay.sample(np.random.default_rng(1), 8)
        opt = Optimizer(online, .0002, 10.)
        loss, norm, error = opt.update(online, batch, target)
        self.assertTrue(np.isfinite(loss + norm)); replay.update_priorities(ids, error)
        self.assertAlmostEqual(replay.tree[1], replay.tree[replay.leaves:].sum())



if __name__ == "__main__":
    unittest.main()
