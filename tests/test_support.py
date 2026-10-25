"""Verify the fixed library restriction independently of measured outcomes."""
import unittest
from checkpoint_rl.runtime import *
from checkpoint_rl.environment import Environment
from checkpoint_rl.rollout import rollouts
from agents.dqn import Network
from agents.dqn import SupportedActor


class SupportedActions(unittest.TestCase):
    def test_action_never_legal_in_training(self):
        worlds = corpus('train'); canonical = canonical_config()
        self.assertLess(max(w['work'] for w in worlds), canonical['interval_actions_seconds'][16])
        for world in worlds:
            self.assertFalse(Environment(world, canonical).mask()[16])

    def test_masks_only_declared_option_without_mutating_inputs(self):
        class Actor:
            def actions(self, x, masks):
                q = np.zeros_like(masks, dtype=float); q[:, 16] = 2.; q[:, 10] = 1.
                return np.argmax(np.where(masks, q, -np.inf), axis=1)
        x = np.zeros((3, 16), np.float32); masks = np.ones((3, 18), bool)
        original = masks.copy(); actor = Actor()
        np.testing.assert_array_equal(actor.actions(x, masks), [16, 16, 16])
        np.testing.assert_array_equal(SupportedActor(actor, [16]).actions(x, masks), [10, 10, 10])
        np.testing.assert_array_equal(masks, original)

    def test_canonical_validation_decisions_unchanged(self):
        worlds = corpus('validation')[:81]; canonical = canonical_config()
        for augmented in (False, True):
            actor = Network(701, 16, 24 if augmented else 18)
            a = rollouts(worlds, canonical, actor=actor, augmented=augmented)
            b = rollouts(worlds, canonical, actor=SupportedActor(actor, [16]), augmented=augmented)
            self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
