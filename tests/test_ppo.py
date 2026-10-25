"""PPO learning, action and checkpoint contract tests."""
import unittest
from checkpoint_rl.runtime import *
from checkpoint_rl.environment import FEATURES
from checkpoint_rl.rollout import ActionEnv, rollouts, gym_environment
from agents.ppo import PPOActor


class PPOContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = corpus('validation')
        cls.canonical = canonical_config(); cls.settings = config()


    def test_ppo_mask_and_save_load_preserve_actions(self):
        import torch
        torch.set_num_threads(1)
        from tempfile import TemporaryDirectory
        from sb3_contrib import MaskablePPO
        for augmented in (False, True):
            gym = gym_environment(self.worlds[:8], self.canonical, self.settings, augmented, 99)
            model = MaskablePPO('MlpPolicy', gym, seed=991, n_steps=16, batch_size=16,
                                n_epochs=1, gamma=1., device='cpu', verbose=0)
            model.learn(32)
            envs = [ActionEnv(w, self.canonical, augmented, self.settings) for w in self.worlds[:8]]
            x = np.array([e.observation() for e in envs]); masks = np.array([e.mask() for e in envs])
            a = PPOActor(model).actions(x, masks)
            self.assertTrue(masks[np.arange(8), a].all())
            with TemporaryDirectory() as temp:
                p = Path(temp) / 'test.zip'; model.save(p)
                other = PPOActor(MaskablePPO.load(p, device='cpu'))
                np.testing.assert_array_equal(a, other.actions(x, masks))



if __name__ == "__main__":
    unittest.main()
