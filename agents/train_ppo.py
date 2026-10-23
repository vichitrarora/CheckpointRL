"""Masked PPO training loop and on-policy optimizer diagnostics."""
import numpy as np
from checkpoint_rl.rollout import gym_environment
from agents.ppo import PPOActor


def train(arm, seed, worlds, validation, canonical, settings, checkpoint, start, curve):
    augmented = settings['arms'][arm]['augmented']
    budget = settings['decisions_per_fit']
    next_val = settings['validation_every']
    actions_n = len(canonical['interval_actions_seconds']) + 1 + (len(settings['fractions']) if augmented else 0)
    import torch
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    class Stats(BaseCallback):
        def __init__(self):
            super().__init__(); self.episodes = self.timeouts = 0
            self.actions_count = np.zeros(actions_n, np.int64); self.exposure = {}; self.returns = []

        def _on_step(self):
            for info in self.locals['infos']:
                self.actions_count[info['chosen_action']] += 1
                cell = str(tuple(info['cell'].values())); self.exposure[cell] = self.exposure.get(cell, 0) + 1
                if 'results' in info:
                    self.episodes += 1; self.timeouts += int(not info['results']['completed'])
                    self.returns.append(info['results']['episode_return'])
            return True
    s = dict(settings['ppo']); n_envs = s.pop('n_envs')
    env = DummyVecEnv([lambda i=i: gym_environment(worlds, canonical, settings, augmented, seed + 90000 + i * 1000)
                      for i in range(n_envs)])
    model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                        policy_kwargs={'net_arch': {'pi': [96, 96], 'vf': [96, 96]}}, **s)
    callback = Stats()
    while model.num_timesteps < budget:
        model.learn(s['n_steps'] * n_envs, callback=callback, reset_num_timesteps=False)
        steps = model.num_timesteps
        if steps >= next_val or steps >= budget:
            stats = {'episodes': callback.episodes, 'training_timeouts': callback.timeouts,
                     'optimizer_epochs': model._n_updates,
                     'optimizer_steps': int(next(iter(model.policy.optimizer.state.values()))['step']),
                     'action_counts': callback.actions_count.tolist(), 'cell_decision_exposure': dict(callback.exposure),
                     'recent_mean_return': float(np.mean(callback.returns[-100:])),
                     'ppo_diagnostics': {k: float(v) for k, v in model.logger.name_to_value.items()
                                         if k.startswith('train/') and np.ndim(v) == 0}}
            checkpoint(arm, seed, steps, PPOActor(model), model.save, validation, canonical, settings, stats, start, curve)
            next_val = (steps // settings['validation_every'] + 1) * settings['validation_every']
    env.close()
    return steps
