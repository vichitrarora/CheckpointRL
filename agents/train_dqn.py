"""Double DQN training loop with prioritized replay and target-network updates."""
import numpy as np
from checkpoint_rl.environment import FEATURES
from checkpoint_rl.rollout import ActionEnv
from agents.dqn import Network, Replay, Optimizer


def train(arm, seed, worlds, validation, canonical, settings, checkpoint, start, curve):
    augmented = settings['arms'][arm]['augmented']
    budget = settings['decisions_per_fit']
    next_val = settings['validation_every']
    actions_n = len(canonical['interval_actions_seconds']) + 1 + (len(settings['fractions']) if augmented else 0)
    s = settings['dqn']; rng = np.random.default_rng(seed); sampler = np.random.default_rng(seed + 90000)
    online = Network(seed, len(FEATURES), actions_n); target = Network(seed + 1, len(FEATURES), actions_n)
    target.copy_from(online)
    replay = Replay(s['replay_capacity'], len(FEATURES), actions_n, s['alpha'])
    opt = Optimizer(online, s['learning_rate'], s['gradient_clip'])
    pick = lambda: ActionEnv(worlds[int(sampler.integers(len(worlds)))], canonical, augmented, settings)
    envs = [pick() for _ in range(s['parallel_envs'])]
    steps = episodes = timeouts = 0; actions_count = np.zeros(actions_n, np.int64)
    exposure = {}; losses = []; norms = []; returns = []
    while steps < budget:
        x = np.array([e.observation() for e in envs]); masks = np.array([e.mask() for e in envs])
        epsilon = 1 + (s['epsilon_end'] - 1) * min(1., steps / (s['epsilon_decay_fraction'] * budget))
        actions = online.actions(x, masks)
        random_actions = np.argmax(np.where(masks, rng.random(masks.shape), -1.), axis=1)
        actions = np.where(rng.random(len(envs)) < epsilon, random_actions, actions)
        nx = []; rewards = []; dones = []; nmasks = []
        for i, (env, action) in enumerate(zip(envs, actions)):
            obs, reward, done, mask = env.step(int(action))
            nx.append(obs); rewards.append(reward * settings['reward_scale']); dones.append(done); nmasks.append(mask)
            actions_count[action] += 1
            cell = str(tuple(env.core.w['cell'].values())); exposure[cell] = exposure.get(cell, 0) + 1
            if done:
                episodes += 1; timeouts += int(not env.completed); returns.append(env.core.return_sum); envs[i] = pick()
        replay.add(x, actions, np.array(rewards, np.float32), np.array(nx), np.array(dones), np.array(nmasks))
        steps += len(envs)
        if steps >= s['warmup_transitions']:
            for _ in range(s['updates_per_vector_step']):
                batch, ids = replay.sample(rng, s['batch_size'], beta=.4 + .6 * min(1., steps / budget))
                loss, norm, error = opt.update(online, batch, target); replay.update_priorities(ids, error)
                losses.append(loss); norms.append(norm)
                if opt.t % s['target_update_optimizer_steps'] == 0:
                    target.copy_from(online)
        if steps >= next_val or steps >= budget:
            stats = {'episodes': episodes, 'training_timeouts': timeouts, 'epsilon': epsilon,
                     'optimizer_steps': opt.t, 'mean_td_loss': float(np.mean(losses)),
                     'mean_gradient_norm': float(np.mean(norms)), 'action_counts': actions_count.tolist(),
                     'cell_decision_exposure': dict(exposure),
                     'recent_mean_return': float(np.mean(returns[-100:]))}
            checkpoint(arm, seed, steps, online, lambda p: online.save(p, {'arm': arm, 'seed': seed, 'steps': steps}),
                       validation, canonical, settings, stats, start, curve)
            next_val += settings['validation_every']; losses.clear(); norms.clear()
    return steps
