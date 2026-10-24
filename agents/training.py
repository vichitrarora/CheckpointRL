"""Train from scratch; checkpoint selection uses validation only."""
from checkpoint_rl.runtime import *
from checkpoint_rl.rollout import rollouts
from checkpoint_rl.metrics import aggregate
import time

def paths(arm, seed):
    return HERE / 'results' / arm / f'seed{seed}', HERE / 'models' / arm / f'seed{seed}'


def checkpoint(arm, seed, steps, actor, save, validation, canonical, settings, stats, start, curve):
    folder, models = paths(arm, seed)
    metrics = aggregate(rollouts(validation, canonical, actor=actor,
                                augmented=settings['arms'][arm]['augmented'], settings=settings))
    modelpath = models / f'step{steps:07d}.{ "npz" if settings["arms"][arm]["algorithm"] == "dqn" else "zip" }'
    save(modelpath)
    row = {'decisions': steps, 'validation': metrics, 'model': modelpath.relative_to(HERE).as_posix(),
           'sha256': sha(modelpath), 'runtime_including_validation_seconds': time.perf_counter() - start,
           **stats}
    curve.append(row); write(folder / 'learning_curve.json', curve)
    print(json.dumps({'arm': arm, 'seed': seed, 'step': steps,
                      'validation_excess_pct': metrics['per_job_excess_pct'],
                      'completion_pct': metrics['completion_pct'], 'seconds': row['runtime_including_validation_seconds']}), flush=True)


def fit(arm, seed, output, smoke=False, decisions=None):
    global HERE
    HERE = Path(output).resolve()
    settings = config(); canonical = canonical_config()
    if smoke:
        settings['decisions_per_fit'] = 8192 if arm == 'dqn' else 512
        settings['validation_every'] = settings['decisions_per_fit'] // 2
        settings['ppo'].update(n_envs=4, n_steps=32, batch_size=64, n_epochs=2)
    elif decisions is not None:
        if decisions < 8192:
            raise ValueError('Use --smoke for short checks, or at least 8192 decisions')
        settings['decisions_per_fit'] = decisions
    write(HERE / 'effective_config.json', settings)
    if arm not in settings['arms'] or seed not in settings['seeds']:
        raise ValueError('Unregistered fit')
    folder, models = paths(arm, seed)
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError('Partial fit retained; do not silently restart: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    worlds, validation = corpus('train'), corpus('validation')
    if smoke:
        validation = validation[:81]
    spec = settings['arms'][arm]; augmented = spec['augmented']
    curve = []; start = time.perf_counter()
    if spec['algorithm'] == 'dqn':
        from agents.train_dqn import train
    else:
        from agents.train_ppo import train
    steps = train(arm, seed, worlds, validation, canonical, settings, checkpoint, start, curve)
    best = min(curve, key=lambda r: (r['validation']['mean_normalized_time'], r['decisions']))
    result = {'arm': arm, 'seed': seed, 'algorithm': spec['algorithm'], 'augmented': augmented,
              'trained_decisions': steps, 'selected_decisions': best['decisions'], 'model': best['model'],
              'sha256': best['sha256'], 'validation': best['validation'], 'retained_checkpoints': len(curve),
              'runtime_including_validation_seconds': time.perf_counter() - start,
              'data_manifest_sha256': sha(ROOT / 'data/manifest.json'),
              'smoke': smoke, 'excluded_action_indices': [16] if arm == 'dqn' else []}
    write(folder / 'complete.json', result)
    return result
