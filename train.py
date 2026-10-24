"""Train one retained RL strategy from scratch, without touching shipped models."""
from checkpoint_rl.runtime import ROOT, config, new_output, verify_assets, write
import argparse
from agents.training import fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent', required=True, choices=['ppo', 'dqn'])
    p.add_argument('--seeds', nargs='+', type=int, help='Default: all three seeds; smoke: seed 4401')
    p.add_argument('--smoke', action='store_true', help='Short optimizer and validation check; not a performance run')
    p.add_argument('--decisions', type=int, help='Optional new training budget; default 2,000,000')
    p.add_argument('--output', type=str, required=True, help='New directory, e.g. runs/ppo-training')
    a = p.parse_args()
    seeds = a.seeds or ([4401] if a.smoke else config()['seeds'])
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config()['seeds']):
        p.error('Use unique registered seeds: 4401, 4402, 4403')
    if a.smoke and a.decisions is not None:
        p.error('--smoke and --decisions are mutually exclusive')
    verify_assets(include_models=False)
    output = new_output(a.output)
    selections = [fit(a.agent, seed, output, a.smoke, a.decisions) for seed in seeds]
    write(output / 'selection.json', {'selections': selections, 'smoke': a.smoke})
    print('Saved selected agents to', output / 'selection.json', flush=True)


if __name__ == '__main__':
    main()
