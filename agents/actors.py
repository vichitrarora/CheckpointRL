"""Inference adapters; all learned decisions come from the saved RL networks."""
from checkpoint_rl.runtime import ROOT, read, sha
from .dqn import Network, SupportedActor
from .ppo import PPOActor


def load_actor(selection, base=ROOT):
    path = base / selection['model']
    if sha(path) != selection['sha256']:
        raise ValueError('Model checksum mismatch: ' + str(path))
    if selection['algorithm'] == 'dqn':
        actor = Network.load(path)
    else:
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(1)
        actor = PPOActor(MaskablePPO.load(path, device='cpu'))
    excluded = selection.get('excluded_action_indices', [])
    return SupportedActor(actor, excluded) if excluded else actor
