"""Deterministic inference adapter for the trained masked PPO policy."""


class PPOActor:
    def __init__(self, model):
        self.model = model

    def actions(self, x, masks):
        return self.model.predict(x, deterministic=True, action_masks=masks)[0]
