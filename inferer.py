import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class BaseInferer(nn.Module):
    def __init__(self, env, ckpt_path, steps=1000):
        super().__init__()
        self.env = env
        self.steps = steps

        self.init_model()
        self.model.load_state_dict(torch.load(ckpt_path))

    def init_model(self):
        raise NotImplementedError

    def infer(self):
        self.before_infer_hook()

        steps = 0
        while steps < self.steps:
            episode_steps = self.infer_one_episode()
            steps += episode_steps

        self.env.close()

    def infer_one_episode(self):
        done = False
        state, _ = self.env.reset()

        steps = 0
        while not done:
            self.env.render()
            # get action & next state
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = self.model(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            next_state, _, done, _, _ = self.env.step(action.item())
            steps += 1

            if done:
                return steps
            else:
                state = next_state

    def before_infer_hook(self):
        self.model.eval()
        logger.info('start infering...')
