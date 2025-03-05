from itertools import count
import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class BaseTrainer(nn.Module):
    def __init__(self, env, run_dir, lr=0.01, num_episodes=-1, log_interval=10, save_interval=100, **kwargs):
        super().__init__()
        self.env = env
        self.run_dir = run_dir
        self.lr = lr
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.init_model()

        self.episode = 0
        self.running_reward = 0

    def init_model(self):
        raise NotImplementedError

    def train(self):
        self.before_train_hook()

        infinite_training = 
        if infinite_training:
            iterator = count(1)
        else:
            iterator = range(self.num_episodes)

        for _ in iterator:
            self.before_train_one_episode_hook()
            module_outputs = self.train_one_episode()
            shoule_break = self.after_train_one_episode_hook(module_outputs)
            if shoule_break:
                self.train_break_hook()
                break

        self.after_train_hook()

    def train_one_episode(self):
        raise NotADirectoryError

    def save_model(self, is_last=False):
        if is_last:
            save_path = self.run_dir / 'model_last.pth'
        else:
            save_path = self.run_dir / f'model_episode{self.episode}.pth'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'save ckpt {save_path}')

    def before_train_hook(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        logger.info('start training...')

    def before_train_one_episode_hook(self):
        """Hook for before train one episode."""
        pass

    def after_train_one_episode_hook(self, module_outputs):
        """Hook of after train one episode, usually log and save."""
        self.episode += 1

        self.log_model_outputs(self.episode, module_outputs)

        # save model if necessary
        if self.episode % self.save_interval == 0:
            self.save_model()
        
        # infinite training loop stop condition
        if self.num_episodes <= 0:
            self.running_reward = 0.99 * self.running_reward + \
                module_outputs['reward'] * 0.01
            if self.running_reward > self.env.spec.reward_threshold:
                logger.info(
                    f'Solved!, running reward is {self.running_reward} at step {self.episode}')
                return True
        return False

    def train_break_hook(self):
        logger.info(f'train break at episode {self.episode}')

    def after_train_hook(self):
        self.save_model(is_last=True)

    def log_model_outputs(self, episode, model_outputs):
        if model_outputs is None:
            return
        if episode % self.log_interval == 0:
            info_list = []
            for k, v in model_outputs.items():
                info_list.append(f'{k}: {v:.4f}')
            log_str = f'Episode {episode}: {", ".join(info_list)}'
            logger.info(log_str)

    @property
    def infinite_training(self):
        return self.num_episodes <= 0
    
class BaseInferer(nn.Module):
    def __init__(self, env, ckpt_path, steps=1000):
        super().__init__()
        self.env = env
        self.steps = steps

        self.init_model(ckpt_path)

    def init_model(self, ckpt_path):
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
            action = self.select_action(logits)
            next_state, _, done, _, _ = self.env.step(action.item())
            steps += 1

            if done:
                return steps
            else:
                state = next_state

    def select_action(self, logits):
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action

    def before_infer_hook(self):
        self.model.eval()
        logger.info('start infering...')
