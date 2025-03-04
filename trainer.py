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

    def init_model(self):
        raise NotImplementedError

    def train(self):
        self.before_train_hook()

        infinite_training = self.num_episodes <= 0
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
