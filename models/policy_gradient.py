from inferer import BaseInferer
from trainer import BaseTrainer
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        x = torch.relu(self.dropout(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PGTrainer(BaseTrainer):
    def __init__(self, running_reward=10, **kwargs):
        super().__init__(**kwargs)
        self.running_reward = running_reward

    def init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = PolicyNetwork(state_dim, 128, action_dim)

    def train_one_episode(self):
        state, _ = self.env.reset()
        done = False
        rewards = []
        log_probs = []

        # forward 1 episode
        while not done:
            # sample
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = self.model(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            # take action
            next_state, reward, done, _, _ = self.env.step(action.item())
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            state = next_state

        # calculate accumulated reward
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + 0.99 * cumulative_reward
            returns.insert(0, cumulative_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # calculate loss
        log_probs_tensor = torch.stack(log_probs)  # len(episode)
        loss = torch.sum(-log_probs_tensor * returns)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'reward': sum(rewards), 'running_reward': self.running_reward}

    def after_train_one_episode_hook(self, module_outputs):
        super().after_train_one_episode_hook(module_outputs)
        self.running_reward = 0.99 * self.running_reward + \
            module_outputs['reward'] * 0.01
        if self.running_reward > self.env.spec.reward_threshold:
            logger.info(
                f'Solved!, running reward is {self.running_reward} at step {self.episode}')
            return True
        return False


class PGInferer(BaseInferer):
    def init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = PolicyNetwork(state_dim, 128, action_dim)


def get_pg_trainer(env, run_dir, **kwargs):
    return PGTrainer(env=env, run_dir=run_dir, **kwargs)


def get_pg_inferer(env, ckpt_path, **kwargs):
    return PGInferer(env=env, ckpt_path=ckpt_path, **kwargs)
