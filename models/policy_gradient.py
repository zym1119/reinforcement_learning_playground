from runner import BaseTrainer, BaseInferer
from utils import TRAINER, INFERER
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


@TRAINER.register('PolicyGradient')
class PGTrainer(BaseTrainer):
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


@INFERER.register('PolicyGradient')
class PGInferer(BaseInferer):
    def init_model(self, ckpt_path):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = PolicyNetwork(state_dim, 128, action_dim)
        self.model.load_state_dict(torch.load(ckpt_path))
