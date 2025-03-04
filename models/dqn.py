from collections import deque
import logging
import random

import torch
import torch.nn as nn
import numpy as np

from trainer import BaseTrainer


logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(p=0.6)

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


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNTrainer(BaseTrainer):
    def __init__(self, *args, buffer_size=1000, batch_size=64, num_episodes=500, target_update_interval=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.target_update_interval = target_update_interval
        self.mse_loss = nn.MSELoss()

    def init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy_network = DQN(state_dim, 128, action_dim)
        self.target_network = DQN(state_dim, 128, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_one_episode(self):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        gamma = 0.99  # a constant
        loss = torch.tensor(0.0)

        while not done:
            # take action
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            # train batch
            if len(self.replay_buffer) >= self.batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(
                    self.batch_size)
                # to tensor
                batch_state = torch.tensor(
                    batch_state, dtype=torch.float32)  # [B, state_dim]
                batch_action = torch.tensor(
                    batch_action, dtype=torch.int64).unsqueeze(1)  # [B, action_dim]
                batch_reward = torch.tensor(
                    batch_reward, dtype=torch.float32).unsqueeze(1)  # [B, 1]
                batch_next_state = torch.tensor(
                    batch_next_state, dtype=torch.float32)  # [B, state_dim]
                batch_done = torch.tensor(
                    batch_done, dtype=torch.float32).unsqueeze(1)  # [B, 1]

                q_values = self.policy_network(
                    batch_state).gather(1, batch_action)
                next_q_values = self.target_network(
                    batch_next_state).max(dim=1).values.unsqueeze(1)

                expected_q_values = batch_reward + \
                    gamma * next_q_values * (1 - batch_done)

                loss = self.mse_loss(q_values, expected_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {'loss': loss.item(), 'reward': total_reward}

    def before_train_hook(self):
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.lr)
        self.policy_network.train()
        self.target_network.eval()
        logger.info('start training...')
        self.update_target_network()

    def before_train_one_episode_hook(self):
        super().before_train_one_episode_hook()

        # update epsilon setting
        epsilon_start = 1.0
        epsilon_end = 0.995
        epsilon_decay = 0.01

        if self.episode == 0:
            self.epsilon = epsilon_start
        else:
            if self.epsilon > epsilon_end:
                self.epsilon *= epsilon_decay

    def after_train_one_episode_hook(self, module_outputs):
        super().after_train_one_episode_hook(module_outputs)

        if self.episode % self.target_update_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        # fill target network with policy network weights
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state):
        """Select action based on state.
        At the beginning of the training, there is a higher probability of randomly
        sampling actions, allowing the model to see more diverse data.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1).item()
        else:
            return random.randint(0, self.action_dim - 1)

    def save_model(self, is_last=False):
        if is_last:
            save_path = self.run_dir / 'model_last.pth'
        else:
            save_path = self.run_dir / f'model_episode{self.episode}.pth'
        torch.save(self.policy_network.state_dict(), save_path)
        logger.info(f'save ckpt {save_path}')


def get_dqn_trainer(env, run_dir, **kwargs):
    return DQNTrainer(env, run_dir, **kwargs)
