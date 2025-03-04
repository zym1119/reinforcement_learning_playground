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
    def __init__(self, *args, buffer_size=1000, batch_size=64, num_episodes=500, target_update=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.target_update = target_update
        self.mse_loss = nn.MSELoss()
        
    def init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy_network = DQN(state_dim, 128, action_dim)
        self.target_network = DQN(state_dim, 128, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def train(self):
        self.before_train_hook()
        
        # train loop
        epsilon_start = 1.0
        epsilon_end = 0.05
        epsilon_decay = 0.001
        
        for episode in range(self.num_episodes):
            self.episode = episode
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-epsilon_decay * episode)
            model_outputs = self.train_one_episode(epsilon)
            
            if (episode + 1) % self.target_update == 0:
                self.update_target_network()
            
            self.log_model_outputs(episode, model_outputs)
            
            # save model if necessary
            if episode % self.save_interval == 0:
                self.save_model()
        
        self.save_model(is_last=True)
    
    def train_one_episode(self, epsilon):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        gamma = 0.99  # a constant
        
        while not done:
            action = self.select_action(state, epsilon)
            next_state, reward, done, _, _= self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(self.replay_buffer) >= self.batch_size:
                state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
                # to tensor
                state = torch.tensor(state, dtype=torch.float32)
                action = torch.tensor(action, dtype=torch.int64).unsqueeze(1)
                reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)
                
                q_values = self.policy_network(state).gather(1, action)
                next_q_values = self.target_network(next_state).max(1)[0].unsqueeze(1)

                expected_q_values = reward + gamma * next_q_values * (1 - done)
                
                loss = self.mse_loss(q_values, expected_q_values)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                return {'loss': loss.item(), 'reward': total_reward}
    
    def before_train_hook(self):
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.policy_network.train()
        self.target_network.eval()
        logger.info('start training...')
        self.update_target_network()
    
    def update_target_network(self):
        # fill target network with policy network weights
        self.target_network.load_state_dict(self.policy_network.state_dict())
    
    def select_action(self, state, epsilon):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() > epsilon:
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

def get_dqn_trainer(env, run_dir, **kwargs):
    return DQNTrainer(env, run_dir, **kwargs)