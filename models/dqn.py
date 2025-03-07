from collections import deque
import logging
import random

import torch
import torch.nn as nn

from runner import BaseTrainer, BaseInferer

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """
    定义深度Q网络（DQN）模型，接受状态输入，输出各动作Q值
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.init_weights()

    def init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        """前向传播，计算各动作Q值"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """经验回放缓冲区，存储和采样经验数据"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加一条经验数据"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验数据"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        """返回缓冲区当前数据数量"""
        return len(self.buffer)

class DQNTrainer(BaseTrainer):
    """DQN训练器，包含训练逻辑"""
    def __init__(self, *args, buffer_size=1000, batch_size=64, num_episodes=500, target_update_interval=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.target_update_interval = target_update_interval
        self.mse_loss = nn.MSELoss()

    def init_model(self):
        """初始化策略网络和目标网络"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy_network = DQN(state_dim, 128, action_dim)
        self.target_network = DQN(state_dim, 128, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_one_episode(self):
        """训练一个回合"""
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        loss = torch.tensor(0.0)

        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.train_one_batch()

        return_dict = {'loss': loss.item(), 'reward': total_reward}
        if self.infinite_training:
            return_dict['running_reward'] = self.running_reward
        return return_dict

    def train_one_batch(self):
        """训练一个批量数据"""
        gamma = 0.99

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(self.batch_size)
        batch_state = torch.tensor(batch_state, dtype=torch.float32)
        batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(1)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
        batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_network(batch_state).gather(1, batch_action)
        next_q_values = self.target_network(batch_next_state).max(dim=1).values.unsqueeze(1)

        expected_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)
        loss = self.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def before_train_hook(self):
        """训练前初始化优化器等操作"""
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.policy_network.train()
        self.target_network.eval()
        logger.info('start training...')
        self.update_target_network()

    def before_train_one_episode_hook(self):
        """训练一个回合前更新epsilon值"""
        super().before_train_one_episode_hook()
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        if self.episode == 0:
            self.epsilon = epsilon_start
        else:
            if self.epsilon > epsilon_end:
                self.epsilon *= epsilon_decay

    def after_train_one_episode_hook(self, module_outputs):
        """训练一个回合后根据间隔更新目标网络"""
        super().after_train_one_episode_hook(module_outputs)
        if self.episode % self.target_update_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        """更新目标网络参数与策略网络一致"""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state):
        """根据epsilon贪婪策略选择动作"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_network(state).max(1)[1].view(1).item()
        else:
            return random.randint(0, self.action_dim - 1)

    def save_model(self, is_last=False):
        """保存模型状态字典"""
        if is_last:
            save_path = self.run_dir / 'model_last.pth'
        else:
            save_path = self.run_dir / f'model_episode{self.episode}.pth'
        torch.save(self.policy_network.state_dict(), save_path)
        logger.info(f'save ckpt {save_path}')

class DQNInferer(BaseInferer):
    """DQN推理器，包含推理逻辑"""
    def init_model(self, ckpt_path):
        """初始化模型并加载预训练参数"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.model = DQN(state_dim, 128, action_dim)
        self.model.load_state_dict(torch.load(ckpt_path))

    def select_action(self, logits):
        """根据模型输出选择动作"""
        action = logits.argmax(dim=-1)
        return action

def get_dqn_trainer(env, run_dir, **kwargs):
    """获取DQN训练器实例"""
    return DQNTrainer(env, run_dir, **kwargs)

def get_dqn_inferer(env, ckpt_path, **kwargs):
    """获取DQN推理器实例"""
    return DQNInferer(env, ckpt_path, **kwargs)