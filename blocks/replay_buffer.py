import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    经验回放缓冲区（Off-policy 算法: DQN, DDQN, SAC, TD3）。
    存储 (obs, action, reward, next_obs, done) 五元组。
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """添加一条经验"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device = None) -> dict:
        """
        随机采样一批数据，返回 Tensor 字典。

        Returns:
            dict with keys: 'obs', 'action', 'reward', 'next_obs', 'done'
        """
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        result = {
            'obs': torch.tensor(np.array(obs), dtype=torch.float32),
            'action': torch.tensor(np.array(action), dtype=torch.long),
            'reward': torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(-1),
            'next_obs': torch.tensor(np.array(next_obs), dtype=torch.float32),
            'done': torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(-1),
        }

        if device:
            result = {k: v.to(device) for k, v in result.items()}

        return result

    def __len__(self) -> int:
        return len(self.buffer)
    
    def __repr__(self):
        return f"ReplayBuffer(size={len(self.buffer)})"


class RolloutBuffer:
    """
    轨迹缓冲区（On-policy 算法: PPO, A2C）。
    存储一次 rollout 的完整数据，更新后清空。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def push(self, obs, action, reward, done, log_prob, value):
        """添加一步数据"""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float, gamma: float,
                                       gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE 优势和折扣回报。

        Args:
            last_value: 最后一步的 V(s) 估计（用于 bootstrap）
            gamma: 折扣因子
            gae_lambda: GAE lambda

        Returns:
            (returns, advantages) 两个 Tensor
        """
        rewards = self.rewards
        dones = self.dones
        values = self.values + [last_value]

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        return returns, advantages

    def get_batches(self, batch_size: int, device: torch.device = None):
        """
        将 buffer 数据切分为 mini-batch 的生成器。

        Yields:
            dict with keys: 'obs', 'actions', 'log_probs', 'returns', 'advantages'
        """
        size = len(self.obs)
        indices = np.random.permutation(size)

        obs = torch.tensor(np.array(self.obs), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long)
        log_probs = torch.stack(self.log_probs).detach()

        for start in range(0, size, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            batch = {
                'obs': obs[idx],
                'actions': actions[idx],
                'log_probs': log_probs[idx],
            }

            if device:
                batch = {k: v.to(device) for k, v in batch.items()}

            yield batch

    def __len__(self) -> int:
        return len(self.obs)
