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

