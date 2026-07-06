import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    经验回放缓冲区（Off-policy 算法: DQN, DDQN, SAC, TD3）。
    存储 (obs, action, reward, next_obs, done) 五元组，obs/next_obs 直接存 tensor。
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        """添加一条经验，obs/next_obs 为 tensor，action 为 int/tensor，reward/done 为标量"""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device = None) -> dict:
        """
        随机采样一批数据，返回 Tensor 字典。

        Returns:
            dict with keys: 'obs', 'action', 'reward', 'next_obs', 'done'
        """
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)

        batch_obs = torch.stack(obs)
        next_batch_obs = torch.stack(next_obs)

        # action 可能是标量（离散）或 tensor（连续）
        if isinstance(action[0], torch.Tensor):
            batch_action = torch.stack(action)
        else:
            batch_action = torch.tensor(action, dtype=torch.long)

        result = {
            'obs': batch_obs,
            'action': batch_action,
            'reward': torch.tensor(reward, dtype=torch.float32).unsqueeze(-1),
            'next_obs': next_batch_obs,
            'done': torch.tensor(done, dtype=torch.float32).unsqueeze(-1),
        }

        if device:
            result = {k: v.to(device) for k, v in result.items()}

        return result

    def __len__(self) -> int:
        return len(self.buffer)
    
    def __repr__(self):
        return f"ReplayBuffer(size={len(self.buffer)})"

