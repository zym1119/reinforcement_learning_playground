from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区，存储和采样经验数据"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """添加一条经验数据"""
        self.buffer.append(args)

    def sample(self, batch_size):
        """随机采样一批经验数据"""
        batch = random.sample(self.buffer, batch_size)
        args = tuple(zip(*batch))
        return args

    def __len__(self):
        """返回缓冲区当前数据数量"""
        return len(self.buffer)
    
    
class BatchReplayBuffer(ReplayBuffer):
    """
    批量经验回放缓冲区，用于存储和采样多个批次的经验数据。
    相比于 ReplayBuffer，输入为带有 batch 维度的数据。
    可以用于训练模型时的批量处理。  
    """

    def push(self, *args):
        """添加一个批次的经验数据"""
        for single_args in zip(*args):
            self.buffer.append(single_args)