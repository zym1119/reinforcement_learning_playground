import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class CategoricalDist:
    """离散动作分布（用于 CartPole 等离散动作空间）"""

    def __init__(self, logits: torch.Tensor):
        self.dist = Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def mode(self) -> torch.Tensor:
        return self.dist.probs.argmax(dim=-1)


class GaussianDist:
    """连续动作分布（用于 Pendulum、MuJoCo 等连续动作空间）"""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        std = log_std.exp()
        self.dist = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        
        return self.dist.rsample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy().sum(dim=-1)

    def mode(self) -> torch.Tensor:
        return self.dist.mean
