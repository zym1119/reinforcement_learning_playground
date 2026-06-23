import torch
import torch.optim as optim

from agents.base import BaseTrainer, BaseInferer
from networks.mlp import build_mlp
from networks.distributions import CategoricalDist
from utils import TRAINER, INFERER


@TRAINER.register('PolicyGradient')
class PGTrainer(BaseTrainer):
    """REINFORCE (vanilla Policy Gradient) 训练器"""

    def init_model(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        self.policy = build_mlp(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['lr'])
        self.gamma = self.config.get('gamma', 0.99)

        # 用于暂存单个 episode 的数据
        self._log_probs = []
        self._rewards = []

    def collect(self) -> dict:
        """跑完整个 episode，收集 log_probs 和 rewards"""
        self.policy.train()
        obs, _ = self.env.reset()
        done = False
        self._log_probs = []
        self._rewards = []
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            logits = self.policy(obs_t)
            dist = CategoricalDist(logits)
            action = dist.sample()
            self._log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            self._rewards.append(reward)
            done = terminated or truncated
            steps += 1

        return {'n_steps': steps, 'episode_reward': sum(self._rewards)}

    def update(self) -> dict:
        """REINFORCE 更新：用折扣回报加权 log_prob"""
        # 计算折扣回报
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 标准化
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算 loss
        log_probs = torch.stack(self._log_probs)
        loss = -(log_probs * returns).mean()

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, obs, deterministic=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
        dist = CategoricalDist(logits)
        if deterministic:
            return dist.mode().item()
        return dist.sample().item()

    def get_state_dict(self):
        return self.policy.state_dict()


@INFERER.register('PolicyGradient')
class PGInferer(BaseInferer):
    """REINFORCE 推理器"""

    def init_model(self, ckpt_path):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        self.policy = build_mlp(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.policy.eval()

    def predict(self, obs, deterministic=True):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
        dist = CategoricalDist(logits)
        if deterministic:
            return dist.mode().item()
        return dist.sample().item()
