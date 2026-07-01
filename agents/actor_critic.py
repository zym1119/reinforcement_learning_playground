import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from blocks.mlp import _get_activation, _init_weights
from blocks.distributions import CategoricalDist
from utils import AGENT


class ActorCriticNetwork(nn.Module):
    """共享 backbone + actor/critic 双头网络"""
    def __init__(self, obs_dim, act_dim, hidden_dims, activation='relu'):
        super().__init__()
        activation_fn = _get_activation(activation)

        # 共享 backbone
        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn())
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # actor head
        self.actor_head = nn.Linear(prev_dim, act_dim)
        # critic head
        self.critic_head = nn.Linear(prev_dim, 1)

        _init_weights(self)

    def forward(self, obs):
        features = self.backbone(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_logits(self, obs):
        features = self.backbone(obs)
        return self.actor_head(features)

    def get_value(self, obs):
        features = self.backbone(obs)
        return self.critic_head(features)


@AGENT.register('ActorCritic')
class ActorCriticAgent(BaseAgent):
    def init_model(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        # 共享 backbone 的 actor-critic 网络
        self.model = ActorCriticNetwork(obs_dim, act_dim, hidden_dims, activation).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        
        self.gamma = self.config.get('gamma', 0.99)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.vf_coef = self.config.get('vf_coef', 0.5)

        self.rollout_length = self.config.get('rollout_length', 128)
        self.n_step = self.config.get('n_step', 5)
        self.normalize_advantage = self.config.get('normalize_advantage', False)

        # 用于暂存rollout数据（只存原始数据，不存梯度相关量）
        self._obs_batch = []
        self._actions = []
        self._rewards = []
        self._dones = []
        self._obs, _ = self.env.reset()
        self._episode_reward = 0.0

    def collect(self) -> dict:
        """采集数据，no_grad推理，只保存 obs/action/reward/done"""
        n_steps = 0
        n_episodes = 0
        return_dict = {}

        while n_steps < self.rollout_length:
            obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits = self.model.get_logits(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self._obs_batch.append(self._obs.copy() if hasattr(self._obs, 'copy') else self._obs)
            self._actions.append(action.item())
            self._rewards.append(reward)
            self._dones.append(done)

            self._obs = next_obs
            n_steps += 1
            self._episode_reward += reward

            if done:
                self._obs, _ = self.env.reset()
                n_episodes += 1
                return_dict['episode_reward'] = self._episode_reward
                self._episode_reward = 0.0
        
        return_dict.update({
            'n_steps': self.rollout_length,
            'n_episodes': n_episodes,
        })
        return return_dict
    
    def update(self) -> dict:
        """A2C 更新，使用 n-step return"""
        # 重新前向计算 log_prob, entropy, value（带梯度）
        obs = torch.tensor(self._obs_batch, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self._actions, dtype=torch.long, device=self.device)

        logits, values = self.model(obs)
        values = values.squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 计算 last_value 用于末尾 bootstrap
        with torch.no_grad():
            last_obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
            last_value = self.model.get_value(last_obs_t).squeeze().item()

        # 计算 n-step return（无梯度，作为 target）
        returns = []
        for t in range(self.rollout_length):
            G = 0.0
            gamma_k = 1.0
            done_encountered = False
            
            for k in range(self.n_step):
                if t + k >= self.rollout_length:
                    break
                G += gamma_k * self._rewards[t + k]
                if self._dones[t + k]:
                    done_encountered = True
                    break
                gamma_k *= self.gamma

            if not done_encountered:
                if t + self.n_step < self.rollout_length:
                    G += gamma_k * values[t + self.n_step].item()
                else:
                    G += gamma_k * last_value
            returns.append(G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 计算 advantage
        advantages = returns - values.detach()
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # actor loss
        actor_loss = -(log_probs * advantages).mean()
        
        # critic loss
        critic_loss = F.mse_loss(values, returns)

        # 总 loss
        loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * entropy

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 清空暂存数据
        self._obs_batch = []
        self._actions = []
        self._rewards = []
        self._dones = []

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': loss.item(),
        }

    def predict(self, obs, deterministic=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model.get_logits(obs_t)
        dist = CategoricalDist(logits=logits)
        if deterministic:
            return dist.mode().item()
        return dist.sample().item()
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_state_dict(self) -> dict:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
    
    def load_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])