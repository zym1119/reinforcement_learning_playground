import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from blocks.distributions import CategoricalDist
from blocks.rollout_buffer import RolloutBuffer
from utils import AGENT


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(obs_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(512, act_dim)
        self.critic_head = nn.Linear(512, 1)
    
    def forward(self, obs):
        x = self.feature_extractor(obs)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value


@AGENT.register('PPOAtari')
class PPOAtariAgent(BaseAgent):
    def init_model(self):
        # 向量化环境用 single_*_space，单环境用 *_space
        obs_space = getattr(self.env, 'single_observation_space', self.env.observation_space)
        act_space = getattr(self.env, 'single_action_space', self.env.action_space)
        obs_dim = obs_space.shape[0]
        act_dim = act_space.n

        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['lr'], eps=1e-5)

        self.batch_size = self.config.get('batch_size', 32)
        self.n_steps = self.config.get('n_steps', 128)
        self.num_envs = self.env.num_envs

        # 初始化 rollout buffer
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            num_envs=self.num_envs,
            obs_shape=obs_space.shape,
            device=self.device,
        )

        # 持久状态
        self._obs, _ = self.env.reset()
        # 持续追踪每个 env 的累计 reward
        self._ep_rewards = torch.zeros(self.num_envs, device=self.device)
    
    def collect(self) -> dict:
        """
        并行环境下采集 n_steps 的数据，存入 rollout buffer。
        """
        self.policy.eval()
        self.buffer.reset()

        n_episodes = 0
        episode_rewards = []

        with torch.no_grad():
            for step in range(self.n_steps):
                logits, value = self.policy(self._obs)
                dist = CategoricalDist(logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, infos = self.env.step(action)
                done = terminated | truncated

                self.buffer.push(
                    self._obs, action, log_prob,
                    reward, done.float(), value.squeeze(-1),
                )

                # 累计 reward 并统计完成的 episode
                self._ep_rewards += reward
                if done.any():
                    done_mask = done.bool()
                    n_episodes += done_mask.sum().item()
                    episode_rewards.extend(self._ep_rewards[done_mask].tolist())
                    self._ep_rewards[done_mask] = 0.0

                self._obs = next_obs

            # 计算最后一步的 value（用于 GAE 计算）
            _, last_value = self.policy(self._obs)
            self._last_value = last_value.squeeze(-1)

        info = {
            'n_steps': self.n_steps * self.num_envs,
            'n_episodes': n_episodes,
        }
        if episode_rewards:
            info['episode_reward'] = sum(episode_rewards) / len(episode_rewards)

        return info
    
    def update(self):
        # 1. 计算 GAE（在 buffer 的时序数据上）
        self.buffer.compute_gae(
            last_value=self._last_value,
            gamma=self.config.get('gamma', 0.99),
            gae_lambda=self.config.get('gae_lambda', 0.95),
        )

        # 2. PPO 多 epoch 更新
        self.policy.train()
        clip_eps = self.config.get('clip_eps', 0.1)
        vf_coef = self.config.get('vf_coef', 0.5)
        ent_coef = self.config.get('ent_coef', 0.01)
        max_grad_norm = self.config.get('max_grad_norm', 0.5)
        n_epochs = self.config.get('n_epochs', 4)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            for batch in self.buffer.iter_batches(self.batch_size):
                logits, values = self.policy(batch['obs'])
                dist = CategoricalDist(logits)
                new_log_probs = dist.log_prob(batch['action'])
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate)
                adv = batch['advantage']
                if self.config.get('normalize_advantage', False):
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = torch.exp(new_log_probs - batch['log_prob'])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch['returns'])

                # Total loss
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates * vf_coef,
            'entropy': total_entropy / n_updates * ent_coef,
        }

    def predict(self, obs, deterministic=False):
        self.policy.eval()
        with torch.no_grad():
            obs = obs.unsqueeze(0) if obs.ndim == 3 else obs  # 如果是单个 obs，增加 batch 维度
            logits, _ = self.policy(obs)
            dist = CategoricalDist(logits)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
        # 单环境返回 int，向量化环境返回 tensor
        if action.numel() == 1:
            return action.item()
        return action
