import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from blocks.mlp import _init_weights, build_mlp
# from blocks.distributions import GaussianDist
from blocks.replay_buffer import ReplayBuffer
from utils import AGENT


class QNetwork(nn.Module):
    """Critic 网络：输入 obs 和 action，输出 Q 值"""

    def __init__(self, obs_dim, act_dim, hidden_dims, activation='relu'):
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, 1, hidden_dims, activation)
        _init_weights(self.net)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q_value = self.net(x)
        return q_value


class ActorNetwork(nn.Module):
    """Actor 网络：输入 obs，输出 mean 和 log_std"""

    def __init__(self, obs_dim, act_dim, hidden_dims, activation='relu'):
        super().__init__()
        self.net = build_mlp(obs_dim, act_dim * 2, hidden_dims, activation)
        _init_weights(self.net)

    def forward(self, obs):
        mean_log_std = self.net(obs.float())
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        return mean, log_std
    
    def get_action(self, obs):
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        raw_action = mean + std * torch.randn_like(mean)  # reparameterization trick
        action = torch.tanh(raw_action)  # squash to [-1, 1]
        # log_prob = log N(raw_action | mean, std) - sum(log(1 - tanh^2(raw_action)))
        log_prob = (-0.5 * (((raw_action - mean) / (std + 1e-6)) ** 2 + 2 * log_std + math.log(2 * math.pi))
                    - torch.log(1 - action ** 2 + 1e-6)).sum(dim=-1, keepdim=True)
        return action, log_prob


@AGENT.register('SAC')
class SACAgent(BaseAgent):
    def init_model(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        self.actor = ActorNetwork(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.critic1 = QNetwork(
            obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.critic2 = QNetwork(
            obs_dim, act_dim, hidden_dims, activation).to(self.device)

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=self.config['lr'])
        self.optimizer_critic = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config['lr'],
        )

        self.batch_size = self.config.get('batch_size', 256)
        self.buffer = ReplayBuffer(self.config.get('buffer_size', 1000000))

        self.alpha = self.config.get('alpha', 0.2)
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.policy_delay = self.config.get('policy_delay', 1)

        self.prefill_replay_buffer()

        self._obs, _ = self.env.reset()
        self._episode_reward = 0.0

    def prefill_replay_buffer(self):
        """预先填充经验回放缓冲区"""
        obs, _ = self.env.reset()
        done = False

        while len(self.buffer) < self.batch_size:
            action = torch.tensor(
                self.env.action_space.sample(), dtype=torch.float32, device=self.device)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                done = False

    def collect(self) -> dict:
        """Step-based collection"""
        info = {'n_steps': 1, 'n_episodes': 0}
        self.actor.eval()

        # get action
        with torch.no_grad():
            action, _ = self.actor.get_action(self._obs)

        # step in env
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self._episode_reward += reward
        self.buffer.push(self._obs, action, reward, next_obs, done)

        if done:
            info['n_episodes'] = 1
            info['episode_reward'] = self._episode_reward
            self._obs, _ = self.env.reset()
            self._episode_reward = 0.0
        else:
            self._obs = next_obs

        return info

    def update(self) -> dict:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

        # sample data from replay buffer
        batch = self.buffer.sample(self.batch_size, device=self.device)
        obs, actions, rewards, next_obs, dones = batch['obs'], batch[
            'action'], batch['reward'], batch['next_obs'], batch['done']
        obs = obs.float()
        next_obs = next_obs.float()

        # calculate target Q value
        with torch.no_grad(): 
            next_action, next_log_prob = self.actor.get_action(next_obs)
            target_q1 = self.target_critic1(next_obs, next_action)
            target_q2 = self.target_critic2(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = rewards + (1 - dones) * self.gamma * target_q

        # update critics
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)

        critic_loss = F.mse_loss(current_q1, target_value) + \
            F.mse_loss(current_q2, target_value)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # update actor
        if self.steps % self.policy_delay == 0:
            action, log_prob = self.actor.get_action(obs)
            q1 = self.critic1(obs, action)

            actor_loss = (self.alpha * log_prob - q1).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
        else:
            actor_loss = obs.sum() * 0.0  # dummy loss for logging

        # soft update target critics
        for p_target, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p_target, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            p_target.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return_dict = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }

        return return_dict

    def predict(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.get_action(obs)
        return action
    
    def get_state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }
    
    def load_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.target_critic1.load_state_dict(state_dict['target_critic1'])
        self.target_critic2.load_state_dict(state_dict['target_critic2'])
        self.optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self.optimizer_critic.load_state_dict(state_dict['optimizer_critic'])
