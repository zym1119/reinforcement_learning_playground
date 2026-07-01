import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from blocks.mlp import build_mlp
from blocks.distributions import CategoricalDist
from utils import AGENT


@AGENT.register('ActorCritic')
class ActorCriticAgent(BaseAgent):
    def init_model(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        # 策略网络
        self.actor = build_mlp(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        # 价值网络
        self.critic = build_mlp(obs_dim, 1, hidden_dims, activation).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.config['lr'])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.config['lr'])
        
        self.gamma = self.config.get('gamma', 0.99)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)

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
                logits = self.actor(obs_t)
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

        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        values = self.critic(obs).squeeze(-1)

        # 计算 last_value 用于末尾 bootstrap
        with torch.no_grad():
            last_obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
            last_value = self.critic(last_obs_t).squeeze().item()

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
        actor_loss -= self.entropy_coef * entropy
        
        # critic loss
        critic_loss = F.mse_loss(values, returns)

        # 更新 actor 和 critic
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.optimizer_critic.step()

        # 清空暂存数据
        self._obs_batch = []
        self._actions = []
        self._rewards = []
        self._dones = []

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'critic_lr': self.optimizer_critic.param_groups[0]['lr']
        }

    def predict(self, obs, deterministic=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
        dist = CategoricalDist(logits=logits)
        if deterministic:
            return dist.mode().item()
        return dist.sample().item()
    
    def get_current_lr(self):
        """返回 actor lr，critic lr 通过 train_info 记录"""
        return self.optimizer_actor.param_groups[0]['lr']

    def get_state_dict(self) -> dict:
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }
    
    def load_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self.optimizer_critic.load_state_dict(state_dict['optimizer_critic'])