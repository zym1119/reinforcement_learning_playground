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

        # 用于暂存rollout数据
        self._logits = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []
        self._obs, _ = self.env.reset()
        self._episode_reward = 0.0

    def collect(self) -> dict:
        """跑N step，收集 log_probs、values 和 rewards"""
        self.actor.train()
        self.critic.train()
        
        n_steps = 0
        n_episodes = 0

        return_dict = {}

        while n_steps < self.rollout_length:
            obs_t = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
            logits = self.actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            value = self.critic(obs_t)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            self._logits.append(logits)
            self._log_probs.append(dist.log_prob(action))
            self._values.append(value.squeeze(0))
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
        values = torch.stack(self._values)
        returns = []
        
        # 只计算前 T - n_step 步的 return，bootstrap 全从 values 取
        for t in range(self.rollout_length - self.n_step):
            G = 0.0
            gamma_k = 1.0
            for k in range(self.n_step):
                G += gamma_k * self._rewards[t + k]
                if self._dones[t + k]:
                    gamma_k = 0.0
                    break
                gamma_k *= self.gamma
            # bootstrap V(s_{t+n})
            G += gamma_k * values[t + self.n_step].item()
            returns.append(G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # 计算 advantage 并归一化
        advantages = returns - values[:-self.n_step]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算 actor loss
        log_probs = torch.stack(self._log_probs[:-self.n_step])
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 计算 entropy loss
        logits = torch.stack(self._logits[:-self.n_step])
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        actor_loss -= self.entropy_coef * entropy
        
        # 计算 critic loss
        # critic_loss = F.mse_loss(values[:-self.n_step], returns)
        critic_loss = F.smooth_l1_loss(values[:-self.n_step], returns.detach())

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
        self._logits = []
        self._log_probs = []
        self._values = []
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