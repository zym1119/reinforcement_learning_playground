import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from blocks.mlp import build_mlp
from blocks.replay_buffer import ReplayBuffer
from utils import AGENT


@AGENT.register('DQN')
class DQNAgent(BaseAgent):
    def init_model(self):
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        hidden_dims = self.config.get('hidden_dims', [128, 128])
        activation = self.config.get('activation', 'relu')

        self.q_net = build_mlp(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.target_q_net = build_mlp(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config['lr'])
        self.gamma = self.config.get('gamma', 0.99)
        self.batch_size = self.config.get('batch_size', 32)

        self.buffer = ReplayBuffer(self.config.get('buffer_size', 10000))
        self.target_update_freq = self.config.get('target_update_freq', 100)
        self.tau = self.config.get('tau', None)  # soft update 系数，设置后启用 soft update
        self.update_count = 0
        
        # 线性 epsilon 衰减
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = self.config.get('epsilon_decay_steps', 10000)
        self.epsilon = self.epsilon_start

        # 持久状态：collect 跨步维护
        self._obs, _ = self.env.reset()
        self._episode_reward = 0.0
    
    def before_train(self):
        super().before_train()
        self.prefill_replay_buffer()  # 在训练前预填充经验回放缓冲区

    def prefill_replay_buffer(self):
        """预先填充经验回放缓冲区"""
        obs, _ = self.env.reset()
        done = False

        while len(self.buffer) < max(self.config.get('buffer_size', 10000) // 10, self.batch_size):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
                done = False
        if hasattr(self, 'logger'):
            self.logger.info(f"Prefilled replay buffer with {len(self.buffer)} transitions.")

    def collect(self) -> dict:
        """Step-based: 每次走 1 步，episode 结束时返回 reward"""
        self.q_net.eval()
        info = {'n_steps': 1, 'n_episodes': 0}

        with torch.no_grad():
            if torch.rand(1).item() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                q_values = self.q_net(self._obs)
                action = q_values.argmax().item()

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
        self.q_net.train()
        # sample data from replay buffer
        batch = self.buffer.sample(self.batch_size, device=self.device)
        q_values = self.q_net(batch['obs']).gather(1, batch['action'].unsqueeze(-1))
        with torch.no_grad():
            next_q_values = self.target_q_net(batch['next_obs']).max(1, keepdim=True)[0]
            target_q_values = batch['reward'] + self.gamma * (1 - batch['done']) * next_q_values

        td_error = q_values - target_q_values
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 同步 target network
        self.update_count += 1
        if self.tau is not None:
            # soft update: θ_target = τ * θ_q + (1-τ) * θ_target
            for p_target, p_q in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                p_target.data.mul_(1 - self.tau).add_(self.tau * p_q.data)
        elif self.update_count % self.target_update_freq == 0:
            # hard update
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 线性衰减 epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.update_count * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

        return {'loss': loss.item(), 'epsilon': self.epsilon}

    def predict(self, obs, deterministic=False):
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(obs)
        if deterministic:
            action = q_values.argmax().item()
        else:
            action = torch.distributions.Categorical(logits=q_values).sample().item()
        return action

    def get_state_dict(self) -> dict:
        return {
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
        }

    def load_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.q_net.load_state_dict(state_dict['q_net'])
        self.target_q_net.load_state_dict(state_dict['target_q_net'])


@AGENT.register('DoubleDQN')
class DoubleDQNAgent(DQNAgent):
    def update(self) -> dict:
        self.q_net.train()
        batch = self.buffer.sample(self.batch_size, device=self.device)
        q_values = self.q_net(batch['obs']).gather(1, batch['action'].unsqueeze(-1))
        with torch.no_grad():
            # Double DQN: 用 online network 选动作，用 target network 评估
            next_actions = self.q_net(batch['next_obs']).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_q_net(batch['next_obs']).gather(1, next_actions)
            target_q_values = batch['reward'] + self.gamma * (1 - batch['done']) * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.tau is not None:
            for p_target, p_q in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                p_target.data.mul_(1 - self.tau).add_(self.tau * p_q.data)
        elif self.update_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.update_count * (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        )

        return {'loss': loss.item(), 'epsilon': self.epsilon}