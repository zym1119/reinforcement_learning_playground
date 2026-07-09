import torch


class RolloutBuffer:
    """
    Rollout 缓冲区（On-policy 算法: PPO, A2C）。
    存储并行环境下 n_steps 的轨迹数据，支持 mini-batch 采样。
    """

    def __init__(self, n_steps: int, num_envs: int, obs_shape: tuple, device: torch.device = None):
        """
        Args:
            n_steps: 每次 rollout 采集的步数
            num_envs: 并行环境数量
            obs_shape: 单个 obs 的 shape（不含 batch 维度）
            device: 存储 tensor 的设备
        """
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = device or torch.device('cpu')

        self.obs = torch.zeros(n_steps, num_envs, *obs_shape, device=self.device)
        self.actions = torch.zeros(n_steps, num_envs, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(n_steps, num_envs, device=self.device)
        self.rewards = torch.zeros(n_steps, num_envs, device=self.device)
        self.dones = torch.zeros(n_steps, num_envs, device=self.device)
        self.values = torch.zeros(n_steps, num_envs, device=self.device)
        self.advantages = torch.zeros(n_steps, num_envs, device=self.device)
        self.returns = torch.zeros(n_steps, num_envs, device=self.device)

        self.pos = 0
        self.full = False

    def push(self, obs, action, log_prob, reward, done, value):
        """
        添加一步数据。

        Args:
            obs: (num_envs, *obs_shape)
            action: (num_envs,)
            log_prob: (num_envs,)
            reward: (num_envs,)
            done: (num_envs,)
            value: (num_envs,)
        """
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def sample(self, batch_size: int, device: torch.device = None) -> dict:
        """
        随机采样 mini-batch。将 (n_steps, num_envs) 展平后随机抽取 batch_size 条。

        Returns:
            dict with keys: 'obs', 'action', 'log_prob', 'reward', 'done',
                            'value', 'advantage', 'returns'
        """
        total = self.n_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)[:batch_size]

        flat_obs = self.obs.reshape(total, *self.obs_shape)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_rewards = self.rewards.reshape(total)
        flat_dones = self.dones.reshape(total)
        flat_values = self.values.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)

        result = {
            'obs': flat_obs[indices],
            'action': flat_actions[indices],
            'log_prob': flat_log_probs[indices],
            'reward': flat_rewards[indices],
            'done': flat_dones[indices],
            'value': flat_values[indices],
            'advantage': flat_advantages[indices],
            'returns': flat_returns[indices],
        }

        if device:
            result = {k: v.to(device) for k, v in result.items()}

        return result

    def iter_batches(self, batch_size: int, device: torch.device = None):
        """
        遍历所有数据，按 batch_size 分批 yield。用于 PPO 多 epoch 更新。

        Yields:
            dict: 同 sample() 返回格式
        """
        total = self.n_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        flat_obs = self.obs.reshape(total, *self.obs_shape)
        flat_actions = self.actions.reshape(total)
        flat_log_probs = self.log_probs.reshape(total)
        flat_rewards = self.rewards.reshape(total)
        flat_dones = self.dones.reshape(total)
        flat_values = self.values.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            batch_idx = indices[start:start + batch_size]
            result = {
                'obs': flat_obs[batch_idx],
                'action': flat_actions[batch_idx],
                'log_prob': flat_log_probs[batch_idx],
                'reward': flat_rewards[batch_idx],
                'done': flat_dones[batch_idx],
                'value': flat_values[batch_idx],
                'advantage': flat_advantages[batch_idx],
                'returns': flat_returns[batch_idx],
            }
            if device:
                result = {k: v.to(device) for k, v in result.items()}
            yield result

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        使用 GAE 计算 advantage 和 returns。

        Args:
            last_value: (num_envs,) 最后一条 obs 的 value，用于末尾 bootstrap
            gamma: 折扣因子
            gae_lambda: GAE 的 lambda 参数
        """
        advantages = torch.zeros(self.n_steps, self.num_envs, device=self.device)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        next_value = last_value

        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # Â_t = δ_t + γ * λ * (1 - done_t) * Â_{t+1}
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]

        returns = advantages + self.values

        # save to buffer
        self.advantages = advantages
        self.returns = returns

    def reset(self):
        """清空 buffer，准备下一轮 rollout"""
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.pos * self.num_envs if not self.full else self.n_steps * self.num_envs

    def __repr__(self):
        return (f"RolloutBuffer(n_steps={self.n_steps}, num_envs={self.num_envs}, "
                f"size={len(self)}, full={self.full})")
