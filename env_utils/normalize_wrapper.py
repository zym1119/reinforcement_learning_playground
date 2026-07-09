import numpy as np
import gymnasium as gym


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1 and self.mean.shape == x.shape:
            # single sample with shape == obs_shape
            batch_mean = x
            batch_var = np.zeros_like(x)
            batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def state_dict(self):
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class NormalizeObservation(gym.Wrapper):
    """Normalize observations.

    Supports two modes:
        - 'running': running mean/std normalization (default)
        - 'fixed': divide by a fixed scale value (e.g. 255 for uint8 images)
    """
    def __init__(self, env, mode='running', epsilon=1e-8, clip=10.0, scale=255.0, **kwargs):
        super().__init__(env)
        self.mode = mode
        self.epsilon = epsilon
        self.clip = clip
        self.scale = scale
        self.training = True

        if mode == 'running':
            self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)

        # 更新 observation_space 的 dtype 为 float32
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low.astype(np.float32),
            high=obs_space.high.astype(np.float32),
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def normalize(self, obs):
        if self.mode == 'fixed':
            return (obs / self.scale).astype(np.float32)
        # running mean/std
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip, self.clip
        ).astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.mode == 'running' and self.training:
            self.obs_rms.update(obs)
        return self.normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.mode == 'running' and self.training:
            self.obs_rms.update(obs)
        return self.normalize(obs), info


class NormalizeReward(gym.Wrapper):
    """Normalize rewards using running std of discounted returns."""
    def __init__(self, env, gamma=0.99, epsilon=1e-8, clip=10.0, **kwargs):
        super().__init__(env)
        self.return_rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.training = True
        self._returns = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.training:
            self._returns = self._returns * self.gamma + reward
            self.return_rms.update(np.array([self._returns]))
            if terminated or truncated:
                self._returns = 0.0
        # normalize by std only (not mean), same as SB3
        normalized_reward = np.clip(
            reward / np.sqrt(self.return_rms.var + self.epsilon),
            -self.clip, self.clip
        )
        return obs, float(normalized_reward), terminated, truncated, info


def make_normalized_env(env_name, normalize_obs=True, normalize_reward=True,
                        gamma=0.99, clip_obs=10.0, clip_reward=10.0, **env_kwargs):
    """Create an env with optional obs/reward normalization wrappers."""
    env = gym.make(env_name, **env_kwargs)
    if normalize_obs:
        env = NormalizeObservation(env, clip=clip_obs)
    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma, clip=clip_reward)
    return env
