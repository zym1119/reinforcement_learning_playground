import gymnasium as gym
import torch


class VecNumpyToTorch(gym.vector.VectorWrapper):
    """
    向量化环境的 numpy->torch 转换 wrapper。
    """

    def __init__(self, env: gym.vector.VectorEnv, device: str = 'cpu'):
        super().__init__(env)
        self.device = torch.device(device)

    def _to_torch(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._to_torch(obs)
        return obs, info

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._to_torch(obs)
        reward = self._to_torch(reward)
        terminated = self._to_torch(terminated, dtype=torch.bool)
        truncated = self._to_torch(truncated, dtype=torch.bool)
        return obs, reward, terminated, truncated, info
