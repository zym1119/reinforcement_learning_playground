from abc import ABC, abstractmethod

import gymnasium as gym
import torch

from utils import (
    Logger,
    get_device,
    set_seed,
    setup_run_dir,
)


class BaseTrainer(ABC):
    """
    训练器基类。子类只需实现 4 个方法：
    - init_model(): 初始化网络、优化器
    - collect() -> dict: 与环境交互采集数据
    - update() -> dict: 执行梯度更新
    - predict(obs, deterministic) -> action: 根据观测选动作
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device(config.get('device', 'auto'))
        set_seed(config.get('seed', 42))

        # 环境
        self.env = gym.make(config['env'], **config.get('env_kwargs', {}))
        self.eval_env = gym.make(config['env'], **config.get('env_kwargs', {}))

        # 训练状态
        self.total_steps = 0
        self.best_eval_reward = -float('inf')

        # 由子类实现
        self.init_model()

    # ==================== 子类必须实现 ====================

    @abstractmethod
    def init_model(self):
        """初始化网络结构和优化器"""
        ...

    @abstractmethod
    def collect(self) -> dict:
        """
        与环境交互采集数据，返回字典必须包含:
        - 'n_steps': 本次采集的步数
        - 'episode_reward': 最近一个完成 episode 的总奖励（可选）
        """
        ...

    @abstractmethod
    def update(self) -> dict:
        """执行梯度更新，返回 loss 等指标的字典"""
        ...

    @abstractmethod
    def predict(self, obs, deterministic: bool = False):
        """根据观测选择动作，返回 numpy/int 格式的 action"""
        ...

    # ==================== 训练主循环 ====================

    def train(self):
        self.before_train()
        while self.total_steps < self.config['total_steps']:
            collect_info = self.collect()
            self.total_steps += collect_info['n_steps']
            train_info = self.update()
            self.after_update(collect_info, train_info)
        self.after_train()

    # ==================== Hooks ====================

    def before_train(self):
        """训练前：建目录、初始化 logger"""
        self.run_dir = setup_run_dir(self.config)
        self.logger = Logger(self.run_dir)
        self.logger.info(f'Algorithm: {self.config["algorithm"]}')
        self.logger.info(f'Env: {self.config["env"]}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Run dir: {self.run_dir}')

    def after_update(self, collect_info: dict, train_info: dict):
        """每次 update 后：logging + eval + checkpoint"""
        n_steps = self.config.get('n_steps', 1)
        total = self.total_steps

        # Logging
        if total % self.config.get('log_interval', 1000) < n_steps:
            for k, v in train_info.items():
                self.logger.log_scalar(f'train/{k}', v, total)
            if 'episode_reward' in collect_info:
                self.logger.log_scalar(
                    'train/episode_reward', collect_info['episode_reward'], total)
            info_str = ', '.join(f'{k}: {v:.4f}' for k, v in train_info.items())
            reward_str = f", reward: {collect_info.get('episode_reward', 'N/A')}"
            self.logger.info(f'[Step {total}] {info_str}{reward_str}')

        # Evaluation
        if total % self.config.get('eval_interval', 10000) < n_steps:
            eval_reward = self.evaluate()
            self.logger.log_scalar('eval/reward', eval_reward, total)
            self.logger.info(f'[Step {total}] eval_reward: {eval_reward:.2f}')
            if eval_reward > self.best_eval_reward:
                self.best_eval_reward = eval_reward
                self.save_checkpoint('model_best.pth')

        # Periodic checkpoint
        if total % self.config.get('save_interval', 20000) < n_steps:
            self.save_checkpoint(f'model_{total}.pth')

    def after_train(self):
        """训练结束：保存最终模型、关闭资源"""
        self.save_checkpoint('model_last.pth')
        self.logger.info(
            f'Training done. Best eval reward: {self.best_eval_reward:.2f}')
        self.logger.close()
        self.env.close()
        self.eval_env.close()

    # ==================== 工具方法 ====================

    def evaluate(self, n_episodes: int = 10) -> float:
        """用 eval_env 评估当前策略，返回平均 reward"""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
        return sum(rewards) / len(rewards)

    def save_checkpoint(self, filename: str):
        """保存模型 checkpoint"""
        path = self.run_dir / 'checkpoints' / filename
        torch.save(self.get_state_dict(), path)
        self.logger.info(f'Saved: {path}')

    def get_state_dict(self) -> dict:
        """获取需要保存的状态字典，子类可 override 以支持多网络"""
        return self.model.state_dict()


class BaseInferer(ABC):
    """
    推理器基类。子类需实现：
    - init_model(ckpt_path): 初始化网络并加载权重
    - predict(obs, deterministic) -> action
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device(config.get('device', 'auto'))
        self.env = gym.make(
            config['env'],
            render_mode='human',
            **config.get('env_kwargs', {}),
        )
        self.steps = config.get('eval_steps', 1000)

        ckpt_path = config.get('ckpt_path')
        if not ckpt_path:
            raise ValueError("config must contain 'ckpt_path' for inference")
        self.init_model(ckpt_path)

    @abstractmethod
    def init_model(self, ckpt_path: str):
        """初始化模型并加载 checkpoint"""
        ...

    @abstractmethod
    def predict(self, obs, deterministic: bool = True):
        """根据观测选择动作"""
        ...

    def run(self):
        """推理主循环"""
        obs, _ = self.env.reset()
        total_reward = 0.0
        for step in range(self.steps):
            action = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                print(f'Episode ended at step {step + 1}, reward: {total_reward:.2f}')
                obs, _ = self.env.reset()
                total_reward = 0.0
        self.env.close()
