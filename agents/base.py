from abc import ABC, abstractmethod
import time

import gymnasium as gym
import torch
import numpy as np

from env_utils.normalize_wrapper import NormalizeObservation, NormalizeReward
from env_utils.make import make_env
from utils import (
    Logger,
    get_device,
    set_seed,
    setup_run_dir,
)


class BaseAgent(ABC):
    """
    RL Agent 基类，统一训练与推理。子类需实现 4 个方法：
    - init_model(): 初始化网络、优化器
    - collect() -> dict: 与环境交互采集数据
    - update() -> dict: 执行梯度更新
    - predict(obs, deterministic) -> action: 根据观测选动作
    """

    def __init__(self, config: dict, mode: str = 'train'):
        """
        Args:
            config: 配置字典
            mode: 'train' 或 'eval'
        """
        self.config = config
        self.mode = mode
        self.device = get_device(config.get('device', 'auto'))
        self.seed = config.get('seed', 42)
        set_seed(self.seed)

        # 环境
        norm_cfg = config.get('normalize', {})
        gamma = config.get('gamma', 0.99)

        if mode == 'train':
            self.env = make_env(
                config['env'],
                num_envs=config.get('num_envs'),
                env_kwargs=config.get('env_kwargs'),
                atari_wrapper=config.get('atari_wrapper'),
                normalize=norm_cfg,
                gamma=gamma,
                device=self.device,
                training=True,
                seed=self.seed,
            )
            self.eval_env = make_env(
                config['env'],
                env_kwargs=config.get('env_kwargs'),
                atari_wrapper=config.get('atari_wrapper'),
                normalize=norm_cfg,
                gamma=gamma,
                device=self.device,
                training=False,
                seed=self.seed + 1000,
            )
        else:
            eval_cfg = config.get('evaluation', {})
            self.dump_video = eval_cfg.get('dump_video', False)
            render_mode = 'rgb_array' if self.dump_video else eval_cfg.get('render_mode', 'rgb_array')
            self.env = make_env(
                config['env'],
                env_kwargs=config.get('env_kwargs'),
                atari_wrapper=config.get('atari_wrapper'),
                normalize=norm_cfg,
                gamma=gamma,
                device=self.device,
                training=False,
                render_mode=render_mode,
                seed=self.seed,
            )
            # eval 模式下的 episode/step 限制
            self.total_episodes = eval_cfg.get('total_episodes')
            self.total_steps = eval_cfg.get('total_steps')
            if not self.total_episodes and not self.total_steps:
                self.total_episodes = 10
            self.video_fps = eval_cfg.get('video_fps', 30)

        # 训练状态
        self.steps = 0
        self.episodes = 0
        self.n_updates = 0
        self.best_eval_reward = -float('inf')
        self._latest_episode_reward = None
        self._last_eval_step = 0
        self._last_save_step = 0

        # 初始化模型
        self.init_model()

        # eval 模式加载 checkpoint
        if mode == 'eval':
            ckpt_path = config.get('ckpt_path')
            if not ckpt_path:
                raise ValueError("config must contain 'ckpt_path' for eval mode")
            self.load_checkpoint(ckpt_path)

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

    @property
    def episode_based(self) -> bool:
        """判断训练模式：config 中有 total_episodes 且大于 0 则为 episode-based"""
        return 'total_episodes' in self.config and self.config['total_episodes'] > 0

    @property
    def _counter(self) -> int:
        """当前进度计数（episode 或 step）"""
        return self.episodes if self.episode_based else self.steps

    @property
    def _target(self) -> int:
        """目标计数"""
        if self.episode_based:
            return self.config['total_episodes']
        return self.config['total_steps']

    def train(self):
        self.before_train()
        self._init_scheduler()
        while self._counter < self._target:
            collect_info = self.collect()
            self.steps += collect_info.get('n_steps', 1)
            self.episodes += collect_info.get('n_episodes', 0)
            if 'episode_reward' in collect_info:
                self._latest_episode_reward = collect_info['episode_reward']
            train_info = self.update()
            self.n_updates += 1
            self._step_scheduler()
            self.after_update(collect_info, train_info)
        self.after_train()

    # ==================== Training Hooks ====================

    def before_train(self):
        """训练前：建目录、初始化 logger"""
        self.run_dir = setup_run_dir(self.config)
        self.logger = Logger(self.run_dir)
        self.logger.info(f'Algorithm: {self.config["algorithm"]}')
        self.logger.info(f'Env: {self.config["env"]}')
        self.logger.info(f'Device: {self.device}')
        mode = 'episode-based' if self.episode_based else 'step-based'
        self.logger.info(f'Mode: {mode}, target: {self._target}')
        self.logger.info(f'Run dir: {self.run_dir}')
        self._train_start_time = time.time()

    def after_update(self, collect_info: dict, train_info: dict):
        """每次 update 后：logging + eval + checkpoint"""
        # Logging（基于 update 计数）
        log_interval = self.config.get('log_interval', 10)
        if self.n_updates % log_interval == 0:
            for k, v in train_info.items():
                self.logger.log_scalar(f'train/{k}', v, self.steps)
            if self._latest_episode_reward is not None:
                self.logger.log_scalar(
                    'train/episode_reward', self._latest_episode_reward, self.steps)
            # 记录当前 lr
            lr = self.get_current_lr()
            if lr is not None:
                self.logger.log_scalar('train/lr', lr, self.steps)
            info_str = ', '.join(f'{k}: {v:.4f}' for k, v in train_info.items())
            reward_val = f"{self._latest_episode_reward:.2f}" if self._latest_episode_reward is not None else "N/A"
            reward_str = f", reward: {reward_val}"
            lr_str = f"lr: {lr:.6f}, " if lr is not None else ""
            eta_str = self._get_eta_str()
            self.logger.info(f'[Ep {self.episodes}, Step {self.steps}] {lr_str}{info_str}{reward_str}{eta_str}')

        # Evaluation（基于步数差值）
        eval_freq = self.config.get('eval_interval', 50000)
        if self.steps - self._last_eval_step >= eval_freq:
            self._last_eval_step = self.steps
            eval_reward = self.evaluate()
            self.logger.log_scalar('eval/reward', eval_reward, self.steps)
            self.logger.info(f'[Ep {self.episodes}, Step {self.steps}] eval_reward: {eval_reward:.2f}')
            if eval_reward > self.best_eval_reward:
                self.best_eval_reward = eval_reward
                self.save_checkpoint('model_best.pth')

        # Periodic checkpoint（基于步数差值）
        save_freq = self.config.get('save_interval', 100000)
        if self.steps - self._last_save_step >= save_freq:
            self._last_save_step = self.steps
            self.save_checkpoint(f'model_step{self.steps}.pth')

    def after_train(self):
        """训练结束：保存最终模型、关闭资源"""
        self.save_checkpoint('model_last.pth')
        self.logger.info(
            f'Training done. Best eval reward: {self.best_eval_reward:.2f}')
        self.logger.close()
        self.env.close()
        self.eval_env.close()

    # ==================== 工具方法 ====================

    def _init_scheduler(self):
        """根据 config 初始化 lr scheduler"""
        self.scheduler = None
        sched_cfg = self.config.get('lr_scheduler')
        if not sched_cfg or not hasattr(self, 'optimizer'):
            return
        sched_type = sched_cfg.get('type', 'StepLR')
        if sched_type == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get('step_size', 200),
                gamma=sched_cfg.get('gamma', 0.9),
            )
        elif sched_type == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_cfg.get('gamma', 0.99),
            )
        elif sched_type == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg.get('T_max', self._target),
            )
        elif sched_type == 'LinearLR':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=sched_cfg.get('start_factor', 1.0),
                end_factor=sched_cfg.get('end_factor', 0.0),
                total_iters=sched_cfg.get('total_iters', self._target),
            )

    def _step_scheduler(self):
        """每次 update 后调用 scheduler.step()"""
        if self.scheduler is not None:
            self.scheduler.step()

    def get_current_lr(self):
        """获取当前学习率，子类如有多个 optimizer 可 override"""
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return None

    def _get_eta_str(self) -> str:
        """估算剩余训练时间"""
        elapsed = time.time() - self._train_start_time
        progress = self._counter / self._target
        if progress <= 0:
            return ''
        eta_seconds = elapsed / progress * (1 - progress)
        # 格式化为 HH:MM:SS 或 Xd HH:MM:SS
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_s = int(eta_seconds % 60)
        if eta_h >= 24:
            days = eta_h // 24
            eta_h = eta_h % 24
            return f', ETA: {days}d {eta_h:02d}:{eta_m:02d}:{eta_s:02d}'
        return f', ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}'

    def evaluate(self, n_episodes: int = 10) -> float:
        """用 eval_env 评估当前策略，返回平均 reward"""
        # 同步 obs 归一化统计量到 eval_env
        self._sync_obs_rms()
        eval_cfg = self.config.get('evaluation', {})
        deterministic = eval_cfg.get('deterministic', True)
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)
        return sum(rewards) / len(rewards)

    def _sync_obs_rms(self):
        """同步 train env 的 obs 归一化统计量到 eval env"""
        train_env = self.env
        eval_env = self.eval_env
        # 找到 NormalizeObservation wrapper
        while hasattr(train_env, 'env'):
            if isinstance(train_env, NormalizeObservation):
                break
            train_env = train_env.env
        while hasattr(eval_env, 'env'):
            if isinstance(eval_env, NormalizeObservation):
                break
            eval_env = eval_env.env
        if isinstance(train_env, NormalizeObservation) and isinstance(eval_env, NormalizeObservation):
            if hasattr(train_env, 'obs_rms') and hasattr(eval_env, 'obs_rms'):
                eval_env.obs_rms.load_state_dict(train_env.obs_rms.state_dict())

    def _get_normalize_state(self) -> dict:
        """获取归一化 wrapper 的统计量"""
        state = {}
        env = self.env
        while hasattr(env, 'env'):
            if isinstance(env, NormalizeObservation) and hasattr(env, 'obs_rms'):
                state['obs_rms'] = env.obs_rms.state_dict()
            if isinstance(env, NormalizeReward):
                state['return_rms'] = env.return_rms.state_dict()
            env = env.env
        return state

    def _load_normalize_state(self, state: dict):
        """加载归一化 wrapper 的统计量"""
        env = self.env
        while hasattr(env, 'env'):
            if isinstance(env, NormalizeObservation) and 'obs_rms' in state and hasattr(env, 'obs_rms'):
                env.obs_rms.load_state_dict(state['obs_rms'])
            if isinstance(env, NormalizeReward) and 'return_rms' in state:
                env.return_rms.load_state_dict(state['return_rms'])
            env = env.env

    def save_checkpoint(self, filename: str):
        """保存模型 checkpoint"""
        path = self.run_dir / 'checkpoints' / filename
        save_dict = self.get_state_dict()
        norm_state = self._get_normalize_state()
        if norm_state:
            save_dict['normalize'] = norm_state
        torch.save(save_dict, path)
        self.logger.info(f'Saved: {path}')

        # 清理旧的 periodic checkpoint（不删 model_best / model_last）
        max_keep = self.config.get('max_keep_ckpts', -1)
        if max_keep > 0 and filename.startswith('model_step'):
            if not hasattr(self, '_periodic_ckpts'):
                self._periodic_ckpts = []
            self._periodic_ckpts.append(path)
            while len(self._periodic_ckpts) > max_keep:
                old = self._periodic_ckpts.pop(0)
                if old.exists():
                    old.unlink()
                    self.logger.info(f'Removed old ckpt: {old}')

    def load_checkpoint(self, ckpt_path: str):
        """加载 checkpoint，子类可 override 以支持多网络"""
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if 'normalize' in state_dict:
            self._load_normalize_state(state_dict.pop('normalize'))
        self.policy.load_state_dict(state_dict)

    def get_state_dict(self) -> dict:
        """获取需要保存的状态字典，子类可 override 以支持多网络"""
        return self.policy.state_dict()

    # ==================== 推理主循环 ====================

    def run(self):
        """推理主循环：支持 episode-based 和 step-based，可选视频录制"""
        self.run_dir = setup_run_dir(self.config)

        obs, _ = self.env.reset()
        total_reward = 0.0
        episode_rewards = []
        step = 0
        ep_step = 0
        frames = []

        while True:
            # 录制帧（绕过 NumpyToTorch wrapper 直接获取 numpy frame）
            if self.dump_video:
                frame = self.env.unwrapped.render()
                frame = self._overlay_text(frame, len(episode_rewards) + 1, ep_step + 1)
                frames.append(frame)

            action = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            step += 1
            ep_step += 1

            if terminated or truncated:
                episode_rewards.append(total_reward)
                print(f'Episode {len(episode_rewards)} ended at step {step}, '
                      f'ep_steps: {ep_step}, reward: {total_reward:.2f}')
                obs, _ = self.env.reset()
                total_reward = 0.0
                ep_step = 0

            # 终止条件
            if self.total_episodes is not None:
                if len(episode_rewards) >= self.total_episodes:
                    break
            else:
                if step >= self.total_steps:
                    break

        self.env.close()

        # 汇总
        if episode_rewards:
            avg = sum(episode_rewards) / len(episode_rewards)
            print(f'\nSummary: {len(episode_rewards)} episodes, {step} steps, avg reward: {avg:.2f}')

        # 保存视频
        if self.dump_video and frames:
            self._save_video(frames)

    def _overlay_text(self, frame, episode: int, ep_step: int):
        """在帧上叠加 episode 和 step 文字"""
        import cv2

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # 左上角: episode
        ep_text = f'episode: {episode}'
        (tw, th), _ = cv2.getTextSize(ep_text, font, font_scale, thickness)
        cv2.rectangle(frame, (5, 5), (10 + tw, 12 + th), bg_color, -1)
        cv2.putText(frame, ep_text, (7, 10 + th), font, font_scale, color, thickness)

        # 右上角: step
        step_text = f'step: {ep_step}'
        (tw, th), _ = cv2.getTextSize(step_text, font, font_scale, thickness)
        cv2.rectangle(frame, (w - 12 - tw, 5), (w - 5, 12 + th), bg_color, -1)
        cv2.putText(frame, step_text, (w - 10 - tw, 10 + th), font, font_scale, color, thickness)

        return frame

    def _save_video(self, frames):
        """保存视频到 work_dir"""
        import imageio
        import time

        video_dir = self.run_dir / 'videos'
        video_dir.mkdir(exist_ok=True)

        filename = time.strftime("%Y%m%d_%H%M%S") + '.mp4'
        video_path = video_dir / filename
        imageio.mimsave(str(video_path), frames, fps=self.video_fps)
        print(f'Video saved to: {video_path}')
