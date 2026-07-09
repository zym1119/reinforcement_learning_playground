from typing import Any

import gymnasium as gym
from gymnasium.wrappers import NumpyToTorch

from env_utils.normalize_wrapper import NormalizeObservation, NormalizeReward


def _is_atari(env_id: str) -> bool:
    """判断是否为 Atari 环境（通过 env_id 包含 NoFrameskip 或 ALE/ 前缀）"""
    return 'NoFrameskip' in env_id or env_id.startswith('ALE/')


def _make_single_env(
    env_id: str,
    env_kwargs: dict | None = None,
    normalize: dict | None = None,
    gamma: float = 0.99,
    device: str = 'cpu',
    training: bool = True,
    render_mode: str | None = None,
    atari_wrapper: dict | None = None,
    seed: int | None = None,
) -> gym.Env:
    """
    创建单个 gymnasium 环境，内部使用。

    Args:
        env_id: 环境 ID
        env_kwargs: 传给 gym.make 的额外参数
        normalize: 归一化配置字典
        gamma: discount factor
        device: torch device
        training: 是否为训练模式
        render_mode: 渲染模式
        atari_wrapper: Atari wrapper 配置（非 None 时应用 AtariWrapper）
        seed: 随机种子
    """
    if env_kwargs is None:
        env_kwargs = {}
    if normalize is None:
        normalize = {}

    make_kwargs = dict(env_kwargs)
    if render_mode is not None:
        make_kwargs['render_mode'] = render_mode

    env = gym.make(env_id, **make_kwargs)

    # Atari preprocessing
    if atari_wrapper is not None:
        from env_utils.atari_wrappers import AtariWrapper
        env = AtariWrapper(env, **atari_wrapper)

    # Observation normalization
    if normalize.get('obs', False):
        clip_obs = normalize.get('clip_obs', 10.0)
        env = NormalizeObservation(env, clip=clip_obs)
        if not training:
            env.training = False

    # Reward normalization (only in training)
    if training and normalize.get('reward', False):
        clip_reward = normalize.get('clip_reward', 10.0)
        env = NormalizeReward(env, gamma=gamma, clip=clip_reward)

    # NumpyToTorch wrapper (outermost)
    env = NumpyToTorch(env, device=device)

    if seed is not None:
        env.reset(seed=seed)

    return env


def _make_vec_env(
    env_id: str,
    num_envs: int,
    env_kwargs: dict | None = None,
    normalize: dict | None = None,
    gamma: float = 0.99,
    device: str = 'cpu',
    training: bool = True,
    render_mode: str | None = None,
    atari_wrapper: dict | None = None,
    seed: int | None = None,
    async_envs: bool = False,
) -> gym.vector.VectorEnv:
    """
    创建向量化环境，内部使用。
    """
    if env_kwargs is None:
        env_kwargs = {}
    if normalize is None:
        normalize = {}

    def _make_env_fn(rank: int):
        def _init():
            make_kwargs = dict(env_kwargs)
            if render_mode is not None:
                make_kwargs['render_mode'] = render_mode
            else:
                make_kwargs.setdefault('render_mode', 'rgb_array')

            env = gym.make(env_id, **make_kwargs)

            # Atari preprocessing
            if atari_wrapper is not None:
from env_utils.atari_wrappers import AtariWrapper
                env = AtariWrapper(env, **atari_wrapper)

            # Observation normalization
            if normalize.get('obs', False):
                clip_obs = normalize.get('clip_obs', 10.0)
                env = NormalizeObservation(env, clip=clip_obs)
                if not training:
                    env.training = False

            # Reward normalization (only in training)
            if training and normalize.get('reward', False):
                clip_reward = normalize.get('clip_reward', 10.0)
                env = NormalizeReward(env, gamma=gamma, clip=clip_reward)

            env.reset(seed=seed + rank if seed is not None else None)
            return env
        return _init

    vec_env_cls = gym.vector.AsyncVectorEnv if async_envs else gym.vector.SyncVectorEnv
    return vec_env_cls([_make_env_fn(i) for i in range(num_envs)])


def make_env(
    env_id: str,
    num_envs: int | None = None,
    env_kwargs: dict | None = None,
    normalize: dict | None = None,
    gamma: float = 0.99,
    device: str = 'cpu',
    training: bool = True,
    render_mode: str | None = None,
    atari_wrapper: dict | bool | None = None,
    seed: int | None = None,
    async_envs: bool = False,
) -> gym.Env | gym.vector.VectorEnv:
    """
    统一环境创建接口。

    Args:
        env_id: 环境 ID (e.g. "CartPole-v1", "PongNoFrameskip-v4")
        num_envs: 并行环境数量。None 或 1 时创建单环境，>1 时创建向量化环境
        env_kwargs: 传给 gym.make 的额外参数
        normalize: 归一化配置字典，支持:
            - obs (bool): 是否归一化 observation
            - reward (bool): 是否归一化 reward（仅 training=True 时生效）
            - clip_obs (float): obs 裁剪范围，默认 10.0
            - clip_reward (float): reward 裁剪范围，默认 10.0
        gamma: discount factor，用于 reward 归一化
        device: torch device（仅单环境时包裹 NumpyToTorch）
        training: 是否为训练模式（False 时不更新归一化统计量，不添加 reward 归一化）
        render_mode: 渲染模式
        atari_wrapper: Atari wrapper 配置:
            - None: 自动检测（env_id 含 NoFrameskip 或 ALE/ 前缀时启用默认配置）
            - True: 使用默认 AtariWrapper 配置
            - False: 不使用 AtariWrapper
            - dict: 自定义 AtariWrapper 参数
        seed: 随机种子
        async_envs: 向量化时是否使用多进程（AsyncVectorEnv）

    Returns:
        num_envs <= 1: 单个 gym.Env（含 NumpyToTorch）
        num_envs > 1: gym.vector.VectorEnv（不含 NumpyToTorch）
    """
    # 解析 atari_wrapper 参数
    if atari_wrapper is None:
        # 自动检测
        atari_cfg = {} if _is_atari(env_id) else None
    elif atari_wrapper is True:
        atari_cfg = {}
    elif atari_wrapper is False:
        atari_cfg = None
    else:
        atari_cfg = dict(atari_wrapper)

    if num_envs is None or num_envs <= 1:
        # 单环境
        return _make_single_env(
            env_id=env_id,
            env_kwargs=env_kwargs,
            normalize=normalize,
            gamma=gamma,
            device=device,
            training=training,
            render_mode=render_mode,
            atari_wrapper=atari_cfg,
            seed=seed,
        )
    else:
        # 向量化环境
        return _make_vec_env(
            env_id=env_id,
            num_envs=num_envs,
            env_kwargs=env_kwargs,
            normalize=normalize,
            gamma=gamma,
            device=device,
            training=training,
            render_mode=render_mode,
            atari_wrapper=atari_cfg,
            seed=seed,
            async_envs=async_envs,
        )

