import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml


# ==================== Config ====================

def deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典，override 覆盖 base"""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_yaml(path: Union[str, Path]) -> dict:
    """加载 YAML 文件，递归处理 _base_ 继承"""
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    if '_base_' in config:
        base_rel = config.pop('_base_')
        base_path = path.parent / base_rel
        base_config = load_yaml(base_path)
        config = deep_merge(base_config, config)

    _auto_cast_numbers(config)
    return config


def _auto_cast_numbers(d: dict):
    """将 YAML 中未被正确解析的数字字符串（如 '1e-2'）转为 float/int"""
    for k, v in d.items():
        if isinstance(v, dict):
            _auto_cast_numbers(v)
        elif isinstance(v, str):
            try:
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass


def load_config(config_path, overrides=None, exp_name=None) -> dict:
    """加载配置文件并应用覆盖参数"""
    config = load_yaml(config_path)

    # overrides
    if overrides:
        for item in overrides:
            k, v = item.split('=', 1)
            config[k] = yaml.safe_load(v)

    # exp_name has highest priority
    if exp_name:
        config['exp_name'] = exp_name

    return config


# ==================== Device & Seed ====================

def get_device(device_str: str = 'auto') -> torch.device:
    """获取设备：auto 自动选择 cuda/cpu"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Run Directory ====================

def setup_run_dir(config: dict) -> Path:
    """创建 work_dir 及子目录，保存配置副本"""
    env_name = config['env'].replace('/', '-')
    algo = config['algorithm']
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # exp_name 优先级: CLI --exp-name > config exp_name > 自动生成
    run_name = config.get('exp_name') or f"{algo}_{env_name}_{timestamp}"
    root = Path(config.get('work_dir', './work_dirs'))
    run_dir = root / run_name

    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    (run_dir / 'tb_logs').mkdir(exist_ok=True)

    # 保存配置副本
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return run_dir


# ==================== Logger ====================

class Logger:
    """统一日志：文本日志 + TensorBoard"""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

        # 文本日志
        self._logger = logging.getLogger('rl')
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        log_filename = time.strftime("%Y%m%d_%H%M") + '.log'
        fh = logging.FileHandler(run_dir / 'logs' / log_filename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # TensorBoard（每次运行独立子目录，方便对比）
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_run_dir = run_dir / 'tb_logs' / time.strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir=str(tb_run_dir))
        except ImportError:
            self.writer = None
            self._logger.warning('TensorBoard not available, skipping.')

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量到 TensorBoard"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def close(self):
        if self.writer:
            self.writer.close()


# ==================== Registry & Factory ====================

class Registry:
    """通用组件注册器"""

    def __init__(self, name: str):
        self._name = name
        self._obj_map = {}

    def register(self, *obj_name):
        """装饰器：注册类到 registry"""
        def wrap(obj_cls):
            names = list(obj_name) if obj_name else [obj_cls.__name__]
            for name in names:
                if name in self._obj_map:
                    raise KeyError(f'Duplicate key {name} in registry {self._name}')
                self._obj_map[name] = obj_cls
            return obj_cls
        return wrap

    def get(self, name: str):
        """获取已注册的类"""
        obj_cls = self._obj_map.get(name)
        if not obj_cls:
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'. "
                f"Available: {list(self._obj_map.keys())}")
        return obj_cls

    @property
    def registered(self) -> list:
        return list(self._obj_map.keys())


AGENT = Registry('agent')


def create_agent(config: dict, mode: str = 'train'):
    """根据 config['algorithm'] 创建 Agent 实例"""
    import agents  # noqa: F401
    agent_cls = AGENT.get(config['algorithm'])
    return agent_cls(config, mode=mode)
