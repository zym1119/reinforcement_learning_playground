import argparse
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

    return config


def load_config() -> dict:
    """从命令行加载配置：--config 指定 YAML，--overrides 覆盖参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='path to config yaml file')
    parser.add_argument('--overrides', nargs='+', default=[],
                        help='key=value pairs, e.g. lr=1e-4 seed=123')
    args = parser.parse_args()

    config = load_yaml(args.config)

    # CLI overrides
    for item in args.overrides:
        k, v = item.split('=', 1)
        config[k] = yaml.safe_load(v)

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


# ==================== Run Directory ====================

def setup_run_dir(config: dict) -> Path:
    """创建 work_dir 及子目录，保存配置副本"""
    env_name = config['env'].replace('/', '-')
    algo = config['algorithm']
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    run_name = config.get('run_name') or f"{algo}_{env_name}_{timestamp}"
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

        fh = logging.FileHandler(run_dir / 'logs' / 'train.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(run_dir / 'tb_logs'))
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


TRAINER = Registry('trainer')
INFERER = Registry('inferer')


def create_trainer(config: dict):
    """根据 config['algorithm'] 创建 Trainer 实例"""
    import agents  # noqa: F401
    trainer_cls = TRAINER.get(config['algorithm'])
    return trainer_cls(config)


def create_inferer(config: dict):
    """根据 config['algorithm'] 创建 Inferer 实例"""
    import agents  # noqa: F401
    inferer_cls = INFERER.get(config['algorithm'])
    return inferer_cls(config)
