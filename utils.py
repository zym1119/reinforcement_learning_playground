import logging
import time
from pathlib import Path
from typing import Optional
from importlib import import_module

import torch

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def prepare_run(run_name: Optional[str] = None, root_dir='./runs', seed=666):
    """
    为训练运行做准备工作，包括生成运行名称、创建运行目录、初始化日志记录和设置随机种子。

    参数:
        run_name (Optional[str]): 运行的名称，如果为 None，则使用当前时间作为名称。
        root_dir (str): 运行目录的根路径，默认为 './runs'。
        seed (int): 随机种子，用于确保结果的可重复性，默认为 666。

    返回:
        run_dir (Path): 运行目录的路径对象。
        logger (logging.Logger): 配置好的日志记录器。
    """
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")

    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    log_file = run_dir / 'log.txt'
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(ch)

    logger.info(f'Run name: {run_name}')
    logger.info(f'Run folder: {run_dir}')

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return run_dir, logger


class Registry:
    """通用组件注册器"""

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def register(self, *obj_name):
        # 用作装饰器
        def wrap(obj_cls):
            if obj_name is None:
                names = [obj_cls.__name__]
            else:
                names = list(obj_name)
            for name in names:
                if name in self._obj_map:
                    raise KeyError(
                        f'Duplicate key {name} in registry {self._name}')
                self._obj_map[name] = obj_cls
            return obj_cls

        return wrap

    def create(self, obj_name, **kwargs):
        obj_cls = self._obj_map.get(obj_name)
        if not obj_cls:
            raise KeyError(
                f"Object {obj_name} not found in registry {self._name}")
        return obj_cls(**kwargs)

    def get(self, obj_name):
        return self._obj_map.get(obj_name)

    @property
    def models(self):
        return list(self._obj_map.keys())


TRAINER = Registry('trainer')
INFERER = Registry('inferer')


def get_trainer(model_type, env, run_dir, **kwargs):
    # load registry
    import_module('models')
    return TRAINER.create(model_type, env=env, run_dir=run_dir, **kwargs)


def get_inferer(model_type, env, ckpt_path, **kwargs):
    # load registry
    import_module('models')
    return INFERER.create(model_type, env=env, ckpt_path=ckpt_path, **kwargs)
