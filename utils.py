# 导入 logging 模块，用于记录程序运行过程中的信息
import logging
# 导入 time 模块，用于获取当前时间
import time
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path
# 从 typing 模块导入 Optional 类型，用于指定参数可以为 None
from typing import Optional

# 导入 torch 深度学习库
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

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return run_dir, logger

def get_trainer(model_type, env, run_dir, **kwargs):
    """
    根据模型类型获取相应的训练器。

    参数:
        model_type (str): 模型的类型，如 'PolicyGradient', 'DQN', 'DoubleDQN'。
        env (object): 训练使用的环境。
        run_dir (Path): 运行目录的路径对象。
        **kwargs: 其他可选参数。

    返回:
        trainer (object): 相应的训练器对象。
    """
    if model_type == 'PolicyGradient':
        from models.policy_gradient import get_pg_trainer
        trainer = get_pg_trainer(env, run_dir, **kwargs)
    elif model_type == 'DQN':
        from models.dqn import get_dqn_trainer
        trainer = get_dqn_trainer(env, run_dir, **kwargs)
    elif model_type == 'DoubleDQN':
        from models.double_dqn import get_double_dqn_trainer
        trainer = get_double_dqn_trainer(env, run_dir, **kwargs)
    elif model_type == 'ActorCritic':
        from models.actor_critic import get_a2c_trainer
        trainer = get_a2c_trainer(env, run_dir, **kwargs)
    else:
        raise NotImplementedError

    return trainer

def get_inferer(model_type, env, ckpt_path, **kwargs):
    """
    根据模型类型获取相应的推理器。

    参数:
        model_type (str): 模型的类型，如 'PolicyGradient', 'DQN', 'DoubleDQN'。
        env (object): 推理使用的环境。
        ckpt_path (str): 模型检查点的路径。
        **kwargs: 其他可选参数。

    返回:
        inferer (object): 相应的推理器对象。
    """
    if model_type == 'PolicyGradient':
        from models.policy_gradient import get_pg_inferer
        inferer = get_pg_inferer(env, ckpt_path, **kwargs)
    elif model_type == 'DQN':
        from models.dqn import get_dqn_inferer
        inferer = get_dqn_inferer(env, ckpt_path, **kwargs)
    elif model_type == 'DoubleDQN':
        from models.double_dqn import get_double_dqn_inferer
        inferer = get_double_dqn_inferer(env, ckpt_path, **kwargs)
    elif model_type == 'ActorCritic':
        from models.actor_critic import get_a2c_inferer
        inferer = get_a2c_inferer(env, ckpt_path, **kwargs)
    else:
        raise NotImplementedError
    return inferer
