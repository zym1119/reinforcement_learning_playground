import logging
import time
from pathlib import Path
from typing import Optional

import torch


logger = logging.getLogger(__name__)


def prepare_run(run_name: Optional[str] = None, root_dir='./runs', seed=666):
    # get run name
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")

    # make run folder
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    # init logger
    log_file = run_dir / 'log.txt'
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(ch)

    # log some useful info
    logger.info(f'Run name: {run_name}')
    logger.info(f'Run folder: {run_dir}')

    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return run_dir, logger


def get_trainer(model_type, env, run_dir, **kwargs):
    if model_type == 'PolicyGradient':
        from models.policy_gradient import get_pg_trainer
        trainer = get_pg_trainer(env, run_dir, **kwargs)
    elif model_type == 'DQN':
        from models.dqn import get_dqn_trainer
        trainer = get_dqn_trainer(env, run_dir, **kwargs)
    else:
        raise NotImplementedError

    return trainer


def get_inferer(model_type, env, ckpt_path, **kwargs):
    if model_type == 'PolicyGradient':
        from models.policy_gradient import get_pg_inferer
        inferer = get_pg_inferer(env, ckpt_path, **kwargs)
    elif model_type == 'DQN':
        from models.dqn import get_dqn_inferer
        inferer = get_dqn_inferer(env, ckpt_path, **kwargs)
    else:
        raise NotImplementedError
    return inferer
