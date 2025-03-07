from itertools import count
import logging

import torch
import torch.nn as nn


# 获取名为当前模块名的日志记录器
logger = logging.getLogger(__name__)


class BaseTrainer(nn.Module):
    """
    训练器的基类，继承自 nn.Module。
    包含训练过程的基本逻辑，如初始化模型、训练循环、保存模型等。
    """
    def __init__(self, env, run_dir, lr=0.01, num_episodes=-1, log_interval=10, save_interval=100, **kwargs):
        """
        初始化训练器。

        参数:
        env (object): 训练使用的环境。
        run_dir (Path): 模型保存的目录。
        lr (float, 可选): 学习率，默认为 0.01。
        num_episodes (int, 可选): 训练的回合数，-1 表示无限训练，默认为 -1。
        log_interval (int, 可选): 日志记录的间隔回合数，默认为 10。
        save_interval (int, 可选): 模型保存的间隔回合数，默认为 100。
        **kwargs: 其他可选参数。
        """
        super().__init__()
        self.env = env
        self.run_dir = run_dir
        self.lr = lr
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.save_interval = save_interval

        # 调用初始化模型的方法
        self.init_model()

        # 当前训练的回合数
        self.episode = 0
        # 运行奖励，用于跟踪平均奖励
        self.running_reward = 0

    def init_model(self):
        """
        初始化模型的抽象方法，需要在子类中实现。
        """
        raise NotImplementedError

    def train(self):
        """
        训练模型的主方法，包含训练前、训练中、训练后的钩子函数。
        """
        self.before_train_hook()

        # 根据 num_episodes 判断是否为无限训练
        if self.infinite_training:
            iterator = count(1)
        else:
            iterator = range(self.num_episodes)

        for _ in iterator:
            self.before_train_one_episode_hook()
            module_outputs = self.train_one_episode()
            shoule_break = self.after_train_one_episode_hook(module_outputs)
            if shoule_break:
                self.train_break_hook()
                break

        self.after_train_hook()

    def train_one_episode(self):
        """
        训练一个回合的抽象方法，需要在子类中实现。
        """
        raise NotADirectoryError

    def save_model(self, is_last=False):
        """
        保存模型的方法。

        参数:
        is_last (bool, 可选): 是否为最后一次保存，默认为 False。
        """
        if is_last:
            save_path = self.run_dir / 'model_last.pth'
        else:
            save_path = self.run_dir / f'model_episode{self.episode}.pth'
        # 保存模型的状态字典
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'save ckpt {save_path}')

    def before_train_hook(self):
        """
        训练前的钩子函数，初始化优化器并将模型设置为训练模式。
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        logger.info('start training...')

    def before_train_one_episode_hook(self):
        """
        训练一个回合前的钩子函数，默认不做任何操作。
        """
        pass

    def after_train_one_episode_hook(self, module_outputs):
        """
        训练一个回合后的钩子函数，通常用于日志记录和模型保存。

        参数:
        module_outputs (dict): 训练一回合的输出结果。

        返回:
        bool: 是否停止训练。
        """
        self.episode += 1

        self.log_model_outputs(self.episode, module_outputs)

        # 按间隔保存模型
        if self.episode % self.save_interval == 0:
            self.save_model()
        
        # 无限训练循环的停止条件
        if self.infinite_training:
            self.running_reward = 0.99 * self.running_reward + \
                module_outputs['reward'] * 0.01
            if self.running_reward > self.env.spec.reward_threshold:
                logger.info(
                    f'Solved!, running reward is {self.running_reward} at step {self.episode}')
                return True
        return False

    def train_break_hook(self):
        """
        训练中断的钩子函数，记录中断信息。
        """
        logger.info(f'train break at episode {self.episode}')

    def after_train_hook(self):
        """
        训练结束后的钩子函数，保存最终模型。
        """
        self.save_model(is_last=True)

    def log_model_outputs(self, episode, model_outputs):
        """
        记录模型输出的日志信息。

        参数:
        episode (int): 当前回合数。
        model_outputs (dict): 模型输出结果。
        """
        if model_outputs is None:
            return
        if episode % self.log_interval == 0:
            info_list = []
            for k, v in model_outputs.items():
                info_list.append(f'{k}: {v:.4f}')
            log_str = f'Episode {episode}: {", ".join(info_list)}'
            logger.info(log_str)

    @property
    def infinite_training(self):
        """
        判断是否为无限训练的属性。

        返回:
        bool: 如果 num_episodes 小于等于 0，则为无限训练。
        """
        return self.num_episodes <= 0
    
class BaseInferer(nn.Module):
    """
    推理器的基类，继承自 nn.Module。
    包含推理过程的基本逻辑，如初始化模型、推理循环等。
    """
    def __init__(self, env, ckpt_path, steps=1000):
        """
        初始化推理器。

        参数:
        env (object): 推理使用的环境。
        ckpt_path (Path): 模型检查点的路径。
        steps (int, 可选): 推理的总步数，默认为 1000。
        """
        super().__init__()
        self.env = env
        self.steps = steps

        # 调用初始化模型的方法
        self.init_model(ckpt_path)

    def init_model(self, ckpt_path):
        """
        初始化模型的抽象方法，需要在子类中实现。

        参数:
        ckpt_path (Path): 模型检查点的路径。
        """
        raise NotImplementedError

    def infer(self):
        """
        推理的主方法，包含推理前的钩子函数和推理循环。
        """
        self.before_infer_hook()

        steps = 0
        while steps < self.steps:
            episode_steps = self.infer_one_episode()
            steps += episode_steps

        # 关闭环境
        self.env.close()

    def infer_one_episode(self):
        """
        推理一个回合的方法。

        返回:
        int: 该回合的步数。
        """
        done = False
        # 重置环境
        state, _ = self.env.reset()

        steps = 0
        while not done:
            # 渲染环境
            self.env.render()
            # 将状态转换为张量
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # 获取模型输出的 logits
            logits = self.model(state_tensor)
            # 选择动作
            action = self.select_action(logits)
            # 执行动作并获取下一个状态
            next_state, _, done, _, _ = self.env.step(action.item())
            steps += 1

            if done:
                return steps
            else:
                state = next_state

    def select_action(self, logits):
        """
        根据模型输出的 logits 选择动作的方法。

        参数:
        logits (torch.Tensor): 模型输出的 logits。

        返回:
        torch.Tensor: 选择的动作。
        """
        # 创建分类分布
        dist = torch.distributions.Categorical(logits=logits)
        # 从分布中采样动作
        action = dist.sample()
        return action

    def before_infer_hook(self):
        """
        推理前的钩子函数，将模型设置为评估模式并记录开始信息。
        """
        self.model.eval()
        logger.info('start infering...')
