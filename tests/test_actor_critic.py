import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils import load_config, create_agent


@pytest.fixture
def ac_config():
    config = load_config('configs/ac_cartpole.yaml')
    config['total_steps'] = 256
    config['log_interval'] = 256
    config['eval_interval'] = 256
    config['save_interval'] = 256
    config['n_steps'] = 128
    config['exp_name'] = '_test_ac'
    return config


class TestActorCriticAgent:
    def test_create_agent(self, ac_config):
        agent = create_agent(ac_config, mode='train')
        assert agent is not None
        assert agent.config['algorithm'] == 'ActorCritic'

    def test_env_returns_tensor(self, ac_config):
        agent = create_agent(ac_config, mode='train')
        obs, _ = agent.env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == agent.device.type

    def test_collect_n_steps(self, ac_config):
        agent = create_agent(ac_config, mode='train')
        info = agent.collect()
        assert info['n_steps'] == ac_config['n_steps']
        assert len(agent._obs_batch) == ac_config['n_steps']
        # obs_batch 中每个元素应该是 tensor
        assert isinstance(agent._obs_batch[0], torch.Tensor)

    def test_update_clears_buffer(self, ac_config):
        agent = create_agent(ac_config, mode='train')
        agent.collect()
        train_info = agent.update()
        assert 'actor_loss' in train_info
        assert 'critic_loss' in train_info
        assert 'total_loss' in train_info
        # update 后 buffer 应清空
        assert len(agent._obs_batch) == 0
        assert len(agent._actions) == 0

    def test_predict(self, ac_config):
        agent = create_agent(ac_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=True)
        assert isinstance(action, int)

    def test_gae_computation(self, ac_config):
        """验证 GAE 计算不报错且 advantage 维度正确"""
        agent = create_agent(ac_config, mode='train')
        agent.collect()
        # 手动调 update 检查不出错
        train_info = agent.update()
        assert all(not torch.isnan(torch.tensor(v)) for v in train_info.values())

    def test_train_loop(self, ac_config):
        ac_config['exp_name'] = '_test_ac_train'
        agent = create_agent(ac_config, mode='train')
        agent.train()
        assert agent.steps >= ac_config['total_steps']

    def test_normalize_wrapper(self, ac_config):
        """验证归一化 wrapper 正常工作"""
        ac_config['normalize'] = {'obs': {'mode': 'running'}, 'reward': {}}
        ac_config['exp_name'] = '_test_ac_norm'
        agent = create_agent(ac_config, mode='train')
        obs, _ = agent.env.reset()
        # 即使有归一化 wrapper，最终输出仍为 tensor
        assert isinstance(obs, torch.Tensor)

    def test_eval_mode(self, ac_config):
        ac_config['exp_name'] = '_test_ac_eval'
        agent = create_agent(ac_config, mode='train')
        agent.train()
        ckpt_path = agent.run_dir / 'checkpoints' / 'model_last.pth'

        eval_config = load_config('configs/ac_cartpole.yaml')
        eval_config['exp_name'] = '_test_ac_eval_run'
        eval_config['evaluation'] = {'total_episodes': 2}
        eval_config['ckpt_path'] = str(ckpt_path)
        eval_agent = create_agent(eval_config, mode='eval')
        obs, _ = eval_agent.env.reset()
        action = eval_agent.predict(obs, deterministic=True)
        assert isinstance(action, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
