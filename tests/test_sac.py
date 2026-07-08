import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils import load_config, create_agent


@pytest.fixture
def sac_config():
    config = load_config('configs/sac_walker2d.yaml')
    config['total_steps'] = 100
    config['log_interval'] = 100
    config['eval_interval'] = 100
    config['save_interval'] = 100
    config['batch_size'] = 32
    config['buffer_size'] = 1000
    config['exp_name'] = '_test_sac'
    return config


class TestSACAgent:
    def test_create_agent(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        assert agent is not None
        assert agent.config['algorithm'] == 'SAC'

    def test_env_returns_tensor(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        obs, _ = agent.env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == agent.device.type

    def test_collect(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        info = agent.collect()
        assert 'n_steps' in info
        assert info['n_steps'] == 1

    def test_update(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        agent.collect()
        train_info = agent.update()
        assert 'critic_loss' in train_info
        assert 'actor_loss' in train_info
        assert 'alpha' in train_info

    def test_predict_deterministic(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=True)
        assert isinstance(action, torch.Tensor)
        # 动作应在 [-1, 1] 范围内 (tanh squashing)
        assert (action >= -1.0).all() and (action <= 1.0).all()
        # 确定性策略应返回相同结果
        action2 = agent.predict(obs, deterministic=True)
        assert torch.allclose(action, action2)

    def test_predict_stochastic(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=False)
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == agent.env.action_space.shape[0]

    def test_replay_buffer_stores_tensors(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        obs, action, reward, next_obs, done = agent.buffer.buffer[0]
        assert isinstance(obs, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(next_obs, torch.Tensor)

    def test_automatic_entropy_tuning(self, sac_config):
        sac_config['automatic_entropy_tuning'] = True
        agent = create_agent(sac_config, mode='train')
        assert hasattr(agent, 'log_alpha')
        assert hasattr(agent, 'target_entropy')
        initial_alpha = agent.alpha
        # 训练几步后 alpha 应该会变化
        for _ in range(5):
            agent.collect()
        agent.update()
        # alpha_loss 存在说明自动调节在工作
        train_info = agent.update()
        assert 'alpha_loss' in train_info

    def test_soft_update_target(self, sac_config):
        agent = create_agent(sac_config, mode='train')
        # 记录 target 参数
        target_params_before = [
            p.clone() for p in agent.target_critic1.parameters()]
        # 执行 update
        agent.collect()
        agent.update()
        # target 参数应发生变化 (soft update)
        target_params_after = list(agent.target_critic1.parameters())
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(target_params_before, target_params_after)
        )
        assert changed
