import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils import load_config, create_agent


@pytest.fixture
def dqn_config():
    config = load_config('configs/dqn_cartpole.yaml')
    config['total_steps'] = 100
    config['log_interval'] = 100
    config['eval_interval'] = 100
    config['save_interval'] = 100
    config['exp_name'] = '_test_dqn'
    return config


class TestDQNAgent:
    def test_create_agent(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        assert agent is not None
        assert agent.config['algorithm'] == 'DQN'

    def test_env_returns_tensor(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        obs, _ = agent.env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == agent.device.type

    def test_collect(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        info = agent.collect()
        assert 'n_steps' in info
        assert info['n_steps'] == 1

    def test_update(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        agent.prefill_replay_buffer()
        agent.collect()
        train_info = agent.update()
        assert 'loss' in train_info
        assert 'epsilon' in train_info

    def test_predict(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=True)
        assert isinstance(action, int)

    def test_replay_buffer_stores_tensors(self, dqn_config):
        agent = create_agent(dqn_config, mode='train')
        agent.prefill_replay_buffer()
        obs, action, reward, next_obs, done = agent.buffer.buffer[0]
        assert isinstance(obs, torch.Tensor)
        assert isinstance(next_obs, torch.Tensor)

    def test_train_loop(self, dqn_config, tmp_path):
        dqn_config['exp_name'] = '_test_dqn_train'
        agent = create_agent(dqn_config, mode='train')
        agent.train()
        # 验证训练完成后步数正确
        assert agent.steps >= dqn_config['total_steps']

    def test_eval_mode(self, dqn_config, tmp_path):
        # 先训练保存 checkpoint
        dqn_config['exp_name'] = '_test_dqn_eval'
        agent = create_agent(dqn_config, mode='train')
        agent.train()
        ckpt_path = agent.run_dir / 'checkpoints' / 'model_last.pth'

        # 用 eval 模式加载
        eval_config = load_config('configs/dqn_cartpole.yaml')
        eval_config['exp_name'] = '_test_dqn_eval_run'
        eval_config['evaluation'] = {'total_episodes': 2}
        eval_config['ckpt_path'] = str(ckpt_path)
        eval_agent = create_agent(eval_config, mode='eval')
        obs, _ = eval_agent.env.reset()
        action = eval_agent.predict(obs, deterministic=True)
        assert isinstance(action, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
