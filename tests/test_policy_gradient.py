import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from utils import load_config, create_agent


@pytest.fixture
def pg_config():
    config = load_config('configs/pg_cartpole.yaml')
    config['total_episodes'] = 5
    config['log_interval'] = 5
    config['eval_interval'] = 5
    config['save_interval'] = 5
    config['exp_name'] = '_test_pg'
    return config


class TestPGAgent:
    def test_create_agent(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        assert agent is not None
        assert agent.config['algorithm'] == 'PolicyGradient'

    def test_env_returns_tensor(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        obs, _ = agent.env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.device.type == agent.device.type

    def test_collect_full_episode(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        info = agent.collect()
        assert info['n_episodes'] == 1
        assert info['n_steps'] > 0
        assert 'episode_reward' in info

    def test_update(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        agent.collect()
        train_info = agent.update()
        assert 'loss' in train_info

    def test_predict_deterministic(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=True)
        assert isinstance(action, int)
        # 确定性策略应该每次返回相同结果
        action2 = agent.predict(obs, deterministic=True)
        assert action == action2

    def test_predict_stochastic(self, pg_config):
        agent = create_agent(pg_config, mode='train')
        obs, _ = agent.env.reset()
        action = agent.predict(obs, deterministic=False)
        assert isinstance(action, int)

    def test_train_loop(self, pg_config):
        pg_config['exp_name'] = '_test_pg_train'
        agent = create_agent(pg_config, mode='train')
        agent.train()
        assert agent.episodes >= pg_config['total_episodes']

    def test_eval_mode(self, pg_config):
        pg_config['exp_name'] = '_test_pg_eval'
        agent = create_agent(pg_config, mode='train')
        agent.train()
        ckpt_path = agent.run_dir / 'checkpoints' / 'model_last.pth'

        eval_config = load_config('configs/pg_cartpole.yaml')
        eval_config['exp_name'] = '_test_pg_eval_run'
        eval_config['evaluation'] = {'total_episodes': 2}
        eval_config['ckpt_path'] = str(ckpt_path)
        eval_agent = create_agent(eval_config, mode='eval')
        obs, _ = eval_agent.env.reset()
        action = eval_agent.predict(obs, deterministic=True)
        assert isinstance(action, int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
