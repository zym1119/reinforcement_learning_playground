# Reinforcement Learning Playground

A modular RL framework for experimenting with various reinforcement learning algorithms.

## Project Structure

```
├── agents/
│   ├── base.py              # BaseAgent: unified train & eval base class
│   ├── policy_gradient.py   # REINFORCE (PGAgent)
│   ├── dqn.py               # DQN / Double DQN
│   ├── actor_critic.py      # Actor-Critic (A2C)
│   ├── ppo_atari.py         # PPO for Atari image observations
│   └── soft_actor_critic.py # SAC (Soft Actor-Critic)
├── blocks/
│   ├── mlp.py               # MLP builder with orthogonal init
│   ├── distributions.py     # CategoricalDist, GaussianDist
│   ├── replay_buffer.py     # ReplayBuffer for off-policy methods
│   └── rollout_buffer.py    # RolloutBuffer for on-policy methods
├── configs/
│   ├── _base_/              # Base configs (default.yaml, pg.yaml, dqn.yaml, sac.yaml, etc.)
│   ├── pg_cartpole.yaml     # CartPole + PolicyGradient
│   ├── dqn_cartpole.yaml    # CartPole + DQN
│   ├── ac_cartpole.yaml     # CartPole + Actor-Critic
│   ├── ppo_breakout.yaml    # Breakout + PPOAtari
│   └── sac_walker2d.yaml    # Walker2d + SAC
├── env_utils/               # Env construction, Atari wrappers, normalization wrappers
├── tools/
│   └── visualize_env.py     # Record env video with random actions
├── train.py                 # Training entry point
├── test.py                  # Evaluation entry point
└── utils.py                 # Config, Logger, Registry, Device/Seed
```

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies for video recording:
```bash
pip install opencv-python imageio[ffmpeg]
```

## Usage

### Training

```bash
python train.py --config configs/pg_cartpole.yaml --exp-name my_experiment
```

Atari PPO example:

```bash
python train.py --config configs/ppo_breakout.yaml --exp-name ppo_breakout
```

Override config values from command line:
```bash
python train.py -c configs/pg_cartpole.yaml --overrides lr=1e-4 seed=123
```

### Evaluation

```bash
python test.py --config configs/pg_cartpole.yaml --ckpt work_dirs/my_experiment/checkpoints/model_best.pth --exp-name eval_test
```

## Config System

YAML configs with `_base_` inheritance (similar to mmdetection):

```yaml
# configs/pg_cartpole.yaml
_base_: _base_/pg.yaml

env: CartPole-v1
total_episodes: 1000
lr: 1e-3
```

Features:
- Hierarchical inheritance via `_base_`
- LR scheduler config (StepLR / ExponentialLR / CosineAnnealingLR / LinearLR)
- Evaluation config (episode/step-based, video recording)
- CLI overrides with `--overrides key=value`

## Atari PPO Notes

`PPOAtari` follows the common Nature CNN / SB3 / CleanRL Atari setup: 84x84 grayscale frame stacks, reward clipping for training, orthogonal initialization, clipped PPO updates, optional advantage normalization, approximate KL logging, and linear learning-rate decay by update count.

Training uses `EpisodicLifeEnv` by default so losing a life produces an on-policy episode boundary. The Atari wrapper records true game-level episode statistics before that life-loss wrapper, so `train/episode_reward` reflects full-game returns instead of per-life returns when Gymnasium reports completed episodes.

Evaluation environments disable `terminal_on_life_loss` and reward clipping to report full-game scores. For games that require `FIRE`, `FireOnLifeLoss` keeps evaluation episodes moving after a life loss without splitting the game into per-life episodes.

## Implement Your Own Algorithm

1. Create `agents/your_algo.py`
2. Subclass `BaseAgent` and implement 4 methods:

```python
from agents.base import BaseAgent
from utils import AGENT

@AGENT.register('YourAlgo')
class YourAgent(BaseAgent):

    def init_model(self):
        """Init networks and optimizer"""
        ...

    def collect(self) -> dict:
        """Interact with env, return {'n_steps': ..., 'episode_reward': ...}"""
        ...

    def update(self) -> dict:
        """Gradient update, return {'loss': ...}"""
        ...

    def predict(self, obs, deterministic=False):
        """Select action from observation"""
        ...
```

3. Register import in `agents/__init__.py`
4. Create a config YAML with `algorithm: YourAlgo`

## Algorithms

| Algorithm | Type | Status |
|-----------|------|--------|
| REINFORCE (Policy Gradient) | On-policy | ✅ |
| DQN / Double DQN | Off-policy | ✅ |
| Actor-Critic (A2C) | On-policy | ✅ |
| SAC (Soft Actor-Critic) | Off-policy | ✅ |
| PPO Atari | On-policy | ✅ |