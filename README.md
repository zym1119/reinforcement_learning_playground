# Reinforcement Learning Playground

A modular RL framework for experimenting with various reinforcement learning algorithms.

## Project Structure

```
├── agents/
│   ├── base.py              # BaseAgent: unified train & eval base class
│   └── policy_gradient.py   # REINFORCE (PGAgent)
├── networks/
│   ├── mlp.py               # MLP builder with orthogonal init
│   └── distributions.py     # CategoricalDist, GaussianDist
├── buffers/
│   └── replay_buffer.py     # ReplayBuffer, RolloutBuffer (with GAE)
├── configs/
│   ├── _base_/              # Base configs (default.yaml, pg.yaml)
│   └── pg_cartpole.yaml     # CartPole + PolicyGradient
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
- LR scheduler config (StepLR / ExponentialLR / CosineAnnealingLR)
- Evaluation config (episode/step-based, video recording)
- CLI overrides with `--overrides key=value`

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
| DQN | Off-policy | Planned |
| PPO | On-policy | Planned |
| A2C | On-policy | Planned |