# reinforcement_learning_playground
My playground for learning RL

# Environment
Install `gym` python package.
```bash
pip install gym
pip install pygame
```

Try this demo code to check `gym` installation.
```python
import gym

env = gym.make('CartPole-v1', render_mode='human')

state = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)
    if done:
        state = env.reset()
env.close()
```


# Usage

train:
```bash
python train.py --model {model_type} --lr {lr} --episodes {num of episodes}
```

test:
```bash
python test.py --model {model_type} -c {ckpt path}
```

# Implement your own method

1. Define you model in `models` folder
2. Implement your own Trainer and Inferer based on `BaseTrainer` or `BaseInferer` defined in `runner.py`