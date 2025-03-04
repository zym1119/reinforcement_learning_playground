import gym

env = gym.make('CartPole-v1', render_mode='human')

state, _ = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()
env.close()
