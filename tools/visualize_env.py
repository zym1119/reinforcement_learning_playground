"""
可视化 Gymnasium 环境：录制所有 episode 到单个视频文件。

Usage:
    python tools/visualize_env.py --env CartPole-v1
    python tools/visualize_env.py --env LunarLander-v3 --steps 500
"""
import argparse
import os

import gymnasium as gym
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize a Gymnasium environment')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Environment ID (e.g. CartPole-v1, LunarLander-v3)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Total steps to run')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS')
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join('./videos', args.env.replace('/', '-'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'replay.mp4')

    env = gym.make(args.env, render_mode='rgb_array')

    obs, _ = env.reset()
    frames = [env.render()]
    total_reward = 0.0
    episode = 1
    episode_rewards = []

    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        total_reward += reward

        if terminated or truncated:
            print(f'Episode {episode} ended at step {step + 1}, reward: {total_reward:.2f}')
            episode_rewards.append(total_reward)
            obs, _ = env.reset()
            total_reward = 0.0
            episode += 1

    env.close()

    # 统计
    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print(f'\nAverage reward over {len(episode_rewards)} episodes: {avg_reward:.2f}')

    # 保存为单个 mp4
    import imageio
    imageio.mimsave(output_path, frames, fps=args.fps)
    print(f'Done. {len(episode_rewards)} episodes, {len(frames)} frames saved to: {os.path.abspath(output_path)}')


if __name__ == '__main__':
    main()
