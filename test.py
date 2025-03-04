import argparse
from pathlib import Path

import torch
from utils import get_inferer

import gym


def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning test script')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--model', type=str, default='PolicyGradient',
                        choices=['PolicyGradient'], help='model type')
    parser.add_argument('--run-name', '-n', type=str,
                        help='experiment run name')
    parser.add_argument('--root-dir', '-r', type=str,
                        default='./runs', help='root folder for saving runs')
    parser.add_argument('--ckpt', '-c', type=str,
                        help='checkpoint path')
    parser.add_argument('--steps', type=int, default=1000,
                        help='test steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    env = gym.make(args.env, render_mode='human')

    # find ckpt
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        assert args.run_name is not None
        ckpt_path = Path(args.root_dir) / f'{args.run_name}/model_last.pth'

    inferer = get_inferer(args.model, env, ckpt_path)
    inferer.infer()
