import argparse

import gym

from utils import prepare_run, get_trainer


def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning train script')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--model', type=str, default='PolicyGradient',
                        choices=['PolicyGradient', 'DQN'], help='model type')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--steps', type=int, default=1000,
                        help='training steps')
    parser.add_argument('--log-interval', type=int,
                        default=10, help='log interval')
    parser.add_argument('--save-interval', type=int,
                        default=200, help='save interval')
    parser.add_argument('--max-episode-steps', type=int,
                        default=1000, help='maximum episode steps')
    parser.add_argument('--run-name', '-n', type=str,
                        help='experiment run name')
    parser.add_argument('--root-dir', '-r', type=str,
                        default='./runs', help='root folder for saving runs')
    parser.add_argument('--model-configs', nargs='+',
                        help='additional model configs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_dir, logger = prepare_run(
        run_name=args.run_name, root_dir=args.root_dir)

    # set up env & model & optimzier
    env = gym.make(args.env)
    trainer = get_trainer(args.model, env, run_dir, lr=args.lr, log_interval=args.log_interval,
                          save_interval=args.save_interval, max_episode_steps=args.max_episode_steps, steps=args.steps)

    # train
    trainer.train()
