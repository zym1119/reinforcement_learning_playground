import argparse

from utils import load_config, create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description='RL Training')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='path to config yaml file')
    parser.add_argument('--exp-name', '-n', type=str, default=None,
                        help='experiment name (work_dirs/{exp_name})')
    parser.add_argument('--overrides', nargs='+', default=[],
                        help='key=value pairs, e.g. lr=1e-4 seed=123')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.overrides, exp_name=args.exp_name)
    trainer = create_trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
