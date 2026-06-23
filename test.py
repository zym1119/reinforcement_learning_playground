import argparse

from utils import load_config, create_inferer


def parse_args():
    parser = argparse.ArgumentParser(description='RL Evaluation')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='path to config yaml file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='path to checkpoint file')
    parser.add_argument('--overrides', nargs='+', default=[],
                        help='key=value pairs')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, overrides=args.overrides)
    config['ckpt_path'] = args.ckpt
    inferer = create_inferer(config)
    inferer.run()


if __name__ == '__main__':
    main()
