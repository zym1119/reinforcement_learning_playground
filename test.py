from utils import load_config, create_inferer


def main():
    config = load_config()
    inferer = create_inferer(config)
    inferer.run()


if __name__ == '__main__':
    main()
