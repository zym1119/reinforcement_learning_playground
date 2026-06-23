from utils import load_config, create_trainer


def main():
    config = load_config()
    trainer = create_trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
