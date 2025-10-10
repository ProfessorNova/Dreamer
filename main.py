from tensorboardX import SummaryWriter

from lib.config import Config
from lib.trainer import train


def main():
    cfg = Config()

    summary_writer = SummaryWriter(logdir=cfg.run_dir) if cfg.create_artifacts else None

    train(cfg, summary_writer=summary_writer)


if __name__ == "__main__":
    main()
