import os

import ale_py
from gymnasium import register_envs
from torch.utils.tensorboard import SummaryWriter

from lib.config import Config
from lib.trainer import train


def main():
    cfg = Config()

    register_envs(ale_py)
    summary_writer = None
    if cfg.create_artifacts:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=cfg.run_dir)

    train(cfg, summary_writer=summary_writer)


if __name__ == "__main__":
    main()
