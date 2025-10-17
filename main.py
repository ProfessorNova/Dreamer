import os

import ale_py
import torch
from gymnasium import register_envs
from tensorboardX import SummaryWriter

from lib.config import Config
from lib.trainer import train

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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
