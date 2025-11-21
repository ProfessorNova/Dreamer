import datetime
import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- Environment and device settings ---
    env_id: str = "ALE/Pong-v5"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    # General
    num_iterations: int = 1_000_000
    replay_capacity: int = 100_000
    batch_size: int = 16
    batch_length: int = 64

    # --- Logging and checkpointing ---
    create_artifacts: bool = False
    run_dir: str = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir: str = os.path.join(run_dir, "checkpoints")
    log_interval: int = 10
    save_model_interval: int = 1000
    video_interval: int = 500
    video_fps: int = 20
    video_max_frames: int = 500

    # --- Random seed for reproducibility ---
    seed: int = 42
