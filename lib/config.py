import datetime
import os
from dataclasses import dataclass

import ale_py
import torch
from gymnasium import register_envs


@dataclass
class Config:
    # Environment and device settings
    env_id: str = "ALE/Breakout-v5"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    total_timesteps: int = 1_000

    # Logging and checkpointing
    create_artifacts: bool = False
    run_dir: str = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir: str = os.path.join(run_dir, "checkpoints")
    log_interval: int = 1_000
    video_fps: int = 30
    video_max_frames: int = 1_000

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        """
        Ensures that arcade environments are properly registered.
        Ensures that necessary directories are created if create_artifacts is True.
        :return: None
        """
        register_envs(ale_py)
        if self.create_artifacts:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
