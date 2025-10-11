import datetime
import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- Environment and device settings ---
    env_id: str = "ALE/Breakout-v5"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_repeat: int = 1

    # --- Training parameters ---
    num_iterations: int = 1_000_000
    train_ratio: float = 32.0
    warmup_steps: int = 2000
    max_credit: int = 2048
    max_steps_per_iter: int = 64
    batch_size: int = 16

    # optimizer
    world_model_lr: float = 1e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4

    # buffer sizes
    buffer_capacity: int = 200_000
    seq_len: int = 50

    # dreamer specific
    free_nats: float = 1.0
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    num_bins: int = 255
    ema_decay: float = 0.995
    imagination_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95

    # network sizes
    embed_size: int = 1024
    deter_size: int = 200
    stoch_size: int = 30
    mlp_units: int = 256
    mlp_depth: int = 16
    entropy_scale: float = 1e-3

    # --- Logging and checkpointing ---
    create_artifacts: bool = True
    run_dir: str = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir: str = os.path.join(run_dir, "checkpoints")
    log_interval: int = 10
    video_interval: int = 100
    video_fps: int = 30
    video_max_frames: int = 1000

    # --- Random seed for reproducibility ---
    seed: int = 42
