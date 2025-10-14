import datetime
import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- Environment and device settings ---
    env_id: str = "ALE/Breakout-v5"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_skip: int = 4

    # --- Training parameters ---
    num_iterations: int = 10_000_000
    replay_ratio: float = 64.0
    warmup_steps: int = 20_000
    max_credit: int = 2048
    max_steps_per_iter: int = 64
    batch_size: int = 16

    # optimizer
    world_model_lr: float = 3e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4

    # buffer sizes
    buffer_capacity: int = 1_000_000
    seq_len: int = 64

    # dreamer specific
    free_nats: float = 3.0
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    imagination_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    critic_replay_weight: float = 1.0

    # network architecture
    embed_size: int = 1024
    deter_size: int = 200
    stoch_num_groups = 32
    stoch_num_classes = 32
    mlp_units: int = 512
    mlp_depth: int = 4
    entropy_scale: float = 1e-3
    gumbel_tau: float = 1.0
    unimix_eps: float = 0.01
    ret_norm_decay: float = 0.99
    ret_norm_min_scale: float = 1.0
    num_bins: int = 255
    ema_decay: float = 0.99
    ema_reg: float = 1.0

    # --- Logging and checkpointing ---
    create_artifacts: bool = True
    run_dir: str = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir: str = os.path.join(run_dir, "checkpoints")
    log_interval: int = 1000
    video_interval: int = 10_000
    video_fps: int = 20
    video_max_frames: int = 1000

    # --- Random seed for reproducibility ---
    seed: int = 42
