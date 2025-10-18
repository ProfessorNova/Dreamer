import datetime
import os
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- Environment and device settings ---
    env_id: str = "ALE/BeamRider-v5"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Hyperparameters ---
    # General
    num_iterations: int = 100_000
    replay_capacity: int = 100_000
    batch_size: int = 16
    batch_length: int = 64
    train_ratio: int = 1024

    # World Model
    num_latents = 64
    classes_per_latent = 64
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    unimix_eps: float = 0.01
    free_bits: float = 1.0
    world_model_lr: float = 3e-4
    world_model_adam_eps: float = 1e-8
    world_model_grad_clip: float = 1000.0

    # Actor Critic
    imagination_horizon: int = 15
    gamma: float = 0.997
    lam: float = 0.95
    critic_num_buckets: int = 255
    critic_bucket_min: float = -20.0
    critic_bucket_max: float = 20.0
    critic_ema_decay: float = 0.98
    critic_ema_regularizer: float = 1.0
    return_normalization_limit: float = 1.0
    return_normalization_decay: float = 0.99
    actor_entropy_scale: float = 3e-4
    actor_ret_norm_limit: float = 1.0
    actor_ret_norm_decay: float = 0.99
    actor_critic_lr: float = 3e-5
    actor_critic_adam_eps: float = 1e-5
    actor_critic_grad_clip: float = 100.0

    # Network sizes
    hidden_size: int = 1024
    base_cnn_channels: int = 64
    mlp_hidden_units: int = 1024

    # --- Logging and checkpointing ---
    create_artifacts: bool = True
    run_dir: str = os.path.join("runs", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_dir: str = os.path.join(run_dir, "checkpoints")
    log_interval: int = 10
    save_interval: int = 1000
    video_interval: int = 500
    video_fps: int = 20
    video_max_frames: int = 500

    # --- Random seed for reproducibility ---
    seed: int = 42
