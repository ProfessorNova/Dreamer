from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.config import Config


def make_env(env_id: str = None) -> gym.Env:
    env = gym.make(env_id, render_mode='rgb_array')
    return env


def log_episode_video(
        cfg: Config,
        summary_writer: SummaryWriter,
        env: gym.Env,
        model: Any,
        global_step: int,
) -> None:
    video_frames = []
    done = False
    obs, _ = env.reset(seed=cfg.seed)
    num_steps = 0
    while not done:
        frame = env.render()
        video_frames.append(frame)

        # TODO: Use model to select action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        num_steps += 1
        if num_steps >= cfg.video_max_frames:
            break

    # Log video to TensorBoard
    summary_writer.add_video(
        tag='episode_video',
        vid_tensor=torch.tensor(np.array(video_frames)).permute(0, 3, 1, 2).unsqueeze(0),
        global_step=global_step,
        fps=cfg.video_fps
    )
