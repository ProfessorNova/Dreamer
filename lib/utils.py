from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from lib.config import Config


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def make_env(env_id) -> gym.Env:
    env = gym.make(env_id, render_mode='rgb_array')
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = ImageToPyTorch(env)
    return env


@torch.inference_mode()
def log_episode_video(
        cfg: Config,
        writer: SummaryWriter,
        env: gym.Env,
        world_model: Any,
        actor: Any,
        global_step: int,
) -> None:
    """
    Roll out a short evaluation episode using the current world model + actor,
    record RGB frames, and write a video to TensorBoard.
    """
    video_frames = []
    done = False
    total_reward = 0.0
    obs, _ = env.reset(seed=cfg.seed)
    current_obs = obs

    # Initialize model state and previous action (one-hot)
    model_state = world_model.init_state(1, cfg.device)
    last_action = torch.zeros(1, env.action_space.n, device=cfg.device)

    steps = 0
    while not done and steps < cfg.video_max_frames:
        video_frames.append(env.render())

        # Get action from the actor
        obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        # TODO: Sample action from actor instead of random
        action_idx = env.action_space.sample()

        # Step the environment
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        total_reward += reward
        done = terminated or truncated

        # Prepare for next step
        action_onehot = np.zeros(env.action_space.n, dtype=np.float32)
        action_onehot[action_idx] = 1.0
        current_obs = next_obs
        last_action = torch.as_tensor(action_onehot, device=cfg.device).unsqueeze(0)

        steps += 1

    # Save video
    video_frames = np.stack(video_frames, axis=0)
    writer.add_video(
        tag="episode_video",
        vid_tensor=torch.from_numpy(video_frames).permute(0, 3, 1, 2).unsqueeze(0),
        global_step=global_step,
        fps=cfg.video_fps
    )
    writer.add_scalar(
        tag="video/episode_total_reward",
        scalar_value=total_reward,
        global_step=global_step
    )
    writer.add_scalar(
        tag="video/episode_length",
        scalar_value=steps,
        global_step=global_step
    )

    print(f"Logged episode video ({steps} steps, total reward: {total_reward}) at step {global_step}")
