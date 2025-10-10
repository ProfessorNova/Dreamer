import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.actor import Actor
from lib.config import Config
from lib.world_model import WorldModel


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Wrapper to change image format from HWC (Height, Width, Channels) to CHW (Channels, Height, Width).

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def observation(self, observation):
        """
        Convert observation from HWC to CHW format.

        Args:
            observation: The original observation in HWC format.

        Returns:
            The observation in CHW format.
        """
        return np.transpose(observation, axes=(2, 0, 1))


def make_env(env_id: str = None) -> gym.Env:
    env = gym.make(env_id, render_mode='rgb_array')
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = ImageToPyTorch(env)
    return env


def log_episode_video(
        cfg: Config,
        summary_writer: SummaryWriter,
        env: gym.Env,
        world_model: WorldModel,
        actor: Actor,
        global_step: int,
) -> None:
    video_frames = []
    done = False
    obs, _ = env.reset(seed=cfg.seed)
    num_steps = 0
    while not done and num_steps < cfg.video_max_frames:
        frame = env.render()
        video_frames.append(frame)

        feat = world_model.get_feat(obs)
        dist = actor(feat)
        action = dist.sample().item()
        obs, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        num_steps += 1

    # Log video to TensorBoard
    summary_writer.add_video(
        tag='episode_video',
        vid_tensor=torch.tensor(np.array(video_frames)).permute(0, 3, 1, 2).unsqueeze(0),
        global_step=global_step,
        fps=cfg.video_fps
    )
