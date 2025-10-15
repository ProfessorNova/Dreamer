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


@torch.no_grad()
def log_episode_video(
        cfg: Config,
        writer: SummaryWriter,
        env: gym.Env,
        world_model,
        actor,
        global_step: int,
) -> None:
    """
    Roll out an eval episode using (world_model + actor), recording env frames.
    The actor samples actions from the current model state (posterior-updated with real obs).
    """
    video_frames = []
    total_reward = 0.0
    steps = 0
    done = False

    obs, _ = env.reset(seed=cfg.seed)
    current_obs = obs

    # Recurrent model state + previous action index
    model_state = world_model.init_state(1, cfg.device)
    last_action_idx = torch.zeros(1, dtype=torch.long, device=cfg.device)

    while not done and steps < cfg.video_max_frames:
        frame = env.render()  # H,W,C (uint8)
        if frame is not None:
            video_frames.append(frame)

        # Update posterior state with the *real* obs and last action
        obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0
        model_state, _ = world_model.step(model_state, a_prev_idx=last_action_idx, x_t=obs_tensor)

        # Sample action from actor
        dist = actor(model_state)
        action_idx = dist.sample().item()

        # Env step
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        total_reward += float(reward)
        done = terminated or truncated

        # Prep next step
        current_obs = next_obs
        last_action_idx.fill_(action_idx)

        steps += 1

    if len(video_frames) == 0:
        return

    # (T,H,W,C) -> (1,T,C,H,W)
    vid = torch.from_numpy(np.stack(video_frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0)
    writer.add_video("episode/video", vid, global_step=global_step, fps=cfg.video_fps)
    writer.add_scalar("episode/total_reward", total_reward, global_step)
    writer.add_scalar("episode/length", steps, global_step)
    print(f"[video] episode: {steps} steps, return {total_reward:.2f}")


# ---------- World-Model visualization helpers ----------

def _stack_lr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """HWC uint8 side-by-side stack."""
    assert left.dtype == np.uint8 and right.dtype == np.uint8
    assert left.shape == right.shape, f"shape mismatch {left.shape} vs {right.shape}"
    return np.concatenate([left, right], axis=1)


@torch.no_grad()
def log_wm_reconstruction_video(
        cfg,
        writer: SummaryWriter,
        env: gym.Env,
        world_model,
        actor,
        global_step: int,
        max_frames: int | None = None,
) -> None:
    """
    Posterior reconstructions vs ground truth:
      - collect a short episode with actor
      - at each step, feed x_t to encoder (posterior) and decode x_hat
      - log side-by-side (GT | Recon)
    """
    if max_frames is None:
        max_frames = cfg.video_max_frames

    # Collect short rollout with actions & ground-truth frames
    obs_seq, act_seq = [], []
    done = False
    steps = 0
    obs, _ = env.reset(seed=cfg.seed)
    current_obs = obs

    model_state = world_model.init_state(1, cfg.device)
    last_action_idx = torch.zeros(1, dtype=torch.long, device=cfg.device)

    while not done and steps < max_frames:
        obs_seq.append(current_obs)  # HWC uint8

        obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0
        model_state, _ = world_model.step(model_state, a_prev_idx=last_action_idx, x_t=obs_tensor)

        dist = actor(model_state)
        a = dist.sample().item()
        act_seq.append(a)

        next_obs, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        current_obs = next_obs
        last_action_idx.fill_(a)
        steps += 1

    if len(obs_seq) == 0:
        return

    # Re-run a posterior pass to produce reconstructions aligned with GT
    frames = []
    model_state = world_model.init_state(1, cfg.device)
    last_action_idx.zero_()
    for t in range(len(obs_seq)):
        x_t = torch.as_tensor(obs_seq[t], dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0

        model_state, info = world_model.step(model_state, a_prev_idx=last_action_idx, x_t=x_t)
        # x_hat is in symlog space in your training; map to pixel space
        x_recon = symexp(info["x_hat"]).clamp(0.0, 1.0)  # (B,C,H,W)
        recon = (x_recon[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        gt = np.transpose(obs_seq[t], (1, 2, 0))  # to HWC
        frames.append(_stack_lr(gt, recon))
        last_action_idx.fill_(act_seq[t])

    vid = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0)
    writer.add_video("wm/reconstruction_vs_gt", vid, global_step=global_step, fps=cfg.video_fps)
    print(f"[video] wm reconstruction vs gt: {len(frames)} frames")


@torch.no_grad()
def log_wm_imagination_video(
        cfg,
        writer: SummaryWriter,
        env: gym.Env,
        world_model,
        actor,
        global_step: int,
        max_frames: int | None = None,
) -> None:
    """
    Prior imagination vs ground truth:
      - collect a short episode (GT obs + action indices)
      - start from posterior at t=0 to align the state
      - from t>=1, run *prior* (x_t=None), using the recorded actions,
        and decode imagined frames to compare with GT.
      - log side-by-side (GT | Imagine)
    """
    if max_frames is None:
        max_frames = cfg.video_max_frames

    # Collect GT sequence + actions with actor
    obs_seq, act_seq = [], []
    done = False
    steps = 0
    obs, _ = env.reset(seed=cfg.seed)
    current_obs = obs

    model_state = world_model.init_state(1, cfg.device)
    last_action_idx = torch.zeros(1, dtype=torch.long, device=cfg.device)

    while not done and steps < max_frames:
        obs_seq.append(current_obs)

        obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0
        model_state, _ = world_model.step(model_state, a_prev_idx=last_action_idx, x_t=obs_tensor)

        dist = actor(model_state)
        a = dist.sample().item()
        act_seq.append(a)

        next_obs, _, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        current_obs = next_obs
        last_action_idx.fill_(a)
        steps += 1

    if len(obs_seq) < 2:
        return

    # Imagine using the recorded actions (align with GT)
    frames = []
    model_state = world_model.init_state(1, cfg.device)
    last_action_idx.zero_()

    # t = 0: posterior update for alignment (decode too for completeness)
    x0 = torch.as_tensor(obs_seq[0], dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0
    model_state, info0 = world_model.step(model_state, a_prev_idx=last_action_idx, x_t=x0)
    xhat0 = symexp(info0["x_hat"]).clamp(0.0, 1.0)
    img0 = (xhat0[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    gt = np.transpose(obs_seq[0], (1, 2, 0))  # to HWC
    frames.append(_stack_lr(gt, img0))
    last_action_idx.fill_(act_seq[0])

    # t >= 1: prior imagination using recorded actions
    for t in range(1, len(obs_seq)):
        # prior step: no x_t
        model_state, info = world_model.step(model_state, a_prev_idx=torch.tensor([act_seq[t - 1]], device=cfg.device),
                                             x_t=None)
        x_im = symexp(info["x_hat"]).clamp(0.0, 1.0)
        im = (x_im[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        gt = np.transpose(obs_seq[t], (1, 2, 0))  # to HWC
        frames.append(_stack_lr(gt, im))

    vid = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0)
    writer.add_video("wm/imagination_vs_gt", vid, global_step=global_step, fps=cfg.video_fps)
    print(f"[video] wm imagination vs gt: {len(frames)} frames")
