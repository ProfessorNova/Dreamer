import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn.functional import softmax

from lib.actor import Actor
from lib.config import Config
from lib.world_model import WorldModel


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def unimix_logits(logits: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Mix eps of uniform into softmax(logits) and return log-probs."""
    probs = softmax(logits, dim=-1)
    k = probs.size(-1)
    mixed = (1.0 - eps) * probs + eps / k
    return torch.log(mixed.clamp_min(1e-12))


def make_env(cfg: Config, eval_env: bool = False) -> gym.Env:
    if eval_env:
        env = gym.make(cfg.env_id, render_mode='rgb_array', frameskip=cfg.frame_skip)
    else:
        env = gym.make(cfg.env_id, frameskip=cfg.frame_skip)
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = ImageToPyTorch(env)
    return env


@torch.inference_mode()
def log_episode_video(
        cfg: Config,
        summary_writer: SummaryWriter,
        env: gym.Env,
        world_model: WorldModel,
        actor: Actor,
        global_step: int,
) -> None:
    """
    Roll out a short evaluation episode using the current world model + actor,
    record RGB frames, and write a video to TensorBoard.
    """
    # store if the models are in training mode
    was_training_wm = world_model.training
    was_training_actor = actor.training
    world_model.eval()
    actor.eval()

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
        embed = world_model.encoder(obs_tensor)
        belief = model_state['deter']
        _, _, stoch_prior, belief = world_model.rssm.prior(model_state["stoch"], belief, last_action)
        _, _, stoch_post = world_model.rssm.posterior(belief, embed)
        model_state = {"deter": belief, "stoch": stoch_post}
        feat = world_model.get_feat(model_state)
        dist = actor(feat)
        action_idx = dist.probs.argmax(dim=-1).item()

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
    summary_writer.add_video(
        tag="episode_video",
        vid_tensor=torch.from_numpy(video_frames).permute(0, 3, 1, 2).unsqueeze(0),
        global_step=global_step,
        fps=cfg.video_fps
    )
    summary_writer.add_scalar(
        tag="video/episode_total_reward",
        scalar_value=total_reward,
        global_step=global_step
    )
    summary_writer.add_scalar(
        tag="video/episode_length",
        scalar_value=steps,
        global_step=global_step
    )

    print(f"Logged episode video ({steps} steps, total reward: {total_reward}) at step {global_step}")

    # Restore model training states
    if was_training_wm:
        world_model.train()
    if was_training_actor:
        actor.train()
