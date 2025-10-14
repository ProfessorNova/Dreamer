"""
Training loop for DreamerV3.

This module provides a high‑level function to train a DreamerV3 agent on
a Gymnasium environment.  It brings together the world model, actor,
critic and replay buffer and performs data collection, model training,
and logging.  The training loop roughly follows the procedure outlined
in the DreamerV3 paper: collect experience, train the world model on
observed sequences, imagine trajectories using the world model to train
the actor and critic, and periodically log metrics and videos【145968576409203†L438-L548】.

Note: Running this code requires PyTorch and Gymnasium installed with
support for the desired environment.  At runtime, adjust hyperparameters
such as learning rates, batch sizes and horizons as needed.  Logging
uses TensorBoard through torch.utils.tensorboard.SummaryWriter.
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.actor import Actor
from lib.critic import Critic
from lib.replay_buffer import ReplayBuffer
from lib.utils import log_video, make_env
from lib.world_model import WorldModel


def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


def train(
        env_name: str,
        num_steps: int = 1_000_000,
        buffer_size: int = 1000,
        batch_size: int = 16,
        seq_len: int = 50,
        imagine_horizon: int = 15,
        train_every: int = 50,
        initial_random_steps: int = 1000,
        log_interval: int = 5000,
        eval_interval: int = 10_000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "runs/dreamerv3",
) -> None:
    """Train a DreamerV3 agent on the specified environment.

    Args:
        env_name: Name of the Gymnasium environment to train on.
        num_steps: Total number of environment steps to collect.
        buffer_size: Maximum number of episodes to keep in replay buffer.
        batch_size: Number of sequences per optimization step.
        seq_len: Length of sequences sampled for world model training.
        imagine_horizon: Imagination horizon for actor/critic training.
        train_every: Number of environment steps between updates.
        initial_random_steps: Number of random actions before using policy.
        log_interval: Steps between logging training metrics.
        eval_interval: Steps between evaluation episodes logged as videos.
        device: Device to run models on ("cuda" or "cpu").
        log_dir: Directory for TensorBoard logs.
    """
    # Create environment
    env = make_env(env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete), "Only discrete actions are supported"

    # Determine observation shape
    obs_example, _ = env.reset()
    if isinstance(obs_space, gym.spaces.Box):
        obs_shape = obs_example.shape  # includes image dims
    elif isinstance(obs_space, gym.spaces.Dict):
        # Flatten dict observations (e.g., {'image': ..., 'state': ...})
        raise NotImplementedError("Dict observation spaces not yet supported")
    else:
        obs_shape = (obs_space.shape[0],)

    action_size = act_space.n
    # Create models
    world_model = WorldModel(obs_shape, action_size).to(device)
    actor = Actor(world_model.deter_size + world_model.stoch_size, action_size).to(device)
    critic = Critic(world_model.deter_size + world_model.stoch_size).to(device)

    # Optimizers
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    buffer = ReplayBuffer(max_size=buffer_size, seq_len=seq_len)
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize environment and episode storage
    obs, _ = env.reset()
    episode_obs = [obs]
    episode_actions: list = []
    episode_rewards: list = []
    episode_continues: list = []
    state = world_model.init_state(batch_size=1, device=device)
    total_steps = 0
    episode = 0
    start_time = time.time()

    while total_steps < num_steps:
        # Select action
        if total_steps < initial_random_steps:
            action = env.action_space.sample()
            log_prob = None
        else:
            # Obtain feature from current model state
            feat = world_model.get_feat({k: v for k, v in state.items()})
            dist = actor(feat)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor([action], device=device))
        next_obs, reward, done, truncated, _ = env.step(action)
        continue_flag = 0.0 if (done or truncated) else 1.0
        # Append to episode storage
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_continues.append(continue_flag)
        episode_obs.append(next_obs)
        total_steps += 1
        # If episode finished, add to replay buffer and reset
        if done or truncated:
            buffer.add_episode(
                obs=np.stack(episode_obs, axis=0),
                actions=np.array(episode_actions, dtype=np.int64),
                rewards=np.array(episode_rewards, dtype=np.float32),
                continues=np.array(episode_continues, dtype=np.float32),
            )
            obs, _ = env.reset()
            episode_obs = [obs]
            episode_actions = []
            episode_rewards = []
            episode_continues = []
            state = world_model.init_state(batch_size=1, device=device)
            episode += 1
            continue
        else:
            obs = next_obs
        # Periodically train models
        if total_steps >= initial_random_steps and total_steps % train_every == 0 and len(buffer.buffer) > 0:
            batch = buffer.sample_batch(batch_size)
            # Prepare tensors
            obs_batch = to_torch(batch["obs"], device)  # (B, L+1, *obs_shape)
            actions_batch = to_torch(batch["actions"], device)  # (B, L)
            rewards_batch = to_torch(batch["rewards"], device)  # (B, L)
            continues_batch = to_torch(batch["continues"], device)  # (B, L)
            # Convert actions to one‑hot vectors for the world model
            B, L = actions_batch.shape
            one_hot_actions = F.one_hot(actions_batch.long(), num_classes=action_size).float()
            # World model training
            wm_opt.zero_grad()
            prev_state = world_model.init_state(B, device=device)
            # Use obs_batch[:, :-1] as inputs and obs_batch[:, 1:] as targets for reconstruction
            wm_loss, post, prior = world_model.loss(
                obs_batch[:, :-1],
                one_hot_actions,
                rewards_batch,
                continues_batch,
                prev_state,
            )
            wm_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            wm_opt.step()
            # Actor and critic training using imagined trajectories
            with torch.no_grad():
                # Starting from posterior states (last state of sequence)
                last_post = {k: v[:, -1].detach() for k, v in post[-1].items() if k in ["deter", "stoch"]}
            # Imagine horizon
            imag_actions = []
            imag_states = []
            # We'll sample from actor while building actions
            state_imag = last_post
            for t in range(imagine_horizon):
                feat = world_model.get_feat(state_imag)
                dist = actor(feat)
                act = dist.sample()
                imag_actions.append(act)
                # Create one‑hot for the RSSM
                act_oh = F.one_hot(act, num_classes=action_size).float().unsqueeze(1)  # (B, 1, action_size)
                # Predict next prior state
                priors = world_model.rssm.imagine(act_oh, {k: v.clone() for k, v in state_imag.items()})
                next_state = {
                    "deter": priors[0]["deter"],
                    "stoch": priors[0]["stoch"],
                }
                imag_states.append(next_state)
                state_imag = next_state
            # Stack imagined actions and states
            imag_actions = torch.stack(imag_actions, dim=1)
            # Compute imagined features and rewards/pcontinue via predictors
            imag_feats = torch.stack([world_model.get_feat(s) for s in imag_states], dim=1)
            imag_rewards = torch.stack(
                [world_model.reward_predictor(s["deter"], s["stoch"]) for s in imag_states], dim=1
            )
            imag_continues = torch.stack(
                [torch.sigmoid(world_model.continue_predictor(s["deter"], s["stoch"])) for s in imag_states], dim=1
            )
            # Compute λ‑returns using critic's target network
            with torch.no_grad():
                values = critic.value(imag_feats.view(-1, imag_feats.size(-1)), use_target=True).view(batch_size,
                                                                                                      imagine_horizon)
            # Bootstrapped lambda return
            # We'll use lambda=0.95 and discount=0.997 as default
            lam = 0.95
            discount = 0.997
            returns = torch.zeros_like(imag_rewards)
            next_value = values[:, -1]
            for t in reversed(range(imagine_horizon)):
                r = imag_rewards[:, t]
                c = imag_continues[:, t]
                v = values[:, t]
                next_value = r + discount * (c * ((1 - lam) * v + lam * next_value))
                returns[:, t] = next_value
            # Update actor
            actor_opt.zero_grad()
            # flatten imag_feats and imag_actions for actor loss
            actor_loss = actor.loss(
                imag_feats.detach(), imag_actions.detach(), returns.detach(), values.detach()
            )
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            actor_opt.step()
            # Update critic
            critic_opt.zero_grad()
            critic_loss = critic.loss(imag_feats.detach(), returns.detach())
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            critic_opt.step()
            critic.update_target()

        # Logging
        if total_steps % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            writer.add_scalar("perf/steps_per_sec", steps_per_sec, total_steps)
            writer.add_scalar("train/world_model_loss", wm_loss.item(), total_steps)
            writer.add_scalar("train/actor_loss", actor_loss.item(), total_steps)
            writer.add_scalar("train/critic_loss", critic_loss.item(), total_steps)
        # Evaluation
        if total_steps % eval_interval == 0:
            # Run a short policy rollout without learning and log video
            eval_frames = []
            eval_env = gym.make(env_name)
            eval_obs, _ = eval_env.reset()
            eval_state = world_model.init_state(batch_size=1, device=device)
            done_eval = False
            t_eval = 0
            while not done_eval and t_eval < 1000:
                eval_frames.append(eval_obs)
                feat = world_model.get_feat(eval_state)
                dist = actor(feat)
                eval_action = dist.sample().item()
                eval_obs, _, done_eval, truncated_eval, _ = eval_env.step(eval_action)
                if done_eval or truncated_eval:
                    break
                t_eval += 1
            # Log video
            log_video(writer, f"eval/video", eval_frames, total_steps)
            eval_env.close()
    env.close()
    writer.close()
