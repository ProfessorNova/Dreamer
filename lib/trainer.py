import time
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.actor import Actor
from lib.config import Config
from lib.critic import Critic
from lib.replay_buffer import ReplayBuffer
from lib.utils import log_episode_video, make_env
from lib.world_model import WorldModel


def train(cfg: Config, summary_writer=None):
    """
    Dreamer-style training loop with:
      - posterior update before acting (collect phase),
      - imagination from the last posterior state of the WM batch,
      - robust batching and logging.
    """

    # --- Environments ---
    env = make_env(cfg.env_id)
    eval_env = make_env(cfg.env_id)

    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # --- Shapes & Sizes ---
    obs_shape = obs_space.shape
    act_size = act_space.n
    print(f"Observation shape: {obs_shape}, Action size: {act_size}")

    # --- Models ---
    world_model = WorldModel(obs_shape, act_size).to(cfg.device)
    feat_size = world_model.deter_size + world_model.stoch_size  # must match Actor/Critic inputs

    actor = Actor(feat_size, act_size).to(cfg.device)
    critic = Critic(feat_size).to(cfg.device)

    # --- Optimizers ---
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # --- Replay Buffer ---
    buffer = ReplayBuffer(max_size=cfg.buffer_size, seq_len=cfg.seq_len)

    # --- Episode state ---
    obs, _ = env.reset()
    episode_obs = [obs]
    episode_actions: list[int] = []
    episode_rewards: list[float] = []
    episode_continues: list[float] = []

    # RSSM state and previous action (one-hot) for the collect loop
    state: Dict[str, torch.Tensor] = world_model.init_state(batch_size=1, device=cfg.device)
    prev_action_onehot = torch.zeros(1, act_size, device=cfg.device)

    total_steps = 0
    episode = 0
    start_time = time.time()

    # Optional: sanity check to ensure model inputs/outputs align
    assert actor.fc[0].in_features == feat_size, "Actor first layer input must equal feat_size"
    assert critic.feat_size == feat_size, "Critic feature size must equal feat_size"

    while total_steps < cfg.num_steps:
        # ---- Select action ----
        if total_steps < cfg.initial_random_steps:
            # Pure exploration for warm-up
            action = env.action_space.sample()
        else:
            # 1) Posterior update with current observation before acting
            with torch.no_grad():
                # Turn current obs into tensor [1, C, H, W]
                if isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
                    obs_t = torch.as_tensor(obs, dtype=torch.uint8, device=cfg.device)
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device)
                if obs_t.ndim == 3:
                    obs_t = obs_t.unsqueeze(0)

                # GRU update with [z_{t-1}, a_{t-1}]
                gru_in = torch.cat([state["stoch"], prev_action_onehot], dim=-1)  # [1, stoch+act]
                deter = world_model.rssm.gru(gru_in, state["deter"])  # [1, deter]

                # Encode and posterior (q(z_t | h_t, x_t))
                embed = world_model.encoder(obs_t)  # [1, embed]
                mean_post, logstd_post = world_model.rssm._posterior(deter, embed)  # [1, stoch]
                stoch = mean_post + torch.randn_like(mean_post) * logstd_post.exp()

                state = {"deter": deter, "stoch": stoch}

                # 2) Actor takes the proper feature [h_t, z_t]
                feat = world_model.get_feat(state)  # [1, feat_size]
                dist = actor(feat)
                action = dist.sample().item()

            # Update prev_action_onehot for the *next* RSSM update
            prev_action_onehot = F.one_hot(torch.tensor([action], device=cfg.device),
                                           num_classes=act_size).float()

        # ---- Step environment ----
        next_obs, reward, done, truncated, _ = env.step(action)
        continue_flag = 0.0 if (done or truncated) else 1.0

        # ---- Store transition in episode buffer ----
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_continues.append(continue_flag)
        episode_obs.append(next_obs)

        total_steps += 1

        # ---- Episode end handling ----
        if done or truncated:
            # Push full episode to replay
            buffer.add_episode(
                obs=np.stack(episode_obs, axis=0),  # (T+1, C, H, W)
                actions=np.array(episode_actions, dtype=np.int64),  # (T,)
                rewards=np.array(episode_rewards, dtype=np.float32),
                continues=np.array(episode_continues, dtype=np.float32),
            )
            # Reset env and per-episode storage
            obs, _ = env.reset()
            episode_obs = [obs]
            episode_actions = []
            episode_rewards = []
            episode_continues = []

            # Reset RSSM state and prev action at episode boundary
            state = world_model.init_state(batch_size=1, device=cfg.device)
            prev_action_onehot = torch.zeros(1, act_size, device=cfg.device)

            episode += 1
            # continue to next while-iteration
            continue
        else:
            # Continue episode
            obs = next_obs

        # ---- Periodic training ----
        wm_loss = None
        actor_loss = None
        critic_loss = None

        ready_for_train = (
                total_steps >= cfg.initial_random_steps and
                total_steps % cfg.train_every == 0 and
                len(buffer.buffer) > 0
        )
        if ready_for_train:
            batch = buffer.sample_batch(cfg.batch_size)

            # Prepare tensors
            # obs_batch shape: (B, L+1, C, H, W) typically, we train WM on L steps using obs[:-1] as inputs
            obs_batch = torch.as_tensor(batch["obs"], dtype=torch.float32, device=cfg.device)
            actions_batch = torch.as_tensor(batch["actions"], dtype=torch.int64, device=cfg.device)  # (B, L)
            rewards_batch = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=cfg.device)  # (B, L)
            continues_batch = torch.as_tensor(batch["continues"], dtype=torch.float32, device=cfg.device)  # (B, L)

            # One-hot actions for WM
            one_hot_actions = F.one_hot(actions_batch.long(), num_classes=act_size).float()  # (B, L, A)

            # Align lengths if obs_batch includes T+1 frames:
            # Use obs_batch[:, :-1] so we have L frames; then ensure others are also L
            if obs_batch.size(1) == actions_batch.size(1) + 1:
                obs_inputs = obs_batch[:, :-1]  # (B, L, C, H, W)
            else:
                obs_inputs = obs_batch  # already aligned

            # World model training
            wm_opt.zero_grad()
            prev_state = world_model.init_state(obs_inputs.size(0), device=cfg.device)

            wm_loss, post, prior = world_model.loss(
                obs_inputs,  # (B, L, C, H, W)
                one_hot_actions,  # (B, L, A)
                rewards_batch,  # (B, L)
                continues_batch,  # (B, L)
                prev_state,
            )
            wm_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            wm_opt.step()

            # ---- Imagination: start from last posterior state of the WM batch ----
            with torch.no_grad():
                # post is a list[dict] of length L; we take the last dict for all batch items
                last_post = {
                    "deter": post[-1]["deter"].detach(),  # (B, deter)
                    "stoch": post[-1]["stoch"].detach(),  # (B, stoch)
                }

            # Build imagined rollout by sampling actions from the actor
            imag_actions = []
            imag_states = []

            state_imag = {k: v.clone() for k, v in last_post.items()}  # start state (B, *)
            B = state_imag["deter"].size(0)

            for _t in range(cfg.imagination_horizon):
                feat_t = world_model.get_feat(state_imag)  # (B, feat_size)
                dist_t = actor(feat_t)
                act_t = dist_t.sample()  # (B,)
                imag_actions.append(act_t)

                # Convert action to one-hot (B, 1, A)
                act_oh = F.one_hot(act_t, num_classes=act_size).float().unsqueeze(1)
                # Imagine one step ahead
                priors = world_model.rssm.imagine(act_oh, state_imag)  # list of length 1
                state_imag = {
                    "deter": priors[0]["deter"],
                    "stoch": priors[0]["stoch"],
                }
                imag_states.append(state_imag)

            # Stack imagined tensors
            imag_actions = torch.stack(imag_actions, dim=1)  # (B, T)
            imag_feats = torch.stack([world_model.get_feat(s) for s in imag_states], dim=1)  # (B, T, feat)
            imag_rewards = torch.stack(
                [world_model.reward_predictor(s["deter"], s["stoch"]) for s in imag_states], dim=1  # (B, T)
            )
            imag_cont = torch.stack(
                [torch.sigmoid(world_model.continue_predictor(s["deter"], s["stoch"])) for s in imag_states], dim=1
            )  # (B, T)

            # Critic target values (from EMA/target head)
            with torch.no_grad():
                values = critic.value(
                    imag_feats.reshape(-1, imag_feats.size(-1)), use_target=True
                ).view(imag_feats.size(0), imag_feats.size(1))  # (B, T)

            # Bootstrapped lambda-returns
            lam = 0.95
            discount = 0.997
            returns = torch.zeros_like(imag_rewards)  # (B, T)
            next_value = values[:, -1]
            for t in reversed(range(cfg.imagination_horizon)):
                r = imag_rewards[:, t]
                c = imag_cont[:, t]
                v = values[:, t]
                next_value = r + discount * (c * ((1.0 - lam) * v + lam * next_value))
                returns[:, t] = next_value

            # ---- Actor update ----
            actor_opt.zero_grad()
            actor_loss = actor.loss(
                imag_feats.detach(), imag_actions.detach(), returns.detach(), values.detach()
            )
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            actor_opt.step()

            # ---- Critic update ----
            critic_opt.zero_grad()
            critic_loss = critic.loss(imag_feats.detach(), returns.detach())
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            critic_opt.step()
            critic.update_target()

        # ---- Logging ----
        if summary_writer is not None:
            if total_steps % cfg.log_interval == 0 and wm_loss is not None:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / max(1e-6, elapsed)
                summary_writer.add_scalar("perf/steps_per_sec", steps_per_sec, total_steps)
                summary_writer.add_scalar("train/world_model_loss", wm_loss.item(), total_steps)
                if actor_loss is not None:
                    summary_writer.add_scalar("train/actor_loss", actor_loss.item(), total_steps)
                if critic_loss is not None:
                    summary_writer.add_scalar("train/critic_loss", critic_loss.item(), total_steps)

            # Periodic evaluation video
            if total_steps % cfg.video_interval == 0 and total_steps > 0:
                log_episode_video(cfg, summary_writer, eval_env, world_model, actor, total_steps)

    env.close()
    eval_env.close()
