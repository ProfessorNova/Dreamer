import os
import time

import gymnasium as gym
import torch
import torch.nn as nn

from lib.actor import Actor
from lib.config import Config
from lib.critic import Critic
from lib.replay_buffer import ReplayBuffer
from lib.utils import log_episode_video, make_env, symexp, log_wm_reconstruction_video, log_wm_imagination_video
from lib.world_model import WorldModel, WorldModelState


def train(cfg: Config, summary_writer=None):
    env = make_env(cfg.env_id)

    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # --- Shapes & Sizes ---
    obs_shape = obs_space.shape
    act_size = act_space.n
    print(f"Observation shape: {obs_shape}, Action size: {act_size}")

    # --- Models ---
    world_model = WorldModel(
        obs_shape=obs_shape,
        action_size=act_size,
        num_latents=cfg.num_latents,
        classes_per_latent=cfg.classes_per_latent,
        hidden_size=cfg.hidden_size,
        base_cnn_channels=cfg.base_cnn_channels,
        mlp_hidden_units=cfg.mlp_hidden_units,
        free_bits=cfg.free_bits,
        beta_pred=cfg.beta_pred,
        beta_dyn=cfg.beta_dyn,
        beta_rep=cfg.beta_rep,
        unimix_eps=cfg.unimix_eps
    ).to(cfg.device)
    model_state_size = cfg.hidden_size + cfg.num_latents * cfg.classes_per_latent
    world_model.train()

    actor = Actor(
        state_size=model_state_size,
        action_size=act_size,
        mlp_hidden_units=cfg.mlp_hidden_units,
        mlp_layers=2,
        entropy_scale=cfg.actor_entropy_scale,
        ret_norm_limit=cfg.actor_ret_norm_limit,
        ret_norm_decay=cfg.actor_ret_norm_decay,
        unimix_eps=cfg.unimix_eps,
    ).to(cfg.device)
    actor.train()

    critic = Critic(
        state_size=model_state_size,
        mlp_hidden_units=cfg.mlp_hidden_units,
        mlp_layers=2,
        num_buckets=cfg.critic_num_buckets,
        bucket_min=cfg.critic_bucket_min,
        bucket_max=cfg.critic_bucket_max,
        ema_decay=cfg.critic_ema_decay,
        ema_regularizer=cfg.critic_ema_regularizer,
    ).to(cfg.device)
    critic.train()

    # print model sizes
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def count_learnable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"World Model parameters: {count_params(world_model):,}")
    print(f"Actor parameters: {count_params(actor):,}")
    print(f"Critic parameters: {count_params(critic):,}")

    print(f"Total parameters: {count_params(world_model) + count_params(actor) + count_params(critic):,}")
    print(
        f"Total learnable parameters: {count_learnable_params(world_model) + count_learnable_params(actor) + count_learnable_params(critic):,}"
    )

    # --- Replay Buffer ---
    buffer = ReplayBuffer(
        capacity=cfg.replay_capacity,
        obs_shape=obs_shape,
        seq_len=cfg.batch_length,
        device=cfg.device
    )

    # --- Optimizers ---
    optim_world_model = torch.optim.Adam(
        world_model.parameters(), lr=cfg.world_model_lr, eps=cfg.world_model_adam_eps, fused=True
    )
    optim_actor = torch.optim.Adam(
        actor.parameters(), lr=cfg.actor_critic_lr, eps=cfg.actor_critic_adam_eps, fused=True
    )
    optim_critic = torch.optim.Adam(
        critic.parameters(), lr=cfg.actor_critic_lr, eps=cfg.actor_critic_adam_eps, fused=True
    )

    # --- Training Loop ---
    time_counter = time.time()

    # Logic for train ratio
    update_credit = 0.0  # accumulated replayed steps
    credits_per_update = cfg.batch_size * cfg.batch_length  # replayed steps per WM update
    updates_done = 0  # total gradient updates run
    update_counter = 0  # counts updates for perf logging
    policy_steps = 0  # total env steps (= policy steps)

    model_state = world_model.init_state(1, cfg.device)
    last_action_idx = torch.zeros(1, dtype=torch.long, device=cfg.device)
    last_cont = torch.ones(1, dtype=torch.float32, device=cfg.device)

    current_obs = None
    for _ in range(cfg.num_iterations):
        # --- Collect a single transition ---
        if current_obs is None:
            obs, _ = env.reset()
            current_obs = obs
        # Sample action using actor on current world model state
        with torch.no_grad():
            # Update model state with latest observation embedding and last action
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0) / 255.0
            model_state, _ = world_model.step(model_state, last_action_idx, obs_tensor, last_cont)
            dist = actor(model_state)
            action_idx = dist.sample().item()

        # Overwrite action with random action if the actor is not yet trained
        if updates_done == 0:
            action_idx = env.action_space.sample()

        # interact with environment
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        cont = not (terminated or truncated)

        buffer.store(current_obs, action_idx, float(reward), cont)

        current_obs = next_obs
        last_action_idx.fill_(action_idx)
        last_cont.fill_(float(cont))

        if not cont:
            current_obs, _ = env.reset()
            model_state = world_model.init_state(1, cfg.device)
            last_action_idx.zero_()
            last_cont.fill_(1.0)

        # For train ratio logic
        policy_steps += 1
        update_credit += cfg.train_ratio

        # --- When enough data collected, update the models ---
        while len(buffer) > cfg.batch_length and update_credit >= credits_per_update:
            print(f"Update {updates_done}, total env steps {policy_steps}, "
                  f"buffer size {len(buffer)}, update credit {update_credit:.1f}")
            # --- train the world model ---
            batch = buffer.sample(cfg.batch_size)
            obs = batch["observations"].float() / 255.0
            actions = batch["actions"]
            rewards = batch["rewards"]
            continues = batch["continues"]

            world_model_loss, world_model_tensor_dict = world_model.loss(obs, actions, rewards, continues)

            optim_world_model.zero_grad()
            world_model_loss.backward()
            nn.utils.clip_grad_norm_(world_model.parameters(), cfg.world_model_grad_clip)
            optim_world_model.step()

            # --- train actor and critic using in imagination ---
            with torch.no_grad():
                # Recycle the model state from the world model training
                start_state: WorldModelState = world_model_tensor_dict["state"]

                # Imagine H steps with the actor
                H = cfg.imagination_horizon
                imagination_states = []
                imagination_actions = []
                imagination_dists = []
                imagination_rewards = []
                imagination_cont_prob = []

                s = start_state
                for _ in range(H):
                    dist = actor(s)
                    a = dist.sample()  # sample action
                    s, info = world_model.step(s, a_prev_idx=a)  # imagine with no obs
                    r = symexp(info["r_hat"].squeeze(-1))  # reward imagination
                    c = torch.sigmoid(info["c_hat"].squeeze(-1))  # continue prob imagination

                    imagination_states.append(s)
                    imagination_actions.append(a)
                    imagination_dists.append(dist)
                    imagination_rewards.append(r)
                    imagination_cont_prob.append(c)

                # stack along time: (B,H, ...)
                Hs = WorldModelState(
                    h=torch.stack([st.h for st in imagination_states], dim=1).detach(),  # (B,H,Hdim)
                    z=torch.stack([st.z for st in imagination_states], dim=1).detach(),  # (B,H,L,K)
                )
                A = torch.stack(imagination_actions, dim=1)  # (B,H)
                R = torch.stack(imagination_rewards, dim=1)  # (B,H)
                C = torch.stack(imagination_cont_prob, dim=1)  # (B,H)

                V = critic.value(Hs)  # (B,H)
                V_last = critic.value(WorldModelState(
                    h=imagination_states[-1].h,
                    z=imagination_states[-1].z
                ))  # (B,)

                lam_returns = torch.zeros_like(V)
                next_ret = V_last
                for t in reversed(range(H)):
                    ret = R[:, t] + cfg.gamma * (1.0 - C[:, t]) * (
                            (1.0 - cfg.lam) * V[:, t] + cfg.lam * next_ret
                    )
                    lam_returns[:, t] = ret
                    next_ret = ret

            # ---- Critic update ----
            critic_loss = critic.loss(
                model_states=Hs,
                returns=lam_returns.detach(),  # (B,H)
            )
            optim_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.actor_critic_grad_clip)
            optim_critic.step()
            critic.update_slow()

            # ---- Actor update ----
            actor_loss = actor.loss(
                model_states=Hs,
                actions=A.detach(),  # (B,H)
                returns=lam_returns.detach(),  # (B,H)
                values=V.detach(),  # (B,H)
            )
            optim_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.actor_critic_grad_clip)
            optim_actor.step()

            # ---- Rich logging ----
            if summary_writer is not None and updates_done % cfg.log_interval == 0 and updates_done > 0:
                with torch.no_grad():
                    # ---- Policy (imagination) ----
                    ent_im = torch.stack([d.entropy().mean() for d in imagination_dists]).mean().item()
                    dist_im = imagination_dists[-1]

                    vals_vec = V.detach().reshape(-1)
                    rew_vec = R.detach().reshape(-1)
                    cont_vec = C.detach().reshape(-1)

                    scale = (actor.ret_scale.p95 - actor.ret_scale.p05).detach().clamp(min=cfg.actor_ret_norm_limit)
                    adv = (lam_returns - V).detach().reshape(-1)
                    adv_scaled = adv / scale

                    # ---- Actor diagnostics ----
                    summary_writer.add_scalar("policy/ret_scale", scale.item(), updates_done)
                    summary_writer.add_scalar("policy/adv_mean", adv.mean().item(), updates_done)
                    summary_writer.add_scalar("policy/adv_std", adv.std(unbiased=False).item(), updates_done)
                    summary_writer.add_scalar("policy/adv_mean_scaled", adv_scaled.mean().item(), updates_done)
                    summary_writer.add_scalar("policy/adv_std_scaled", adv_scaled.std(unbiased=False).item(),
                                              updates_done)
                    summary_writer.add_scalar("policy/imagination_entropy", ent_im, updates_done)
                    summary_writer.add_histogram("policy/imagination_probs", dist_im.probs, updates_done)

                    # ---- Critic diagnostics ----
                    val_err = (lam_returns - V).reshape(-1)
                    summary_writer.add_scalar("value/mae_vs_returns", val_err.abs().mean().item(), updates_done)
                    summary_writer.add_scalar("value/bias_vs_returns", val_err.mean().item(), updates_done)

                    # ---- Critic/value & return stats ----
                    summary_writer.add_scalar("value/mean", vals_vec.mean().item(), updates_done)
                    summary_writer.add_scalar("value/std", vals_vec.std(unbiased=False).item(), updates_done)
                    summary_writer.add_scalar("value/lambda_ret_mean", lam_returns.mean().item(), updates_done)
                    summary_writer.add_scalar("value/lambda_ret_std", lam_returns.std(unbiased=False).item(),
                                              updates_done)

                    # ---- World model breakdown (from last WM update) ----
                    wmd = world_model_tensor_dict
                    summary_writer.add_scalar("world_model/pred_loss", wmd["pred_loss"].item(), updates_done)
                    summary_writer.add_scalar("world_model/dyn_loss", wmd["dyn_loss"].item(), updates_done)
                    summary_writer.add_scalar("world_model/rep_loss", wmd["rep_loss"].item(), updates_done)

                    # ---- Heads (reward/continue) ----
                    summary_writer.add_scalar("world_model/rew_mean", rew_vec.mean().item(), updates_done)
                    summary_writer.add_scalar("world_model/rew_std", rew_vec.std(unbiased=False).item(), updates_done)
                    summary_writer.add_scalar("world_model/cont_mean", cont_vec.mean().item(), updates_done)

                    # ---- Losses ----
                    summary_writer.add_scalar("train/world_model_loss", world_model_loss.item(), updates_done)
                    summary_writer.add_scalar("train/actor_loss", actor_loss.item(), updates_done)
                    summary_writer.add_scalar("train/critic_loss", critic_loss.item(), updates_done)

                    # ---- Performance ----
                    elapsed = time.time() - time_counter
                    time_counter = time.time()
                    iters_per_second = update_counter / elapsed
                    update_counter = 0
                    summary_writer.add_scalar("perf/updates_per_second", iters_per_second, updates_done)

                    # ---- Console log ----
                    print(f"Update {updates_done}, World Model Loss: {world_model_loss.item():.3f}, "
                          f"Actor Loss: {actor_loss.item():.3f}, "
                          f"Critic Loss: {critic_loss.item():.3f}, "
                          f"Updates/sec: {iters_per_second:.3f}")

            # Periodic evaluation video
            if updates_done % cfg.video_interval == 0 and updates_done > 0 and summary_writer is not None:
                world_model.eval()
                actor.eval()
                log_episode_video(cfg, summary_writer, env, world_model, actor, updates_done)
                log_wm_reconstruction_video(cfg, summary_writer, env, world_model, actor, updates_done)
                log_wm_imagination_video(cfg, summary_writer, env, world_model, actor, updates_done)
                world_model.train()
                actor.train()

            # Save model
            if updates_done % cfg.save_interval == 0 and updates_done > 0 and cfg.checkpoint_dir is not None:
                torch.save({
                    "world_model": world_model.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "optim_world_model": optim_world_model.state_dict(),
                    "optim_actor": optim_actor.state_dict(),
                    "optim_critic": optim_critic.state_dict(),
                }, os.path.join(cfg.checkpoint_dir, f"ckpt.pth"))
                print(f"Saved model checkpoint to {cfg.checkpoint_dir}")

            # consume credit and count this update
            update_credit -= credits_per_update
            updates_done += 1
            update_counter += 1

    env.close()
