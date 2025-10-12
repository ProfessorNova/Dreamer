import time

import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from lib.actor import Actor
from lib.config import Config
from lib.critic import Critic
from lib.replay_buffer import ReplayBuffer
from lib.utils import log_episode_video, make_env, log_wm_recon_video, log_wm_openloop_video_teacher, \
    log_wm_openloop_video_actor
from lib.world_model import WorldModel


def train(cfg: Config, summary_writer=None):
    env = make_env(cfg)
    eval_env = make_env(cfg, eval_env=True)

    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(act_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # --- Shapes & Sizes ---
    obs_shape = obs_space.shape
    act_size = act_space.n
    print(f"Observation shape: {obs_shape}, Action size: {act_size}")

    # --- Models ---
    world_model = WorldModel(
        obs_shape, act_size, cfg.embed_size, cfg.deter_size, cfg.stoch_size,
        cfg.free_nats, cfg.beta_pred, cfg.beta_dyn, cfg.beta_rep,
        units=cfg.mlp_units, depth=cfg.mlp_depth
    ).to(cfg.device)
    feat_size = cfg.deter_size + cfg.stoch_size

    actor = Actor(
        feat_size, act_size, entropy_scale=cfg.entropy_scale,
        units=cfg.mlp_units, depth=cfg.mlp_depth, unimix_eps=cfg.unimix_eps,
        ret_decay=cfg.ret_norm_decay, ret_min_scale=cfg.ret_norm_min_scale,
    ).to(cfg.device)

    critic = Critic(
        feat_size, num_bins=cfg.num_bins, ema_decay=cfg.ema_decay, ema_reg=cfg.ema_reg,
        units=cfg.mlp_units, depth=cfg.mlp_depth
    ).to(cfg.device)

    # print model sizes
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    print(f"World Model parameters: {count_params(world_model):,}")
    print(f"Actor parameters: {count_params(actor):,}")
    print(f"Critic parameters: {count_params(critic):,}")

    # --- Replay Buffer ---
    buffer = ReplayBuffer(cfg.buffer_capacity, obs_shape, act_size, cfg.seq_len, cfg.device)

    # --- Optimizers ---
    optim_model = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # --- Training Loop ---
    time_counter = time.time()
    iter_counter = 0

    env_steps = 0  # how many env steps have been taken
    credit = 0.0  # how many gradient steps are owed

    loss = None
    actor_loss = None
    critic_loss = None

    model_state = world_model.init_state(1, cfg.device)
    last_action = torch.zeros(1, act_size, device=cfg.device)

    current_obs = None
    for it in range(cfg.num_iterations):
        # --- Collect a single transition ---
        if current_obs is None:
            obs, _ = env.reset()
            current_obs = obs
        # Create one‑hot action from last action index
        # Sample action using actor on current world model state
        with torch.no_grad():
            # Update model state with latest observation embedding and last action
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
            # Perform posterior update
            embed = world_model.encoder(obs_tensor)
            belief = model_state['deter']
            _, _, stoch_prior, belief = world_model.rssm.prior(model_state["stoch"], belief, last_action)
            _, _, stoch_post = world_model.rssm.posterior(belief, embed)
            model_state = {"deter": belief, "stoch": stoch_post}
            feat = world_model.get_feat(model_state)
            dist = actor(feat)
            action_idx = dist.sample().item()

            # random steps for initial exploration
            if env_steps < cfg.warmup_steps:
                action_idx = env.action_space.sample()

            # log real-env policy stats
            if summary_writer is not None and it % cfg.log_interval == 0 and env_steps >= cfg.warmup_steps:
                ent_env = dist.entropy().mean().item()
                summary_writer.add_scalar("policy/env_entropy", ent_env, it)
                summary_writer.add_histogram("policy/env_probs", dist.probs, it)

        # interact with environment
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        done = terminated or truncated

        # Store transition in replay buffer
        a_onehot = torch.nn.functional.one_hot(
            torch.tensor(action_idx, device=cfg.device), num_classes=act_size
        ).float()
        buffer.store(current_obs, a_onehot.cpu().numpy(), float(reward), next_obs, done)

        current_obs = next_obs
        last_action = a_onehot.unsqueeze(0)
        env_steps += 1
        if done:
            current_obs, _ = env.reset()
            model_state = world_model.init_state(1, cfg.device)
            last_action = torch.zeros(1, act_size, device=cfg.device)

        if len(buffer) > max(cfg.batch_size, cfg.seq_len) and env_steps >= cfg.warmup_steps:
            # Get credit for this env step
            minibatches_per_env_step = cfg.replay_ratio / (cfg.batch_size * cfg.seq_len * cfg.frame_skip)
            credit = min(cfg.max_credit, credit + minibatches_per_env_step)

            # --- train the world model ---
            steps_done = 0
            while credit >= 1.0 and steps_done < cfg.max_steps_per_iter:
                batch = buffer.sample(cfg.batch_size)
                obs = batch["observations"]
                actions = batch["actions"]
                rewards = batch["rewards"]
                dones = batch["dones"]
                continues = (~dones).float()

                state = world_model.init_state(cfg.batch_size, cfg.device)
                loss, _, _ = world_model.loss(obs, actions, rewards, continues, state)

                optim_model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
                optim_model.step()

                # --- actor-critic in imagination ---
                with torch.no_grad():
                    state0 = world_model.init_state(cfg.batch_size, cfg.device)
                    embeds = world_model.encoder(obs)
                    post, _ = world_model.rssm.observe(embeds, actions, state0)
                    last_state = {
                        "deter": torch.stack([s["deter"] for s in post], dim=1)[:, -1],
                        "stoch": torch.stack([s["stoch"] for s in post], dim=1)[:, -1],
                    }

                    # roll out imagination
                    state = {"deter": last_state["deter"].clone(), "stoch": last_state["stoch"].clone()}
                    feats_list, actions_list = [], []
                    B = state["stoch"].size(0)

                    for t in range(cfg.imagination_horizon):
                        feat = world_model.get_feat(state)
                        dist_im_t = actor(feat)
                        action = dist_im_t.sample()
                        if action.dim() == 0: action = action.view(1)
                        if action.size(0) != B: action = action.expand(B)
                        a_onehot = one_hot(action, num_classes=act_size).float()
                        feats_list.append(feat)
                        actions_list.append(action)
                        _, _, stoch, deter = world_model.rssm.prior(state["stoch"], state["deter"], a_onehot)
                        state = {"deter": deter, "stoch": stoch}

                    imag_feats = torch.stack(feats_list, dim=1)  # (B,H,F)
                    imag_actions = torch.stack(actions_list, dim=1)  # (B,H)
                    B, H, F = imag_feats.shape
                    flat = imag_feats.view(B * H, F)
                    deter_flat, stoch_flat = flat[:, :cfg.deter_size], flat[:, cfg.deter_size:]

                    rewards_pred = world_model.reward_predictor.value(deter_flat, stoch_flat).view(B, H)
                    continues_pred = torch.sigmoid(
                        world_model.continue_predictor(deter_flat, stoch_flat)).view(B, H)

                    # online values for targets and bootstrap
                    values = critic.value(flat).view(B, H)
                    bootstrap = critic.value(world_model.get_feat(state))

                    # compute lambda-returns
                    lambda_returns = torch.zeros_like(values)
                    values_all = torch.cat([values, bootstrap.unsqueeze(1)], dim=1)
                    # values_all: [B, H+1], rewards_pred, continues_pred: [B, H]
                    ret = bootstrap
                    for i in reversed(range(H)):
                        discount = (cfg.gamma * continues_pred[:, i]).clamp_max(1.0)
                        ret = rewards_pred[:, i] + discount * ((1 - cfg.lam) * values_all[:, i + 1] + cfg.lam * ret)
                        lambda_returns[:, i] = ret
                    advantages = lambda_returns - values

                # detach imagined features so no gradients through world model
                detached_imag_feats = imag_feats.detach()
                detached_lamda_returns = lambda_returns.detach()
                detached_values = values.detach()
                detached_advantages = advantages.detach()
                detached_rewards_pred = rewards_pred.detach()
                detached_continues_pred = continues_pred.detach()

                # ---- Critic update ----
                critic_loss = critic.loss(detached_imag_feats, detached_lamda_returns)
                optim_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
                optim_critic.step()
                critic.update_target()

                # ---- Actor update ----
                actor_loss = actor.loss(detached_imag_feats, imag_actions, detached_lamda_returns, detached_values)
                optim_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
                optim_actor.step()

                # ---- Rich logging ----
                if summary_writer is not None and steps_done == 0:
                    with torch.no_grad():
                        dist_im = actor(detached_imag_feats.view(-1, detached_imag_feats.size(-1)))
                        ent_im = dist_im.entropy().mean().item()
                        adv_vec = detached_advantages.view(-1)
                        vals_vec = detached_values.view(-1)
                        rew_vec = detached_rewards_pred.view(-1)
                        cont_vec = detached_continues_pred.view(-1)

                    summary_writer.add_scalar("policy/imag_entropy", ent_im, it)
                    summary_writer.add_histogram("policy/imag_probs", dist_im.probs, it)
                    summary_writer.add_scalar("value/mean", vals_vec.mean().item(), it)
                    summary_writer.add_scalar("value/std", vals_vec.std(unbiased=False).item(), it)
                    summary_writer.add_scalar("returns/lambda_mean", lambda_returns.mean().item(), it)
                    summary_writer.add_scalar("returns/lambda_std", lambda_returns.std(unbiased=False).item(), it)
                    summary_writer.add_scalar("adv/mean", adv_vec.mean().item(), it)
                    summary_writer.add_scalar("adv/std", adv_vec.std(unbiased=False).item(), it)
                    summary_writer.add_scalar("reward_pred/mean", rew_vec.mean().item(), it)
                    summary_writer.add_scalar("reward_pred/std", rew_vec.std(unbiased=False).item(), it)
                    summary_writer.add_scalar("continue_pred/mean", cont_vec.mean().item(), it)

                # Pay back the owed credit
                credit -= 1.0
                credit = max(0.0, credit)
                steps_done += 1

        # ---- Console + scalar logs ----
        if it % cfg.log_interval == 0 and it > 0 and loss is not None:
            elapsed = time.time() - time_counter
            time_counter = time.time()
            iters_per_second = iter_counter / elapsed
            iter_counter = 0
            print(f"Iters {it}, WM Loss: {loss.item():.3f}, "
                  f"Actor Loss: {actor_loss.item():.3f}, "
                  f"Critic Loss: {critic_loss.item():.3f}, "
                  f"Iters/sec: {iters_per_second:.3f}")
            if summary_writer is not None:
                summary_writer.add_scalar("perf/iters_per_second", iters_per_second, it)
                summary_writer.add_scalar("train/world_model_loss", loss.item(), it)
                if actor_loss is not None:
                    summary_writer.add_scalar("train/actor_loss", actor_loss.item(), it)
                if critic_loss is not None:
                    summary_writer.add_scalar("train/critic_loss", critic_loss.item(), it)

        # Periodic evaluation video
        if it % cfg.video_interval == 0 and it > 0 and summary_writer is not None and env_steps >= cfg.warmup_steps:
            world_model.eval()
            actor.eval()
            log_wm_recon_video(cfg, summary_writer, world_model, buffer, it, seq_len=64)
            log_wm_openloop_video_teacher(cfg, summary_writer, world_model, buffer, it, context=16, horizon=48)
            log_wm_openloop_video_actor(cfg, summary_writer, world_model, actor, buffer, it, context=16, horizon=48)
            log_episode_video(cfg, summary_writer, eval_env, world_model, actor, it)
            world_model.train()
            actor.train()

        iter_counter += 1

    env.close()
    eval_env.close()
