import gymnasium as gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn.functional import softmax, one_hot

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
        writer: SummaryWriter,
        env: gym.Env,
        world_model: WorldModel,
        actor: Actor,
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


# --- Helpers for logging videos comparing GT vs reconstructions/predictions ---

def _to_uint8_chw(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [0,1] then to uint8 CHW."""
    x = x.detach().cpu()
    if x.dtype.is_floating_point:
        x = x.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x  # CHW


def _tile_horiz(chw_list: list[torch.Tensor]) -> torch.Tensor:
    """Horizontally concat list of CHW uint8 tensors sharing H,W."""
    return torch.cat(chw_list, dim=-1)  # concat width


def _stack_video(frames_chw: list[torch.Tensor]) -> torch.Tensor:
    """
    TensorBoardX add_video expects (N, T, C, H, W).
    Here we build a single N=1 video from a list of CHW frames.
    """
    vid = torch.stack(frames_chw, dim=0)  # (T, C, H, W)
    return vid.unsqueeze(0)  # (1, T, C, H, W)


@torch.inference_mode()
def log_wm_recon_video(cfg, writer, world_model, buffer, global_step, seq_len: int = 16):
    """
    Posterior reconstructions: for one sequence from replay, show GT vs recon.
    Produces a video where each frame is [GT | Recon] for time t=0...seq_len-1.
    """
    # sample a single sequence
    batch = buffer.sample(1)
    obs = batch["observations"][:, :seq_len]  # (1, T, C, H, W) images expected
    actions = batch["actions"][:, :seq_len]  # (1, T, A)

    if obs.dim() != 5:
        print("log_wm_recon_video: non-image observations; skipping.")
        return

    embeds = world_model.encoder(obs.to(cfg.device))  # (1,T,E)
    state0 = world_model.init_state(1, cfg.device)
    post, _ = world_model.rssm.observe(embeds, actions.to(cfg.device), state0)

    deter = torch.stack([s["deter"] for s in post], dim=1)  # (1,T,D)
    stoch = torch.stack([s["stoch"] for s in post], dim=1)  # (1,T,S)

    # decode reconstructions
    rec = world_model.decoder(
        deter.reshape(-1, deter.size(-1)),
        stoch.reshape(-1, stoch.size(-1))
    )
    # reshape back to (T, C, H, W), clamp to [0,1]
    if rec.dim() == 4:  # image path produced NCHW already (N=T)
        rec = rec
    else:
        raise RuntimeError("Decoder produced non-image in log_wm_recon_video.")

    rec = rec.view(seq_len, *rec.shape[1:])  # (T,C,H,W)
    gt = obs.squeeze(0).float() / 255.0  # (T,C,H,W)

    # Build frames: [GT | Recon]
    frames = []
    for t in range(seq_len):
        row = _tile_horiz([_to_uint8_chw(gt[t]), _to_uint8_chw(rec[t])])
        frames.append(row)
    vid = _stack_video(frames)
    writer.add_video("wm/recon_gt_vs_pred", vid, global_step=global_step, fps=cfg.video_fps)


@torch.inference_mode()
def log_wm_openloop_video_teacher(cfg, writer, world_model, buffer, global_step,
                                  context: int = 8, horizon: int = 24):
    """
    Open-loop prediction given *real* future actions from replay.
    Frames show [GT | Model] where:
      - first `context` steps use posterior reconstructions (teacher-forced)
      - following `horizon` steps decode the PRIOR rolled with real actions
    """
    batch = buffer.sample(1)
    obs = batch["observations"]  # (1, T, C, H, W)
    actions = batch["actions"]  # (1, T, A)

    if obs.dim() != 5:
        print("log_wm_openloop_video_teacher: non-image observations; skipping.")
        return

    T = obs.size(1)
    if T < context + horizon:
        # resample until we have enough steps
        seq_len = context + horizon
        batch = buffer.sample(1)
        obs = batch["observations"][:, :seq_len]
        actions = batch["actions"][:, :seq_len]

    embeds = world_model.encoder(obs.to(cfg.device))
    state0 = world_model.init_state(1, cfg.device)
    post, _ = world_model.rssm.observe(embeds[:, :context], actions[:, :context], state0)

    last = {
        "deter": torch.stack([s["deter"] for s in post], dim=1)[:, -1],
        "stoch": torch.stack([s["stoch"] for s in post], dim=1)[:, -1],
    }

    # decode teacher-forced recon for context frames
    deter_ctx = torch.stack([s["deter"] for s in post], dim=1).reshape(-1, post[0]["deter"].size(-1))
    stoch_ctx = torch.stack([s["stoch"] for s in post], dim=1).reshape(-1, post[0]["stoch"].size(-1))
    rec_ctx = world_model.decoder(deter_ctx, stoch_ctx).view(context, -1, obs.size(-2), obs.size(-1))  # (C,C,H,W)

    # roll prior with REAL future actions for horizon steps
    priors = world_model.rssm.imagine(actions[:, context:context + horizon].to(cfg.device), last)
    deter_fut = torch.stack([s["deter"] for s in priors], dim=1).reshape(-1, last["deter"].size(-1))
    stoch_fut = torch.stack([s["stoch"] for s in priors], dim=1).reshape(-1, last["stoch"].size(-1))
    pred_fut = world_model.decoder(deter_fut, stoch_fut).view(horizon, -1, obs.size(-2), obs.size(-1))

    # Build frames
    gt = (obs.squeeze(0).float() / 255.0)[:context + horizon]  # (C+H,C,H,W)
    frames = []
    for t in range(context + horizon):
        pred = rec_ctx[t] if t < context else pred_fut[t - context]
        frame = _tile_horiz([_to_uint8_chw(gt[t]), _to_uint8_chw(pred)])
        frames.append(frame)
    vid = _stack_video(frames)
    writer.add_video("wm/openloop_teacher_gt_vs_pred", vid, global_step=global_step, fps=cfg.video_fps)


@torch.inference_mode()
def log_wm_openloop_video_actor(cfg, writer, world_model, actor, buffer, global_step,
                                context: int = 8, horizon: int = 24):
    """
    Open-loop dream where the *actor* selects actions in imagination.
    Frames show [Posterior-ctx | Actor-dream]. Useful to visualize compounding error.
    """
    batch = buffer.sample(1)
    obs = batch["observations"]  # (1, T, C, H, W)
    actions = batch["actions"]  # (1, T, A)

    if obs.dim() != 5:
        print("log_wm_openloop_video_actor: non-image observations; skipping.")
        return

    embeds = world_model.encoder(obs[:, :context].to(cfg.device))
    state0 = world_model.init_state(1, cfg.device)
    post, _ = world_model.rssm.observe(embeds, actions[:, :context].to(cfg.device), state0)

    last = {
        "deter": torch.stack([s["deter"] for s in post], dim=1)[:, -1],
        "stoch": torch.stack([s["stoch"] for s in post], dim=1)[:, -1],
    }

    # decode posterior recon for the context
    deter_ctx = torch.stack([s["deter"] for s in post], dim=1).reshape(-1, post[0]["deter"].size(-1))
    stoch_ctx = torch.stack([s["stoch"] for s in post], dim=1).reshape(-1, post[0]["stoch"].size(-1))
    rec_ctx = world_model.decoder(deter_ctx, stoch_ctx).view(context, -1, obs.size(-2), obs.size(-1))

    # actor-driven imagination
    state = {k: v.clone() for k, v in last.items()}
    dream_frames = []
    for _ in range(horizon):
        feat = world_model.get_feat(state)
        dist = actor(feat)
        a = one_hot(dist.probs.argmax(dim=-1), num_classes=actions.size(-1)).float()
        _, _, stoch, deter = world_model.rssm.prior(state["stoch"], state["deter"], a)
        state = {"deter": deter, "stoch": stoch}
        # decode this dreamed step
        frame = world_model.decoder(deter, stoch).squeeze(0)  # CHW
        dream_frames.append(frame)

    # Build frames: show context recon then dreamed future; GT (left column) exists only for context
    gt_ctx = (obs.squeeze(0).float() / 255.0)[:context]
    frames = []
    for t in range(context + horizon):
        if t < context:
            left = _to_uint8_chw(gt_ctx[t])
            right = _to_uint8_chw(rec_ctx[t])
        else:
            left = _to_uint8_chw(rec_ctx[-1])
            right = _to_uint8_chw(dream_frames[t - context])
        frames.append(_tile_horiz([left, right]))
    vid = _stack_video(frames)
    writer.add_video("wm/openloop_actor_ctx_vs_dream", vid, global_step=global_step, fps=cfg.video_fps)
