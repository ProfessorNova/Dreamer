from lib.config import Config
from lib.utils import log_episode_video, make_env


def train(cfg: Config, summary_writer=None):
    env = make_env(cfg.env_id)

    for t in range(cfg.total_timesteps):
        if t % cfg.log_interval == 0:
            if summary_writer is not None:
                log_episode_video(cfg, summary_writer, env, None, t)
            print(f"Time step: {t}")
