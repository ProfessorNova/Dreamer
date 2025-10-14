"""
Entry point for training a DreamerV3 agent.

This script can be executed to train the agent on a given Gymnasium
environment.  It parses command‑line arguments for configuration and
invokes the training loop defined in `trainer.py`.  Example usage:

```
python -m dreamerv3.main --env PongNoFrameskip-v4 --num_steps 500000
```

Make sure to install the required dependencies (gymnasium, torch,
tensorboard) before running this script.  The TensorBoard logs can be
viewed with:

```
tensorboard --logdir runs/dreamerv3
```
"""
from __future__ import annotations

import argparse

import ale_py
from gymnasium import register_envs

from lib.trainer import train


def main() -> None:
    register_envs(ale_py)
    parser = argparse.ArgumentParser(description="Train DreamerV3 agent")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5", help="Gymnasium environment name")
    parser.add_argument("--num_steps", type=int, default=200_000, help="Number of environment steps to run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length for world model training")
    parser.add_argument("--imagine_horizon", type=int, default=15, help="Imagination horizon for actor/critic updates")
    parser.add_argument("--log_dir", type=str, default="runs/dreamerv3", help="TensorBoard log directory")
    args = parser.parse_args()
    train(
        env_name=args.env,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        imagine_horizon=args.imagine_horizon,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
