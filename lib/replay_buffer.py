"""
Simple replay buffer for DreamerV3.

This buffer stores episodes of experience and allows sampling of
contiguous sequences for training the world model and imagined rollout
starting states.  Each episode consists of observations, actions,
rewards, and continuation flags (1 − done).  When sampling a batch,
episodes are chosen uniformly and a random window of fixed length is
extracted from each episode.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int = 1000, seq_len: int = 50):
        self.max_size = max_size
        self.seq_len = seq_len
        self.buffer: deque = deque(maxlen=max_size)

    def add_episode(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, continues: np.ndarray) -> None:
        """
        Add a full episode to the buffer.

        Args:
            obs: numpy array of shape (T+1, *obs_shape) containing observations.
            actions: numpy array of shape (T,) containing integer actions.
            rewards: numpy array of shape (T,) containing rewards.
            continues: numpy array of shape (T,) containing continuation flags (1−done).
        """
        self.buffer.append({
            "obs": obs,  # includes final observation
            "actions": actions,
            "rewards": rewards,
            "continues": continues,
        })

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of sequences for training.

        Returns:
            A dict containing obs, actions, rewards, continues each of shape
            (batch, seq_len, ...).  Observations include seq_len consecutive
            frames, so the returned obs has shape (batch, seq_len+1, *obs_shape).
        """
        assert len(self.buffer) > 0, "ReplayBuffer is empty"
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_continues = []
        for _ in range(batch_size):
            ep = random.choice(self.buffer)
            # pick random start index such that we have seq_len transitions
            max_start = len(ep["actions"]) - self.seq_len
            if max_start <= 0:
                start = 0
            else:
                start = random.randint(0, max_start)
            end = start + self.seq_len
            batch_obs.append(ep["obs"][start: end + 1])
            batch_actions.append(ep["actions"][start:end])
            batch_rewards.append(ep["rewards"][start:end])
            batch_continues.append(ep["continues"][start:end])
        return {
            "obs": np.stack(batch_obs, axis=0),
            "actions": np.stack(batch_actions, axis=0),
            "rewards": np.stack(batch_rewards, axis=0),
            "continues": np.stack(batch_continues, axis=0),
        }
