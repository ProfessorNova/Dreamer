from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Cyclic replay buffer for sequence sampling."""

    def __init__(
            self,
            capacity: int,
            obs_shape: Tuple[int, ...],
            action_dim: int,
            seq_len: int,
            device: torch.device
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.device = device

        # We store observations as uint8 to save memory if images
        self.observations = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8)
        self.next_observations = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8)

        # Store actions as integers for discrete actions or as floats if continuous
        self.actions = torch.empty((self.capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.dones = torch.empty(self.capacity, dtype=torch.bool)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.idx] = torch.as_tensor(obs, dtype=torch.uint8)
        self.actions[self.idx] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = torch.as_tensor(next_obs, dtype=torch.uint8)
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of contiguous sequences.

        Returns a dictionary containing sequences of observations,
        actions, rewards, next observations and done flags.  Observations and
        next observations are returned as floats in the range ``[0, 255]``.
        """
        assert len(self) >= self.seq_len, "Not enough data to sample."

        # Determine the valid range of starting indices. We avoid sampling
        # sequences that wrap around the end of the buffer.
        max_start = self.capacity if self.full else self.idx - self.seq_len
        assert max_start > 0, "Buffer is too small to sample the desired sequence length."
        starts = np.random.randint(0, max_start, size=batch_size)
        seq_range = np.arange(self.seq_len)
        indices = (starts[:, None] + seq_range[None, :]) % self.capacity

        obs = self.observations[indices].to(self.device).float()
        actions = self.actions[indices].to(self.device)
        rewards = self.rewards[indices].to(self.device)
        next_obs = self.next_observations[indices].to(self.device).float()
        dones = self.dones[indices].to(self.device)

        return {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_obs,
            "dones": dones,
        }
