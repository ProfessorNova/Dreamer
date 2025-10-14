from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Cyclic replay buffer for sequence sampling."""

    def __init__(
            self,
            capacity: int,
            obs_shape: Tuple[int, ...],
            seq_len: int,
            device: torch.device
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.seq_len = seq_len
        self.device = device

        self.observations = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.empty(self.capacity, dtype=torch.long)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.dones = torch.empty(self.capacity, dtype=torch.bool)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.idx] = torch.as_tensor(obs, dtype=torch.uint8)
        self.actions[self.idx] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        assert len(self) >= self.seq_len, "Not enough data to sample."

        if self.full:
            # valid starts are every index except the last (seq_len - 1) positions before idx
            # build valid starts in absolute space then map modulo capacity
            valid_count = self.capacity - self.seq_len
            base = (np.arange(valid_count) + self.idx) % self.capacity
            starts = np.random.choice(base, size=batch_size, replace=True)
        else:
            # we can start at [0 ... idx - seq_len]
            max_start = self.idx - self.seq_len
            assert max_start >= 0
            starts = np.random.randint(0, max_start + 1, size=batch_size)

        seq_range = np.arange(self.seq_len)
        indices = (starts[:, None] + seq_range[None, :]) % self.capacity

        obs = self.observations[indices].to(self.device).float()
        actions = self.actions[indices].to(self.device)
        rewards = self.rewards[indices].to(self.device)
        dones = self.dones[indices].to(self.device)

        return {
            "observations": obs.contiguous(),
            "actions": actions.contiguous(),
            "rewards": rewards.contiguous(),
            "dones": dones.contiguous(),
        }
