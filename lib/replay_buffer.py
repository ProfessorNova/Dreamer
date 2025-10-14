from typing import Dict, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_shape: Tuple[int, ...],
            seq_len: int,
            device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.seq_len = int(seq_len)
        self.device = device

        # storage (CPU)
        self.observations = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.empty(self.capacity, dtype=torch.long)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.continues = torch.empty(self.capacity, dtype=torch.bool)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def store(self, obs: np.ndarray, action: int, reward: float, cont: bool) -> None:
        """
        Add a single transition to the buffer.
        cont = True if the episode continues after this transition, else False at terminal.
        """
        self.observations[self.idx] = torch.as_tensor(obs, dtype=torch.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.continues[self.idx] = bool(cont)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a sequence batch of observations, actions, rewards, and continues.
        Observations are returned exactly as stored (uint8).
        """
        assert len(self) >= self.seq_len, "Not enough data to sample."

        B, T = batch_size, self.seq_len

        if self.full:
            # Uniform over all starts; modulo below handles wrap-around.
            starts = np.random.randint(0, self.capacity, size=B, dtype=np.int64)
        else:
            max_start = self.idx - T
            if max_start < 0:
                raise RuntimeError("Not enough data to sample the requested sequence length.")
            starts = np.random.randint(0, max_start + 1, size=B, dtype=np.int64)

        # build index matrix (B,T)
        seq_range = np.arange(T, dtype=np.int64)
        idx_mat = (starts[:, None] + seq_range[None, :]) % self.capacity

        # gather tensors
        obs = self.observations[idx_mat]  # (B,T,C,H,W) uint8
        actions = self.actions[idx_mat]  # (B,T) long
        rewards = self.rewards[idx_mat]  # (B,T) float32
        cont = self.continues[idx_mat]  # (B,T) bool

        # move to device; add final dims for r,c
        obs = obs.to(self.device)  # (B,T,C,H,W) uint8
        actions = actions.to(self.device)  # (B,T) long
        rewards = rewards.to(self.device)  # (B,T) float32
        continues = cont.to(self.device)  # (B,T) bool

        return {
            "observations": obs.contiguous(),
            "actions": actions.contiguous(),
            "rewards": rewards.contiguous(),
            "continues": continues.contiguous(),
        }
