from typing import Tuple, Dict

import numpy as np
import torch
from typing_extensions import SupportsFloat


class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_shape: Tuple[int, ...],
            device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self.device = device

        # storage (CPU)
        self.observations = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.empty(self.capacity, dtype=torch.long)
        self.rewards = torch.empty(self.capacity, dtype=torch.float32)
        self.terminated = torch.empty(self.capacity, dtype=torch.bool)
        self.truncated = torch.empty(self.capacity, dtype=torch.bool)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def store(
            self,
            obs: np.ndarray,
            action: int,
            reward: SupportsFloat,
            terminated: bool,
            truncated: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self.observations[self.idx] = torch.as_tensor(obs, dtype=torch.uint8)
        self.actions[self.idx] = int(action)
        self.rewards[self.idx] = float(reward)
        self.terminated[self.idx] = bool(terminated)
        self.truncated[self.idx] = bool(truncated)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(
            self,
            batch_size: int,
            sequence_len: int,
            avoid_term_trunc: bool = False,
            max_tries_per_item: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a sequence batch of observations, actions, rewards, and done flags.

        Args:
            batch_size: number of sequences to sample (B)
            sequence_len: window length to sample (L)
            avoid_term_trunc: if True, guarantee *no* terminated/truncated=True
                inside the sampled window for each sequence.
            max_tries_per_item: rejection-sampling cap per sequence to avoid
                infinite loops when there aren't enough clean windows.

        Returns:
            Dict of tensors on self.device with shapes:
              - observations: (B, L, *obs_shape) uint8
              - actions:      (B, L) long
              - rewards:      (B, L) float32
              - terminated:   (B, L) bool
              - truncated:    (B, L) bool
        """
        assert len(self) >= sequence_len, "Not enough data to sample."
        B, L = batch_size, sequence_len

        # Helper to draw a single valid start index
        def draw_start() -> int:
            if self.full:
                return np.random.randint(0, self.capacity, dtype=np.int64)
            max_start = self.idx - L
            if max_start < 0:
                raise RuntimeError(
                    "Not enough data to sample the requested sequence length."
                )
            return np.random.randint(0, max_start + 1, dtype=np.int64)

        starts = np.empty(B, dtype=np.int64)
        for b in range(B):
            tries = 0
            while True:
                s = draw_start()
                if not avoid_term_trunc:
                    starts[b] = s
                    break
                # Check window for any term/trunc=True (wrap with modulo)
                window = (s + np.arange(L, dtype=np.int64)) % self.capacity
                if not (self.terminated[window].any().item() or self.truncated[window].any().item()):
                    starts[b] = s
                    break
                tries += 1
                if tries >= max_tries_per_item:
                    # If we can't find a clean span, fall back to using s.
                    # Alternatively: raise an error here if you prefer.
                    starts[b] = s
                    break

        # Build (B, L) index matrix with wrap-around
        idx_mat = (starts[:, None] + np.arange(L, dtype=np.int64)[None, :]) % self.capacity

        # Gather tensors (still CPU)
        obs = self.observations[idx_mat]  # (B, L, *obs_shape) uint8
        actions = self.actions[idx_mat]  # (B, L) long
        rewards = self.rewards[idx_mat]  # (B, L) float32
        terminated = self.terminated[idx_mat]  # (B, L) bool
        truncated = self.truncated[idx_mat]  # (B, L) bool

        # Move to device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)

        return {
            "observations": obs.contiguous(),
            "actions": actions.contiguous(),
            "rewards": rewards.contiguous(),
            "terminated": terminated.contiguous(),
            "truncated": truncated.contiguous(),
        }
