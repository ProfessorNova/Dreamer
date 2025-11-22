# losses/normalization.py
import torch
from torch import nn


class LossRMSNormalizer(nn.Module):
    """
    Tracks running RMS of named loss terms and normalizes them.
    """

    def __init__(self, momentum: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("state", torch.zeros(0))  # will hold values
        self.names = []

    def _get_index(self, name: str):
        if name not in self.names:
            self.names.append(name)
            new_state = torch.cat(
                [self.state, torch.tensor([0.0], device=self.state.device)]
            )
            self.state = new_state
        return self.names.index(name)

    def forward(self, name: str, value: torch.Tensor):
        idx = self._get_index(name)
        v = value.detach().float()
        rms = torch.sqrt(torch.mean(v ** 2) + self.eps)
        self.state[idx] = (
                self.momentum * self.state[idx] + (1 - self.momentum) * rms
        )
        normed = value / (self.state[idx] + self.eps)
        return normed
