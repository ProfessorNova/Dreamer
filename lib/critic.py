"""
Critic network and training for DreamerV3.

The critic estimates the distribution of returns for model states under
the current policy.  Instead of predicting a single scalar value, the
critic uses a categorical distribution over exponentially spaced bins
and is trained via two‑hot regression【145968576409203†L638-L667】.  This design allows the
critic to represent multi‑modal or highly skewed return distributions and
decouples the scale of gradients from the magnitude of targets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the symlog transformation element‑wise."""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Apply the inverse symlog (symexp) transformation."""
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def value_bins(num_bins: int = 255, min_value: float = -20.0, max_value: float = 20.0) -> torch.Tensor:
    """
    Construct exponentially spaced bins for value distribution heads.

    Args:
        num_bins: Number of discrete bins to create.
        min_value: Minimum value in symlog space.
        max_value: Maximum value in symlog space.

    Returns:
        A 1D tensor containing the bin centers in original space.
    """
    lin = torch.linspace(min_value, max_value, num_bins)
    bins = symexp(lin)
    return bins


def two_hot_encode(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Encode continuous values into two-hot vectors over the given bins.

    Args:
        values: Tensor of shape (batch,) containing continuous target values.
        bins: 1D tensor of shape (num_bins,) containing bin centers.

    Returns:
        Tensor of shape (batch, num_bins) with two-hot encoded rows.
    """
    v = values.view(-1).unsqueeze(-1)  # (N, 1)
    diff = (v - bins.view(1, -1)).abs()  # (N, num_bins)

    top2 = diff.topk(k=2, largest=False)
    idx1, idx2 = top2.indices[:, 0], top2.indices[:, 1]  # (N,), (N,)
    d1 = diff[torch.arange(diff.size(0)), idx1]
    d2 = diff[torch.arange(diff.size(0)), idx2]

    denom = d1 + d2 + 1e-8
    w1, w2 = d2 / denom, d1 / denom  # weights sum to 1

    N, K = v.size(0), bins.numel()
    one_hot = torch.zeros((N, K), device=values.device, dtype=values.dtype)

    # Use scatter_add_ to avoid deprecated reduce="add"
    idx = torch.stack([idx1, idx2], dim=1).long()  # (N, 2)
    w = torch.stack([w1, w2], dim=1)  # (N, 2)
    one_hot.scatter_add_(dim=1, index=idx, src=w)

    return one_hot


class Critic(nn.Module):
    """Distributional critic with a categorical value head."""

    def __init__(
            self,
            feat_size: int,
            hidden_size: int = 400,
            num_bins: int = 255,
            ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.bins = value_bins(num_bins)  # (num_bins,) tensor
        self.feat_size = feat_size
        self.fc = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bins),
        )
        # Create a target network for EMA updates
        self.ema_decay = ema_decay
        self.target_fc = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bins),
        )
        # Initialize target with same weights
        self._update_target_network(tau=0.0)

    def forward(self, feat: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Return logits for the value distribution."""
        if use_target:
            return self.target_fc(feat)
        return self.fc(feat)

    def value(self, feat: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Compute expected value as the mean of the categorical distribution."""
        logits = self.forward(feat, use_target)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins.to(feat.device)).sum(dim=-1)

    def loss(
            self,
            feats: torch.Tensor,
            target_returns: torch.Tensor,
            replay_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute critic loss for a batch of states.

        Args:
            feats: Tensor of shape (batch, time, feat_size).
            target_returns: Tensor of shape (batch, time) containing λ‑returns.
            replay_weight: Additional weight applied to replay loss.

        Returns:
            Scalar critic loss.
        """
        batch, time, feat_size = feats.size()
        logits = self.forward(feats.view(-1, feat_size))
        logits = logits.view(batch * time, self.num_bins)
        # Encode target returns into two hot vectors over bins
        target_flat = target_returns.view(-1)
        two_hot = two_hot_encode(target_flat, self.bins.to(target_returns.device))
        # Cross entropy loss
        ce = -(two_hot.detach() * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        ce = ce.view(batch, time)
        loss = ce.mean()
        # If replay, scale loss
        return replay_weight * loss

    def update_target(self) -> None:
        """Update target network parameters with exponential moving average."""
        for p, tp in zip(self.fc.parameters(), self.target_fc.parameters()):
            tp.data.mul_(self.ema_decay)
            tp.data.add_((1 - self.ema_decay) * p.data)

    def _update_target_network(self, tau: float = 1.0) -> None:
        """Initialize or fully update the target network parameters."""
        for p, tp in zip(self.fc.parameters(), self.target_fc.parameters()):
            tp.data.copy_(p.data * tau + tp.data * (1 - tau))
