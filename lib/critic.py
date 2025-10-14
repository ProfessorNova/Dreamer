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

from lib.utils import value_bins, two_hot_encode


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
            imagined: bool = True,
            replay_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute critic loss for a batch of states.

        Args:
            feats: Tensor of shape (batch, time, feat_size).
            target_returns: Tensor of shape (batch, time) containing λ‑returns.
            imagined: Whether the states come from imagined rollouts. If False,
                the loss is considered replay loss with a smaller weight.
            replay_weight: Additional weight applied to replay loss.

        Returns:
            Scalar critic loss.
        """
        batch, time, feat_size = feats.size()
        logits = self.forward(feats.view(-1, feat_size))
        logits = logits.view(batch * time, self.num_bins)
        # Encode target returns into twohot vectors over bins
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
