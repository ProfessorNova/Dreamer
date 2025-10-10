from typing import Tuple

import torch
import torch.nn as nn


class ReturnScaler:
    """
    Maintain running range estimates of returns for normalization.

    DreamerV3 normalizes returns to be roughly in [0, 1] but only scales
    down large magnitudes while leaving small returns untouched.
    This class tracks the 5th and 95th percentiles using an exponential
    moving average and provides scaling factors for normalization.
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.range = None

    def update(self, returns: torch.Tensor) -> None:
        # Compute percentiles along batch and time dimensions
        p5 = torch.quantile(returns, 0.05).item()
        p95 = torch.quantile(returns, 0.95).item()
        current_range = max(p95 - p5, 1e-6)
        if self.range is None:
            self.range = current_range
        else:
            self.range = self.momentum * self.range + (1 - self.momentum) * current_range

    def scale(self, returns: torch.Tensor) -> torch.Tensor:
        if self.range is None:
            return returns
        # Only scale returns larger than 1 by dividing by the running range
        return returns / max(1.0, self.range)


class Actor(nn.Module):
    """Discrete action actor network."""

    def __init__(self, feat_size: int, action_size: int, hidden_size: int = 400, entropy_scale: float = 3e-4):
        super().__init__()
        self.entropy_scale = entropy_scale
        self.fc = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.return_scaler = ReturnScaler()

    def forward(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.fc(feat)
        return torch.distributions.Categorical(logits=logits)

    def act(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return them along with log probabilities."""
        dist = self.forward(feat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def loss(
            self,
            feats: torch.Tensor,
            actions: torch.Tensor,
            returns: torch.Tensor,
            values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the actor loss for a batch of imagined trajectories.

        Args:
            feats: Tensor of shape (batch, time, feat_size) representing model
                features.
            actions: Tensor of shape (batch, time) with sampled actions.
            returns: Tensor of shape (batch, time) with λ‑returns (bootstrapped
                future rewards) computed by the critic.
            values: Tensor of shape (batch, time) with predicted values from the
                critic.

        Returns:
            Scalar actor loss.
        """
        batch, time, feat_size = feats.size()
        dist = self.forward(feats.view(-1, feat_size))
        log_probs = dist.log_prob(actions.view(-1))
        log_probs = log_probs.view(batch, time)
        entropies = dist.entropy().view(batch, time)
        # Compute advantage as normalized returns minus values
        adv = returns - values
        # Update return scaler and normalize advantages
        self.return_scaler.update(returns.detach())
        norm_adv = self.return_scaler.scale(adv)
        # Actor loss: negative log likelihood weighted by advantage and entropy regularizer
        loss = - (norm_adv.detach() * log_probs).mean() - self.entropy_scale * entropies.mean()
        return loss
