from typing import Tuple

import torch
import torch.nn as nn

from lib.nn_blocks import MLP
from lib.utils import unimix_logits


class Actor(nn.Module):
    def __init__(self, feat_size: int, action_size: int,
                 entropy_scale: float = 1e-2, units: int = 256, depth: int = 16) -> None:
        super().__init__()
        self.action_size = action_size
        self.entropy_scale = entropy_scale
        self.mlp = MLP(feat_size, action_size, units=units, depth=depth)

    def forward(self, feat: torch.Tensor, eps: float = 0.01) -> torch.distributions.Categorical:
        logits = self.mlp(feat)
        logp = unimix_logits(logits, eps)
        return torch.distributions.Categorical(logits=logp)

    def act(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(feat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def loss(self, feats: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor,
             values: torch.Tensor) -> torch.Tensor:
        b, t, f = feats.shape
        dist = self.forward(feats.view(b * t, f))
        log_probs = dist.log_prob(actions.view(-1)).view(b, t)
        entropies = dist.entropy().view(b, t)
        advantage = returns - values
        adv_std = advantage.std().clamp(min=1e-3)
        norm_adv = (advantage / adv_std).clamp(-5.0, 5.0)
        return -(norm_adv.detach() * log_probs).mean() - self.entropy_scale * entropies.mean()
