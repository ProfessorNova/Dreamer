import torch
import torch.nn as nn

from lib.nn_blocks import MLP
from lib.utils import unimix_logits


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(x.abs()))


def value_bins(num_bins: int = 255, min_value: float = -20.0, max_value: float = 20.0) -> torch.Tensor:
    return torch.linspace(min_value, max_value, num_bins)


def two_hot_encode(values_symlog: torch.Tensor, bins_symlog: torch.Tensor) -> torch.Tensor:
    v = values_symlog.view(-1, 1)
    diff = (v - bins_symlog.view(1, -1)).abs()
    top2 = diff.topk(k=2, largest=False)
    idx1, idx2 = top2.indices[:, 0], top2.indices[:, 1]
    d1 = diff[torch.arange(diff.size(0)), idx1]
    d2 = diff[torch.arange(diff.size(0)), idx2]
    denom = d1 + d2 + 1e-8
    w1, w2 = d2 / denom, d1 / denom
    N, K = v.size(0), bins_symlog.numel()
    target = torch.zeros((N, K), device=values_symlog.device, dtype=values_symlog.dtype)
    idx = torch.stack([idx1, idx2], dim=1).long()
    w = torch.stack([w1, w2], dim=1)
    target.scatter_add_(dim=1, index=idx, src=w)
    return target


class Critic(nn.Module):
    def __init__(self, feat_size: int, num_bins: int = 255, ema_decay: float = 0.99,
                 units: int = 256, depth: int = 16):
        super().__init__()
        self.num_bins = num_bins
        self.bins_symlog = value_bins(num_bins)
        self.net = MLP(feat_size, num_bins, units=units, depth=depth)
        self.target_net = MLP(feat_size, num_bins, units=units, depth=depth)
        self.ema_decay = ema_decay
        self._hard_update()

    def _hard_update(self):
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.copy_(p.data)

    def update_target(self):
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.data.mul_(self.ema_decay).add_(p.data * (1 - self.ema_decay))

    def forward(self, feat: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        return (self.target_net if use_target else self.net)(feat)

    def value(self, feat: torch.Tensor, use_target: bool = False, eps: float = 0.01) -> torch.Tensor:
        logits = self.forward(feat, use_target=use_target)
        logp = unimix_logits(logits, eps)
        probs = logp.exp()
        bins = self.bins_symlog.to(feat.device)
        exp_symlog = (probs * bins).sum(dim=-1)
        return symexp(exp_symlog)

    def loss(self, feats: torch.Tensor, target_returns: torch.Tensor, replay_weight: float = 1.0,
             eps: float = 0.01) -> torch.Tensor:
        B, T, F = feats.size()
        flat = feats.view(B * T, F)
        logits = self.forward(flat)
        y_symlog = symlog(target_returns.view(-1))
        bins = self.bins_symlog.to(feats.device)
        target_twohot = two_hot_encode(y_symlog, bins)
        logp = unimix_logits(logits, eps)
        ce = -(target_twohot.detach() * logp).sum(dim=-1)
        return replay_weight * ce.mean()
