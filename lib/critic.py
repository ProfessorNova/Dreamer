import torch
import torch.nn as nn

from lib.nn_blocks import ResidualMLP


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


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
    idx = torch.stack([idx1, idx2], dim=1)
    w = torch.stack([w1, w2], dim=1)
    target.scatter_add_(dim=1, index=idx.long(), src=w)
    return target


class Critic(nn.Module):
    def __init__(self, feat_size, num_bins=255, ema_decay=0.99, ema_reg=1.0,
                 units=256, depth=16):
        super().__init__()
        self.num_bins = num_bins
        self.ema_decay = ema_decay
        self.ema_reg = ema_reg

        # Register bins as buffer (auto device, no grads)
        self.register_buffer("bins_symlog", value_bins(num_bins), persistent=False)

        self.net = ResidualMLP(feat_size, num_bins, units=units, depth=depth, zero_init=True)
        self.target_net = ResidualMLP(feat_size, num_bins, units=units, depth=depth, zero_init=True)

        # Freeze target + eval
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.target_net.eval()

        self._hard_update()

    @torch.no_grad()
    def _hard_update(self):
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.copy_(p)

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.net.parameters(), self.target_net.parameters()):
            tp.lerp_(p, 1.0 - self.ema_decay)

    def forward(self, feat: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        return (self.target_net if use_target else self.net)(feat)

    @torch.no_grad()
    def value(self, feat: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        logits = self.forward(feat, use_target=use_target)
        probs = logits.softmax(dim=-1)
        bins = self.bins_symlog

        pos_mask = (bins >= 0).float()
        neg_mask = 1.0 - pos_mask
        exp_pos = (probs * (bins * pos_mask)).sum(dim=-1)
        exp_neg = (probs * (bins * neg_mask)).sum(dim=-1)
        exp_symlog = exp_pos + exp_neg
        return symexp(exp_symlog)

    def loss(self, feats: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        B, T, F = feats.shape
        flat = feats.contiguous().view(B * T, F)
        logits = self.net(flat)  # online

        y_symlog = symlog(target_returns.reshape(-1))
        target_twohot = two_hot_encode(y_symlog, self.bins_symlog)
        ce = -(target_twohot.detach() * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

        # EMA consistency regularizer: KL(ema || online) = CE(ema, online) - const
        with torch.no_grad():
            ema_probs = self.target_net(flat).softmax(dim=-1)
        reg = -(ema_probs * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

        return ce + self.ema_reg * reg
