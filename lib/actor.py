import torch
import torch.nn as nn

from lib.nn_blocks import ResidualMLP


class EMAPercentileScale(nn.Module):
    def __init__(self, decay: float = 0.99, min_scale: float = 1.0):
        super().__init__()
        self.decay = decay
        self.min_scale = min_scale
        self.register_buffer("p5", torch.tensor(0.0))
        self.register_buffer("p95", torch.tensor(1.0))
        self._init = False

    def update_get_scale(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().reshape(-1).float()
        if x.numel() == 0:
            return torch.clamp(self.p95 - self.p5, min=self.min_scale)
        q05 = torch.quantile(x, 0.05)
        q95 = torch.quantile(x, 0.95)
        if not self._init:
            self.p5.copy_(q05)
            self.p95.copy_(q95)
            self._init = True
        else:
            self.p5.mul_(self.decay).add_(q05 * (1.0 - self.decay))
            self.p95.mul_(self.decay).add_(q95 * (1.0 - self.decay))
        return torch.clamp(self.p95 - self.p5, min=self.min_scale)


class Actor(nn.Module):
    def __init__(self, feat_size, action_size, entropy_scale=1e-3,
                 units=256, depth=16, unimix_eps=0.01, ret_decay=0.99, ret_min_scale=1.0):
        super().__init__()
        self.action_size = action_size
        self.entropy_scale = entropy_scale
        self.unimix_eps = unimix_eps
        self.mlp = ResidualMLP(feat_size, action_size, units, depth)
        self.ret_norm = EMAPercentileScale(decay=ret_decay, min_scale=ret_min_scale)

    def _dist(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.mlp(feat)
        probs = logits.softmax(dim=-1)
        # unimix -> no zero probs; small clamp just for safety
        k = probs.size(-1)
        probs = (1 - self.unimix_eps) * probs + self.unimix_eps / k
        probs = probs.clamp_min(1e-8)
        return torch.distributions.Categorical(probs=probs)

    def forward(self, feat):
        return self._dist(feat)

    def loss(self, feats, actions, returns, values):
        B, T, F = feats.shape
        dist = self._dist(feats.view(B * T, F))
        log_probs = dist.log_prob(actions.view(-1)).view(B, T)
        entropies = dist.entropy().view(B, T)

        adv = returns - values  # [B, T]
        scale = self.ret_norm.update_get_scale(returns)  # EMA of p95-p5 from lambda-returns
        adv_norm = (adv / scale).clamp(-5.0, 5.0).detach()

        loss_pg = -(adv_norm * log_probs).mean()
        loss_ent = - self.entropy_scale * entropies.mean()
        return loss_pg + loss_ent
