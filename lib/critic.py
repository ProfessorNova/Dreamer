import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import symlog, symexp
from lib.world_model import WorldModelState


class Critic(nn.Module):
    """
    The critic neural network learns to estimate the returns purely from abstract sequences predicted by the world model.
    """

    def __init__(
            self,
            state_size: int,
            hidden: int = 512,
            depth: int = 2,
            num_buckets: int = 255,
            bucket_min: float = -20.0,
            bucket_max: float = 20.0,
            ema_decay: float = 0.98,
            ema_regularizer: float = 1.0,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.register_buffer("buckets", torch.linspace(bucket_min, bucket_max, num_buckets))
        self.bucket_min = bucket_min
        self.bucket_max = bucket_max
        self.bucket_delta = (bucket_max - bucket_min) / (num_buckets - 1)

        layers = []
        dim = state_size
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_buckets)

        # zero-init output to avoid large early values
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # EMA of the target network
        self.ema_decay = ema_decay
        self.ema_regularizer = ema_regularizer
        self.slow = copy.deepcopy(self)
        for p in self.slow.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _state_vec(s: WorldModelState) -> torch.Tensor:
        """
        Convert WorldModelState to a flat feature vector.
        Works for both single states (B,H) and sequences of states (B,T,H).
        Also stops gradients to the world model.
        """
        h, z = s.h, s.z
        # Support (B,H) or (B,T,H)
        if h.dim() == 3:
            B, T, H = h.shape
            z_flat = z.reshape(B, T, -1)  # (B,T,L*K)
            features = torch.cat([h, z_flat], dim=-1)  # (B,T,H+L*K)
        else:
            B, H = h.shape
            z_flat = z.reshape(B, -1)  # (B,L*K)
            features = torch.cat([h, z_flat], dim=-1)  # (B,H+L*K)
        return features

    def forward(self, models_states: WorldModelState) -> torch.Tensor:
        """
        Forward pass through the critic network.
        """
        x = self._state_vec(models_states)
        if x.dim() == 3:
            B, T, D = x.shape
            x = self.mlp(x.reshape(B * T, D))
            logits = self.head(x).reshape(B, T, self.num_buckets)  # (B,T,num_buckets)
        else:
            x = self.mlp(x)  # (B,hidden)
            logits = self.head(x)  # (B,num_buckets)
        return logits

    def dist(self, models_states: WorldModelState) -> torch.distributions.Categorical:
        """
        Get a Categorical distribution over returns for the given states.
        """
        logits = self.forward(models_states)
        return torch.distributions.Categorical(logits=logits)

    def value(self, model_states: WorldModelState) -> torch.Tensor:
        """
        Estimate the value of a single state or a sequence of states.
        """
        logits = self(model_states)
        probs = logits.softmax(dim=-1)
        exp_buckets = (probs * self.buckets).sum(dim=-1)
        return symexp(exp_buckets)

    def _twohot(self, x_symlog: torch.Tensor) -> torch.Tensor:
        """
        Two-hot encode symlog targets against fixed buckets.
        Returns a tensor of shape (..., num_buckets) where the last dimension sums to 1.
        """
        # clamp into bucket range
        x = x_symlog.clamp(min=self.bucket_min, max=self.bucket_max)

        # fractional bucket index
        idx = (x - self.bucket_min) / self.bucket_delta
        lo = torch.floor(idx).long()
        hi = (lo + 1).clamp(max=self.num_buckets - 1)

        w_hi = (idx - lo.float()).clamp(0.0, 1.0)
        w_lo = 1.0 - w_hi

        # flatten, scatter, then reshape back
        flat = x.reshape(-1)
        lo_f = lo.reshape(-1)
        hi_f = hi.reshape(-1)
        w_lo_f = w_lo.reshape(-1)
        w_hi_f = w_hi.reshape(-1)

        y = x.new_zeros((flat.numel(), self.num_buckets))
        y.scatter_(1, lo_f.unsqueeze(1), w_lo_f.unsqueeze(1))
        y.scatter_add_(1, hi_f.unsqueeze(1), w_hi_f.unsqueeze(1))
        return y.view(*x.shape, self.num_buckets)

    @torch.no_grad()
    def update_slow(self):
        """
        Update the EMA target network.
        """
        for p, p_slow in zip(self.parameters(), self.slow.parameters()):
            p_slow.data.mul_(self.ema_decay)
            p_slow.data.add_(p.data * (1.0 - self.ema_decay))

    def loss(
            self,
            model_states: WorldModelState,
            returns: torch.Tensor,  # (B,) or (B,T) lambda-returns
    ) -> torch.Tensor:
        y = self._twohot(symlog(returns))
        logits = self(model_states)
        logp = F.log_softmax(logits, dim=-1)
        cross_entropy = -(y * logp).sum(dim=-1)  # (B,) or (B,T)
        loss = cross_entropy.mean()

        # EMA regularization towards slow network
        if self.ema_regularizer > 0.0:
            with torch.no_grad():
                slow_logits = self.slow(model_states)
                slow_probs = F.softmax(slow_logits, dim=-1)
            cross_entropy_slow = -(slow_probs * logp).sum(dim=-1)
            loss += self.ema_regularizer * cross_entropy_slow.mean()

        return loss
