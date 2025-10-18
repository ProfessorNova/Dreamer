import torch
import torch.nn as nn

from lib.utils import log_unimix
from lib.world_model import WorldModelState


class EMAPercentileScale(nn.Module):
    def __init__(self, decay: float = 0.99, min_scale: float = 1.0):
        super().__init__()
        self.decay = decay
        self.min_scale = min_scale
        self.register_buffer("p05", torch.tensor(0.0))
        self.register_buffer("p95", torch.tensor(0.0))

    def update_get_S(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().reshape(-1).float()
        cur_S = (self.p95 - self.p05).clamp_min(self.min_scale)

        if x.numel() == 0:
            return cur_S

        q05 = torch.quantile(x, 0.05)
        q95 = torch.quantile(x, 0.95)
        q95 = torch.maximum(q95, q05 + x.new_tensor(1e-8))

        if self.training:
            d = 1.0 - self.decay
            self.p05.mul_(self.decay).add_(d * q05)
            self.p95.mul_(self.decay).add_(d * q95)

        S = (self.p95 - self.p05)
        return torch.maximum(S, x.new_tensor(self.min_scale))


class Actor(nn.Module):
    """
    The actor neural network learns behaviors purely from abstract sequences predicted by the world model.
    During environment interaction, we select actions by sampling from the actor
    network without lookahead planning. The actor and critic operate on model states.

    The actor aims to maximize the expected return for each model state.
    """

    def __init__(
            self,
            state_size: int,
            action_size: int,
            mlp_hidden_units: int = 512,
            mlp_layers: int = 2,
            entropy_scale: float = 1e-3,
            ret_norm_limit: float = 1.0,
            ret_norm_decay: float = 0.99,
            unimix_eps: float = 0.01,
    ):
        super(Actor, self).__init__()
        self.entropy_scale = entropy_scale
        self.unimix_eps = unimix_eps

        layers = []
        dim = state_size
        for _ in range(mlp_layers):
            layers.append(nn.RMSNorm(dim))
            layers.append(nn.Linear(dim, mlp_hidden_units))
            layers.append(nn.SiLU())
            dim = mlp_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mlp_hidden_units, action_size)

        # zero-init output to have initially uniform policy
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.ret_scale = EMAPercentileScale(decay=ret_norm_decay, min_scale=ret_norm_limit)

    @staticmethod
    def _state_vec(s: WorldModelState) -> torch.Tensor:
        """
        Convert WorldModelState to a flat feature vector.
        Works for both single states (B,H) and sequences of states (B,T,H).
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

    def forward(self, model_states: WorldModelState) -> torch.distributions.Distribution:
        x = self._state_vec(model_states)
        # Support sequences of states (B,T,H+L*K) or single states (B,H+L*K)
        if x.dim() == 3:
            B, T, D = x.shape
            x = self.mlp(x.reshape(B * T, D))
            logits = self.head(x).reshape(B, T, -1)
        else:
            x = self.mlp(x)
            logits = self.head(x)

        mixed_log_probs = log_unimix(logits, self.unimix_eps, dim=-1)
        return torch.distributions.Categorical(logits=mixed_log_probs)

    def loss(
            self,
            model_states: WorldModelState,
            actions: torch.Tensor,  # (B,) or (B,T) long indices
            returns: torch.Tensor,  # (B,) or (B,T) lambda-returns
            values: torch.Tensor,  # (B,) or (B,T) predicted values
    ) -> torch.Tensor:
        dist = self(model_states)
        log_probs = dist.log_prob(actions)

        scale = self.ret_scale.update_get_S(returns.detach())
        adv = (returns - values).detach()
        adv_scaled = adv / scale
        policy_loss = -(adv_scaled * log_probs).mean()

        entropy = dist.entropy().mean()
        loss = policy_loss - self.entropy_scale * entropy
        return loss
