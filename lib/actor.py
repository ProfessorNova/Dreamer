import torch
import torch.nn as nn

from lib.world_model import WorldModelState


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
            entropy_scale: float = 1e-3,
            hidden: int = 512,
            depth: int = 2,
    ):
        super(Actor, self).__init__()
        self.entropy_scale = entropy_scale

        layers = []
        dim = state_size
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, action_size)

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
        return torch.distributions.Categorical(logits=logits)

    def loss(
            self,
            model_states: WorldModelState,
            actions: torch.Tensor,  # (B,) or (B,T) long indices
            returns: torch.Tensor,  # (B,) or (B,T) lambda-returns
            values: torch.Tensor  # (B,) or (B,T) predicted values
    ) -> torch.Tensor:
        action_dist = self(model_states)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        # TODO: Use EMAPercentileScaler for advantage normalization
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        loss = policy_loss - self.entropy_scale * entropy
        return loss
