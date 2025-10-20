import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.utils import symlog, log_unimix


@dataclass
class WorldModelState:
    h: torch.Tensor  # (B, hidden_size)
    z: torch.Tensor  # (B, num_latents, classes_per_latent)


class SequenceModel(nn.Module):
    """
    Sequence model with recurrent state predicts next state h_t
    given current state h_{t-1}, stochastic representation z_{t-1}, and action a_{t-1}.
    """

    def __init__(
            self,
            action_size: int,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden_size: int = 512,
    ):
        super().__init__()
        self.register_buffer("z_reset", torch.full((num_latents, classes_per_latent), 1.0 / classes_per_latent))
        self.a_emb = nn.Embedding(action_size, 64)

        self.z_proj = nn.Linear(num_latents * classes_per_latent, hidden_size)
        self.a_proj = nn.Linear(64, hidden_size)

        self.norm = nn.RMSNorm(hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)

    def forward(
            self,
            h_prev: torch.Tensor,
            z_prev: torch.Tensor,
            a_prev_idx: torch.Tensor,
            c_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h_prev: (B, hidden_size)
            z_prev: (B, num_latents, classes_per_latent)
            a_prev_idx: (B,) with values in [0, action_size-1]
            c_prev: (B, 1)

        Returns:
            h_t: (B, hidden_size)
        """
        # reset hidden state and z if c_prev is 0
        if c_prev is not None:
            h_prev = h_prev * c_prev
            m = c_prev.view(-1, 1, 1)
            z_prev = z_prev * m + (1.0 - m) * self.z_reset.unsqueeze(0)

        # embed and set action to zero if c_prev is 0
        a_vec = self.a_emb(a_prev_idx)
        if c_prev is not None:
            a_vec = a_vec * c_prev

        z_flat = z_prev.view(z_prev.size(0), -1)
        x = self.z_proj(z_flat)
        x = x.add_(self.a_proj(a_vec))
        x = self.norm(x)
        return self.rnn(x, h_prev)


class Encoder(nn.Module):
    """
    Encoder that maps sensory inputs x_t to stochastic representations z_t.
    The posterior predictor that predicts z_t given h_t and x_t -> will act as a teacher for the dynamics predictor.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden_size: int = 512,
            base_cnn_channels: int = 32,
    ):
        super().__init__()
        C, H, W = obs_shape
        assert H % 16 == 0 and W % 16 == 0, "H and W must be divisible by 16"
        self.C, self.H, self.W = C, H, W
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

        self.conv = nn.Sequential(
            nn.Conv2d(C, base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base_cnn_channels, 2 * base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(2 * base_cnn_channels, 4 * base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(4 * base_cnn_channels, 8 * base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(obs_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + hidden_size, num_latents * classes_per_latent),
        )

    @torch.no_grad()
    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(o.nelement() / o.size(0))

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W)
            h_t: (B, hidden_size)

        Returns:
            logits over classes per latent: (B, num_latents, classes_per_latent)
        """
        B, C, H, W = x_t.shape
        assert (C, H, W) == (self.C, self.H, self.W), f"Got {(C, H, W)}, expected {(self.C, self.H, self.W)}"
        conv_out = self.conv(x_t)
        combined = torch.cat([conv_out, h_t.detach()], dim=-1)
        logits = self.fc(combined).view(-1, self.num_latents, self.classes_per_latent)
        return logits


class DynamicsPredictor(nn.Module):
    """
    Predictor that predicts next stochastic representation z_t.
    The prior predictor that predicts z_t given h_t -> learns to model the dynamics of the environment.
    """

    def __init__(
            self,
            hidden_size: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            mlp_hidden_units: int = 512,
            mlp_layers: int = 3,
    ):
        super().__init__()

        layers = []
        dim = hidden_size
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_hidden_units))
            layers.append(nn.SiLU())
            layers.append(nn.RMSNorm(mlp_hidden_units))
            dim = mlp_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mlp_hidden_units, num_latents * classes_per_latent)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, hidden_size)

        Returns:
            logits over classes per latent: (B, num_latents, classes_per_latent)
        """
        mlp_out = self.mlp(h_t)  # (B, mlp_hidden_units)
        logits = self.head(mlp_out).view(-1, self.num_latents, self.classes_per_latent)
        return logits


class RewardPredictor(nn.Module):
    """
    Predictor that predicts reward r_t with the concatenation of h_t and z_t.
    """

    def __init__(
            self,
            hidden_size: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            mlp_hidden_units: int = 512,
            mlp_layers: int = 1,
    ):
        super().__init__()

        layers = []
        dim = hidden_size + num_latents * classes_per_latent
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_hidden_units))
            layers.append(nn.SiLU())
            layers.append(nn.RMSNorm(mlp_hidden_units))
            dim = mlp_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mlp_hidden_units, 1)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, hidden_size)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            reward r_t: (B, 1)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)  # (B, num_latents * classes_per_latent)
        combined = torch.cat([h_t, z_t_flat], dim=-1)  # (B, hidden_size + num_latents * classes_per_latent)
        mlp_out = self.mlp(combined)  # (B, mlp_hidden_units)
        reward = self.head(mlp_out)  # (B, 1)
        return reward


class ContinuePredictor(nn.Module):
    """
    Predictor that predicts episode continuation flags c_t with the concatenation of h_t and z_t
    with c_t either being 0 or 1.
    """

    def __init__(
            self,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            mlp_hidden_units: int = 512,
            mlp_layers: int = 1,
    ):
        super().__init__()

        layers = []
        dim = h_dim + num_latents * classes_per_latent
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, mlp_hidden_units))
            layers.append(nn.SiLU())
            layers.append(nn.RMSNorm(mlp_hidden_units))
            dim = mlp_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(mlp_hidden_units, 1)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, hidden_size)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            continuation flag logits c_t: (B, 1)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)  # (B, num_latents * classes_per_latent)
        combined = torch.cat([h_t, z_t_flat], dim=-1)  # (B, hidden_size + num_latents * classes_per_latent)
        mlp_out = self.mlp(combined)  # (B, mlp_hidden_units)
        continue_logit = self.head(mlp_out)  # (B, 1)
        return continue_logit


class Decoder(nn.Module):
    """
    Decoder that reconstructs sensory inputs x_t from the concatenation of h_t and z_t.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden_size: int = 512,
            base_cnn_channels: int = 32,
    ):
        super().__init__()
        C, H, W = obs_shape
        assert H % 16 == 0 and W % 16 == 0, "H and W must be divisible by 16"
        self.C, self.H, self.W = C, H, W
        self.base_h, self.base_w = H // 16, W // 16
        self.cnn_multiplier = base_cnn_channels

        B = base_cnn_channels

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + num_latents * classes_per_latent,
                      base_cnn_channels * 8 * self.base_h * self.base_w),
            nn.SiLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8 * base_cnn_channels, 4 * base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(4 * base_cnn_channels, 2 * base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(2 * base_cnn_channels, base_cnn_channels, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(base_cnn_channels, C, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, hidden_size)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            reconstructed x_t: (B, C, H, W)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)
        combined = torch.cat([h_t, z_t_flat], dim=-1)
        fc_out = self.fc(combined)  # (B, cnn_multiplier * 8 * base_h * base_w)
        deconv_input = fc_out.view(B, self.cnn_multiplier * 8, self.base_h, self.base_w)
        x_recon = self.deconv(deconv_input)  # (B, C, H, W)
        assert x_recon.shape == (B, self.C, self.H, self.W)
        return x_recon


class WorldModel(nn.Module):
    """
    The World Model is a RSSM that combines the above components.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_size: int,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden_size: int = 512,
            base_cnn_channels: int = 32,
            mlp_hidden_units: int = 512,
            free_bits: float = 1.0,
            beta_pred: float = 1.0,
            beta_dyn: float = 0.5,
            beta_rep: float = 0.1,
            unimix_eps: float = 0.01,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.classes_per_latent = classes_per_latent
        self.free_bits = free_bits
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.unimix_eps = unimix_eps

        self.seq = SequenceModel(
            action_size=action_size,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden_size=hidden_size,
        )
        self.enc = Encoder(
            obs_shape=obs_shape,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden_size=hidden_size,
            base_cnn_channels=base_cnn_channels,
        )
        self.dyn = DynamicsPredictor(
            hidden_size=hidden_size,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            mlp_hidden_units=mlp_hidden_units,
            mlp_layers=3,
        )
        self.rew = RewardPredictor(
            hidden_size=hidden_size,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            mlp_hidden_units=mlp_hidden_units,
            mlp_layers=1,
        )
        self.cont = ContinuePredictor(
            h_dim=hidden_size,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            mlp_hidden_units=mlp_hidden_units,
            mlp_layers=1,
        )
        self.dec = Decoder(
            obs_shape=obs_shape,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden_size=hidden_size,
            base_cnn_channels=base_cnn_channels,
        )

        # zero-init reward and continue predictors to avoid large early prediction errors
        nn.init.zeros_(self.rew.head.weight)
        nn.init.zeros_(self.rew.head.bias)
        nn.init.zeros_(self.cont.head.weight)
        nn.init.zeros_(self.cont.head.bias)

    @torch.no_grad()
    def init_state(self, batch_size: int, device=None, dtype=None) -> WorldModelState:
        h0 = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        z0 = torch.full(
            (batch_size, self.num_latents, self.classes_per_latent),
            1.0 / self.classes_per_latent,
            device=device,
            dtype=dtype,
        )
        return WorldModelState(h=h0, z=z0)

    def step(
            self,
            state_prev: WorldModelState,
            a_prev_idx: torch.Tensor,  # (B,)
            x_cur: Optional[torch.Tensor] = None,  # (B,C,H,W) or None for imagination
            c_prev: Optional[torch.Tensor] = None,  # (B,1) or None either 0 or 1
    ) -> Tuple[WorldModelState, Dict[str, Any]]:
        """
        One RSSM step.
        If x_t is provided -> posterior; else -> prior (imagination).
        Returns new state and a dict with heads + logits for losses.
        """
        h_prev, z_prev = state_prev.h, state_prev.z

        # The sequence model updates the recurrent state h_t
        h_cur = self.seq(h_prev, z_prev, a_prev_idx, c_prev)  # (B, hidden_size)

        # The encoder maps sensory input x_t to posterior logits
        z_logits_post = self.enc(x_cur, h_cur) if x_cur is not None else None  # (B, L, K) or None

        # The dynamics predictor predicts prior logits from h_t
        z_logits_prior = self.dyn(h_cur)  # (B, L, K)

        # The logits for z_t are from the encoder if available, else from the dynamics predictor
        z_logits = z_logits_post if z_logits_post is not None else z_logits_prior  # (B, L, K)

        # Sample z_t
        z = self._sample_z(z_logits)  # (B, L, K)

        # Heads (reconstruction, reward, continue)
        r_hat = self.rew(h_cur, z)  # (B,1)
        c_hat = self.cont(h_cur, z)  # (B,1)
        x_hat = self.dec(h_cur, z)  # (B,C,H,W)

        new_state = WorldModelState(h=h_cur, z=z)
        info: Dict[str, Any] = {
            "prior_logits": z_logits_prior,
            "post_logits": z_logits_post,  # None in imagination
            "r_hat": r_hat,  # (B,1)
            "c_hat": c_hat,  # (B,1)
            "x_hat": x_hat,  # (B,C,H,W)
        }
        return new_state, info

    def _sample_z(self, logits: torch.Tensor) -> torch.Tensor:
        log_prob = log_unimix(logits, self.unimix_eps, dim=-1)
        return F.gumbel_softmax(log_prob, tau=1, hard=True, dim=-1)

    def _kl_loss(self, q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
        # log-unimix returns log((1-eps)*softmax + eps/K)
        log_q = log_unimix(q_logits, self.unimix_eps, dim=-1)  # (B, L, K)
        log_p = log_unimix(p_logits, self.unimix_eps, dim=-1)  # (B, L, K)

        q = log_q.exp()  # (B, L, K)
        kl_latents = (q * (log_q - log_p)).sum(dim=-1)  # (B, L)

        # per-latent free bits, split across L when summing
        fb_per_latent = self.free_bits / float(self.num_latents)
        kl_latents = kl_latents.clamp_min(fb_per_latent)  # (B, L)

        return kl_latents.sum(dim=-1)  # (B,)

    @staticmethod
    def _prediction_loss(
            x_true: torch.Tensor,  # (B,C,H,W)
            x_hat: torch.Tensor,  # (B,C,H,W)
            r_true: torch.Tensor,  # (B,1)
            r_hat: torch.Tensor,  # (B,1)
            c_true: torch.Tensor,  # (B,1) {0, 1}
            c_hat: torch.Tensor,  # (B,1)
    ) -> torch.Tensor:
        """
        The prediction loss trains the decoder and reward predictor via the symlog loss
        and the continue predictor via binary classification loss.
        """
        # Image reconstruction
        img_loss = 0.5 * (x_hat - x_true).pow(2).mean(dim=(-3, -2, -1))  # (B,)

        # Reward in symlog space
        rew_loss = 0.5 * (r_hat - symlog(r_true)).pow(2).mean(dim=-1)  # (B,)

        # Continue flag
        cont_loss = F.binary_cross_entropy_with_logits(c_hat, c_true, reduction="none").squeeze(-1)  # (B,)

        return img_loss + rew_loss + cont_loss

    def _dynamics_loss(
            self,
            post_logits: torch.Tensor,  # (B,L,K)
            prior_logits: torch.Tensor,  # (B,L,K)
    ) -> torch.Tensor:
        """
        The dynamics loss trains the sequence model to predict the next representation
        by minimizing the KL divergence between the predictor and the next stochastic representation.
        """
        post_logits_detached = post_logits.detach()  # stop gradient to the encoder

        kl = self._kl_loss(post_logits_detached, prior_logits)
        return kl

    def _representation_loss(
            self,
            post_logits: torch.Tensor,  # (B,L,K)
            prior_logits: torch.Tensor,  # (B,L,K)
    ) -> torch.Tensor:
        """
        The representation loss trains the representations to become more predictable if the dynamics
        cannot predict their distribution, allowing us to use a factorized dynamics predictor for fast sampling
        when training the actor and critic.
        """
        prior_logits_detached = prior_logits.detach()  # stop gradient to the dynamics predictor

        kl = self._kl_loss(post_logits, prior_logits_detached)
        return kl

    def loss(
            self,
            obs: torch.Tensor,  # (B,T,C,H,W)
            actions: torch.Tensor,  # (B,T) action indices
            rewards: torch.Tensor,  # (B,T)
            continues: torch.Tensor,  # (B,T) {0, 1}
            init_state: Optional[WorldModelState] = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Roll out a subsequence of length T and compute DreamerV3 world-model loss.
        Returns (total_loss, metrics_dict).
        """
        B, T = obs.size(0), obs.size(1)

        # prev actions a_{t-1}: left-shift, inject 0 for t=0
        a_prev = torch.roll(actions, 1, dims=1)
        a_prev[:, 0] = 0

        # Add dim for rewards to be (B,T,1)
        rewards = rewards.unsqueeze(-1).contiguous()

        # Add dim for continues to be (B,T,1)
        continues = continues.float().unsqueeze(-1).contiguous()

        # At t=0 there is no boundary
        c_prev = torch.ones(B, 1, device=obs.device)

        state = init_state if init_state is not None else self.init_state(B, device=obs.device, dtype=obs.dtype)

        pred_sum = torch.zeros((), device=obs.device, dtype=obs.dtype)
        dyn_sum = torch.zeros((), device=obs.device, dtype=obs.dtype)
        rep_sum = torch.zeros((), device=obs.device, dtype=obs.dtype)

        for t in range(T):
            # one RSSM step
            state, info = self.step(
                state_prev=state,
                a_prev_idx=a_prev[:, t],
                x_cur=obs[:, t],
                c_prev=c_prev,
            )

            # prediction loss
            pred_sum = pred_sum + self._prediction_loss(
                x_true=obs[:, t],
                x_hat=info["x_hat"],
                r_true=rewards[:, t],
                r_hat=info["r_hat"],
                c_true=continues[:, t],
                c_hat=info["c_hat"],
            ).mean()

            # dynamics loss
            dyn_sum = dyn_sum + self._dynamics_loss(info["post_logits"], info["prior_logits"]).mean()

            # representation loss
            rep_sum = rep_sum + self._representation_loss(info["post_logits"], info["prior_logits"]).mean()

            # continue flag and x_prev for next step
            c_prev = continues[:, t]

        # average over time
        pred = pred_sum / T
        dyn = dyn_sum / T
        rep = rep_sum / T

        total_loss = self.beta_pred * pred + self.beta_dyn * dyn + self.beta_rep * rep
        return total_loss, {
            "total_loss": total_loss,
            "pred_loss": pred,
            "dyn_loss": dyn,
            "rep_loss": rep,
            "state": state,
        }
