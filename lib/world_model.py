from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, kl_divergence

from lib.utils import symlog, log_unimix


@dataclass
class WorldModelState:
    h: torch.Tensor  # (B, h_dim)
    z: torch.Tensor  # (B, num_latents, classes_per_latent)


class SequenceModel(nn.Module):
    """
    Sequence model with recurrent state predicts next state h_t
    given current state h_{t-1}, stochastic representation z_{t-1}, and action a_{t-1}.
    Part of the World Model.
    """

    def __init__(
            self,
            action_size: int,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden: int = 512,
    ):
        super().__init__()
        self.action_size = action_size
        in_dim = h_dim + num_latents * classes_per_latent + action_size
        self.input_layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
        )
        self.rnn = nn.GRUCell(hidden, h_dim)

    def forward(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: (B, h_dim)
            z_prev: (B, num_latents, classes_per_latent)
            a_prev_idx: (B,) with values in [0, action_size-1]

        Returns:
            h_t: (B, h_dim)
        """
        a_onehot = F.one_hot(a_prev_idx, num_classes=self.action_size).float()
        z_flat = z_prev.reshape(z_prev.size(0), -1)
        x = torch.cat([h_prev, z_flat, a_onehot], dim=-1)
        x = self.input_layer(x)
        return self.rnn(x, h_prev)


class Encoder(nn.Module):
    """
    Encoder that maps sensory inputs x_t to stochastic representations z_t.
    The posterior predictor that predicts z_t given h_t and x_t -> will act as a teacher for the dynamics predictor.
    Part of the World Model.
    Expects inputs to be 64x64 images.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_latents: int = 32,
            classes_per_latent: int = 32,
            h_dim: int = 512,
            cnn_multiplier: int = 32,
            hidden: int = 512,
    ):
        super().__init__()
        C, H, W = obs_shape
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv = nn.Sequential(
            nn.Conv2d(C, cnn_multiplier, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(cnn_multiplier, cnn_multiplier * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(cnn_multiplier * 2, cnn_multiplier * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(cnn_multiplier * 4, cnn_multiplier * 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out_size + h_dim),
            nn.Linear(conv_out_size + h_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_latents * classes_per_latent),
        )
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    @torch.no_grad()
    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(o.nelement() / o.size(0))

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, C, H, W)
            h_t: (B, h_dim)

        Returns:
            logits over classes per latent: (B, num_latents, classes_per_latent)
        """
        B = x_t.size(0)
        assert x_t.dim() == 4 and x_t.shape == (B, 3, 64, 64)
        conv_out = self.conv(x_t)  # (B, conv_out)
        combined = torch.cat([conv_out, h_t], dim=-1)  # (B, conv_out + h_dim)
        logits = self.fc(combined).view(-1, self.num_latents, self.classes_per_latent)
        return logits


class DynamicsPredictor(nn.Module):
    """
    Predictor that predicts next stochastic representation z_t.
    The prior predictor that predicts z_t given h_t -> learns to model the dynamics of the environment.
    Part of the World Model.
    """

    def __init__(
            self,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_latents * classes_per_latent)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, h_dim)

        Returns:
            logits over classes per latent: (B, num_latents, classes_per_latent)
        """
        mlp_out = self.mlp(h_t)  # (B, dense_hidden_units)
        logits = self.head(mlp_out).view(-1, self.num_latents, self.classes_per_latent)
        return logits


class RewardPredictor(nn.Module):
    """
    Predictor that predicts reward r_t with the concatenation of h_t and z_t.
    """

    def __init__(
            self,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            hidden: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim + num_latents * classes_per_latent
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, h_dim)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            reward r_t: (B, 1)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)  # (B, num_latents * classes_per_latent)
        combined = torch.cat([h_t, z_t_flat], dim=-1)  # (B, h_dim + num_latents * classes_per_latent)
        mlp_out = self.mlp(combined)  # (B, dense_hidden_units)
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
            hidden: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim + num_latents * classes_per_latent
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.SiLU())
            dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        self.num_latents, self.classes_per_latent = num_latents, classes_per_latent

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, h_dim)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            continuation flag logits c_t: (B, 1)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)  # (B, num_latents * classes_per_latent)
        combined = torch.cat([h_t, z_t_flat], dim=-1)  # (B, h_dim + num_latents * classes_per_latent)
        mlp_out = self.mlp(combined)  # (B, dense_hidden_units)
        continue_logit = self.head(mlp_out)  # (B, 1)
        return continue_logit


class Decoder(nn.Module):
    """
    Decoder that reconstructs sensory inputs x_t from the concatenation of h_t and z_t.
    Returns 64x64 images.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            num_latents: int = 32,
            classes_per_latent: int = 32,
            h_dim: int = 512,
            cnn_multiplier: int = 32,
            hidden: int = 512,
    ):
        super().__init__()
        C, H, W = obs_shape

        self.fc = nn.Sequential(
            nn.LayerNorm(h_dim + num_latents * classes_per_latent),
            nn.Linear(h_dim + num_latents * classes_per_latent, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cnn_multiplier * 8 * 4 * 4),
            nn.SiLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(cnn_multiplier * 8, cnn_multiplier * 4, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(cnn_multiplier * 4, cnn_multiplier * 2, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(cnn_multiplier * 2, cnn_multiplier, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(cnn_multiplier, C, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, h_dim)
            z_t: (B, num_latents, classes_per_latent)

        Returns:
            reconstructed x_t: (B, C, H, W)
        """
        B = h_t.size(0)
        z_t_flat = z_t.view(B, -1)  # (B, num_latents * classes_per_latent)
        combined = torch.cat([h_t, z_t_flat], dim=-1)  # (B, h_dim + num_latents * classes_per_latent)
        fc_out = self.fc(combined)  # (B, cnn_multiplier * 8 * 4 * 4)
        deconv_input = fc_out.view(B, -1, 4, 4)  # (B, cnn_multiplier * 8, 4, 4)
        x_recon = self.deconv(deconv_input)  # (B, C, H, W)
        assert x_recon.dim() == 4 and x_recon.shape == (B, 3, 64, 64)
        return x_recon


class WorldModel(nn.Module):
    """
    The World Model is a RSSM that combines the above components.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_size: int,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            cnn_multiplier: int = 32,
            hidden: int = 512,
            depth: int = 2,
            free_bits: float = 1.0,
            beta_pred: float = 1.0,
            beta_dyn: float = 0.5,
            beta_rep: float = 0.1,
            unimix_eps: float = 0.01,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.h_dim = h_dim
        self.num_latents = num_latents
        self.classes_per_latent = classes_per_latent
        self.free_bits = free_bits
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.unimix_eps = unimix_eps

        self.seq = SequenceModel(
            action_size=action_size,
            h_dim=h_dim,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden=hidden,
        )
        self.enc = Encoder(
            obs_shape=obs_shape,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            h_dim=h_dim,
            cnn_multiplier=cnn_multiplier,
            hidden=hidden,
        )
        self.dyn = DynamicsPredictor(
            h_dim=h_dim,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden=hidden,
            depth=depth,
        )
        self.rew = RewardPredictor(
            h_dim=h_dim,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden=hidden,
            depth=depth,
        )
        self.cont = ContinuePredictor(
            h_dim=h_dim,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            hidden=hidden,
            depth=depth,
        )
        self.dec = Decoder(
            obs_shape=obs_shape,
            num_latents=num_latents,
            classes_per_latent=classes_per_latent,
            h_dim=h_dim,
            cnn_multiplier=cnn_multiplier,
            hidden=hidden,
        )

        # zero-init reward and continue predictors to avoid large early prediction errors
        nn.init.zeros_(self.rew.head.weight)
        nn.init.zeros_(self.rew.head.bias)
        nn.init.zeros_(self.cont.head.weight)
        nn.init.zeros_(self.cont.head.bias)

    @torch.no_grad()
    def init_state(self, batch_size: int, device=None, dtype=None) -> WorldModelState:
        h0 = torch.zeros(batch_size, self.h_dim, device=device, dtype=dtype)
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

        if c_prev is not None:
            # reset carry-over across episode boundaries
            mask_h = c_prev  # (B,1)
            mask_z = c_prev.view(-1, 1, 1)  # (B,1,1) to broadcast over (L,K)
            h_prev = h_prev * mask_h
            z_prev = z_prev * mask_z

        # The sequence model updates the recurrent state h_t
        h_cur = self.seq(h_prev, z_prev, a_prev_idx)  # (B, h_dim)

        # The encoder maps sensory input x_t to posterior logits
        z_logits_post = self.enc(x_cur, h_cur) if x_cur is not None else None  # (B, L, K) or None

        # The dynamics predictor predicts prior logits from h_t
        z_logits_prior = self.dyn(h_cur)  # (B, L, K)

        # The logits for z_t are from the encoder if available, else from the dynamics predictor
        z_logits = z_logits_post if z_logits_post is not None else z_logits_prior  # (B, L, K)

        # Sample z_t
        z = self._sample_z(z_logits, hard=True)  # (B, L, K)

        # Heads (reconstruction, reward, continue)
        r_hat = self.rew(h_cur, z)  # (B,1)
        c_hat = self.cont(h_cur, z)  # (B,1)
        x_hat = self.dec(h_cur, z)  # (B,C,H,W)

        new_state = WorldModelState(h=h_cur, z=z)
        info: Dict[str, Any] = {
            "prior_logits": z_logits_prior,
            "post_logits": z_logits_post,  # None in imagination
            "r_hat": r_hat,  # (B,1) in symlog space
            "c_hat": c_hat,  # (B,1) logits
            "x_hat": x_hat,  # (B,C,H,W) in symlog space
        }
        return new_state, info

    @staticmethod
    def _mse_symlog(pred: torch.Tensor, target: torch.Tensor, reduce_over: Tuple[int, ...]) -> torch.Tensor:
        """
        Mean squared error in symlog space.
        """
        squared_error = (pred - symlog(target)) ** 2
        return 0.5 * squared_error.sum(dim=reduce_over)

    def _sample_z(self, logits: torch.Tensor, hard: bool = True) -> torch.Tensor:
        log_prob = log_unimix(logits, self.unimix_eps, dim=-1)
        return F.gumbel_softmax(log_prob, hard=hard, dim=-1)

    def _kl_categorical_logits(self, q_logits: torch.Tensor, p_logits: torch.Tensor) -> torch.Tensor:
        """
        KL[ q || p ] for factorized categoricals over (L,K), using unimix on both.
        Sum over K and L -> (B,).
        """
        K = q_logits.size(-1)

        # Unimix both sides: (1 - eps)*softmax + eps/K
        q_probs = (1.0 - self.unimix_eps) * F.softmax(q_logits, dim=-1) + self.unimix_eps / float(K)
        p_probs = (1.0 - self.unimix_eps) * F.softmax(p_logits, dim=-1) + self.unimix_eps / float(K)

        # Build distributions; shapes broadcast over (B, L)
        q = Categorical(probs=q_probs)
        p = Categorical(probs=p_probs)

        # KL per (B, L), then sum over L
        kl = kl_divergence(q, p).sum(dim=-1)  # -> (B,)
        return kl

    def _prediction_loss(
            self,
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
        # MSE loss in symlog space for image and reward
        # (mean over C,H,W for image, mean over 1 for reward)
        img_loss = self._mse_symlog(x_hat, x_true, reduce_over=(-3, -2, -1))
        rew_loss = self._mse_symlog(r_hat, r_true, reduce_over=(-1,))

        # Binary cross-entropy loss for continue flag
        cont_loss = F.binary_cross_entropy_with_logits(c_hat, c_true, reduction="none").squeeze(-1)

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

        kl = self._kl_categorical_logits(post_logits_detached, prior_logits)
        kl = torch.clamp(kl, min=self.free_bits)  # free bits
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

        kl = self._kl_categorical_logits(post_logits, prior_logits_detached)
        kl = torch.clamp(kl, min=self.free_bits)  # free bits
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

        # -------------------
        # TODO: set action to 0 where continue is 0 (maybe handle it in the sequence forward function instead)
        # a_onehot = F.one_hot(a_prev_idx, num_classes=self.action_size).float()  # (B,A)
        # if c_prev is not None:
        #     a_onehot = a_onehot * c_prev  # (B,1) -> zero vector when starting new episode
        # z_flat = z_prev.reshape(z_prev.size(0), -1)
        # x = torch.cat([h_prev, z_flat, a_onehot], dim=-1)
        # x = self.input_layer(x)
        # return self.rnn(x, h_prev)
        # -------------------

        # prev actions a_{t-1}: left-shift, inject 0 for t=0
        a_prev = torch.zeros(B, T, dtype=torch.long, device=obs.device)
        a_prev[:, 1:] = actions[:, :-1]

        # Add dim for rewards to be (B,T,1)
        rewards = rewards.unsqueeze(-1)

        # Add dim for continues to be (B,T,1)
        continues = continues.float().unsqueeze(-1)

        # At t=0 there is no boundary
        c_prev = torch.ones(B, 1, device=obs.device)

        state = init_state if init_state is not None else self.init_state(B, device=obs.device, dtype=obs.dtype)

        pred_losses = []
        dyn_losses = []
        rep_losses = []

        for t in range(T):
            # one RSSM step
            state, info = self.step(
                state_prev=state,
                a_prev_idx=a_prev[:, t],
                x_cur=obs[:, t],
                c_prev=c_prev,
            )

            # prediction loss
            pred_loss = self._prediction_loss(
                x_true=obs[:, t],  # (B,C,H,W)
                x_hat=info["x_hat"],  # (B,C,H,W)
                r_true=rewards[:, t],  # (B,1)
                r_hat=info["r_hat"],  # (B,1)
                c_true=continues[:, t],  # (B,1)
                c_hat=info["c_hat"],  # (B,1)
            )
            pred_losses.append(pred_loss)

            # dynamics loss
            dyn_loss = self._dynamics_loss(
                post_logits=info["post_logits"],
                prior_logits=info["prior_logits"],
            )
            dyn_losses.append(dyn_loss)

            # representation loss
            rep_loss = self._representation_loss(
                post_logits=info["post_logits"],
                prior_logits=info["prior_logits"],
            )
            rep_losses.append(rep_loss)

            # continue flag for next step
            c_prev = continues[:, t]

        # Stack losses over time -> (T,B), then mean over time and batch -> (1,)
        pred = torch.stack(pred_losses, dim=0).mean()
        dyn = torch.stack(dyn_losses, dim=0).mean()
        rep = torch.stack(rep_losses, dim=0).mean()

        total_loss = self.beta_pred * pred + self.beta_dyn * dyn + self.beta_rep * rep
        tensor_dict: Dict[str, Any] = {
            "total_loss": total_loss,
            "pred_loss": pred,
            "dyn_loss": dyn,
            "rep_loss": rep,
            "state": state,
        }
        return total_loss, tensor_dict
