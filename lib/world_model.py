from typing import Tuple

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """
    Sequence model with recurrent state predicts next state h_t
    given current state h_{t-1}, stochastic representation z_{t-1}, and action a_{t-1}.
    Part of the World Model.
    """

    def __init__(
            self,
            action_dim: int,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            dense_hidden_units: int = 512,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.LayerNorm(h_dim + num_latents * classes_per_latent + action_dim),
            nn.Linear(h_dim + num_latents * classes_per_latent + action_dim, dense_hidden_units),
            nn.SiLU(inplace=True),
        )
        self.rnn = nn.GRUCell(dense_hidden_units, h_dim)

    def forward(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: (B, h_dim)
            z_prev: (B, num_latents, classes_per_latent)
            a_prev: (B, action_dim)

        Returns:
            h_t: (B, h_dim)
        """
        B = h_prev.size(0)
        z_prev_flat = z_prev.view(B, -1)
        rnn_input = torch.cat([h_prev, z_prev_flat, a_prev], dim=-1)
        input_layer_out = self.input_layer(rnn_input)
        h_t = self.rnn(input_layer_out, h_prev)
        return h_t


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
            dense_hidden_units: int = 512,
    ):
        super().__init__()
        C, H, W = obs_shape
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv = nn.Sequential(
            nn.Conv2d(C, cnn_multiplier, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(cnn_multiplier, cnn_multiplier * 2, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(cnn_multiplier * 2, cnn_multiplier * 4, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(cnn_multiplier * 4, cnn_multiplier * 8, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.LayerNorm(conv_out_size + h_dim),
            nn.Linear(conv_out_size + h_dim, dense_hidden_units),
            nn.SiLU(inplace=True),
            nn.Linear(dense_hidden_units, num_latents * classes_per_latent),
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
            dense_hidden_units: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, dense_hidden_units))
            layers.append(nn.SiLU(inplace=True))
            dim = dense_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dense_hidden_units, num_latents * classes_per_latent)
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


class WorldModel(nn.Module):
    """
    The World Model is a RSSM that consists of a sequence model, an encoder, and a dynamics predictor.
    """
    raise NotImplementedError


class RewardPredictor(nn.Module):
    """
    Predictor that predicts reward r_t with the concatenation of h_t and z_t.
    """

    def __init__(
            self,
            h_dim: int = 512,
            num_latents: int = 32,
            classes_per_latent: int = 32,
            dense_hidden_units: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim + num_latents * classes_per_latent
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, dense_hidden_units))
            layers.append(nn.SiLU(inplace=True))
            dim = dense_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dense_hidden_units, 1)
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
            dense_hidden_units: int = 512,
            depth: int = 2,
    ):
        super().__init__()
        layers = []
        dim = h_dim + num_latents * classes_per_latent
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, dense_hidden_units))
            layers.append(nn.SiLU(inplace=True))
            dim = dense_hidden_units
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dense_hidden_units, 1)
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
            dense_hidden_units: int = 512,
    ):
        super().__init__()
        C, H, W = obs_shape

        self.fc = nn.Sequential(
            nn.LayerNorm(h_dim + num_latents * classes_per_latent),
            nn.Linear(h_dim + num_latents * classes_per_latent, dense_hidden_units),
            nn.SiLU(inplace=True),
            nn.Linear(dense_hidden_units, cnn_multiplier * 8 * 4 * 4),
            nn.SiLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(cnn_multiplier * 8, cnn_multiplier * 4, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(cnn_multiplier * 4, cnn_multiplier * 2, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(cnn_multiplier * 2, cnn_multiplier, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(inplace=True),
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
        return x_recon
