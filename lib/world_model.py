from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.nn_blocks import MLP


class Encoder(nn.Module):
    """Encode image or vector observations into a fixed‑size embedding.

    If the observation has three dimensions it is assumed to be an image in
    (C, H, W) format and a small convolutional network is used.  Otherwise a
    simple multilayer perceptron (MLP) is applied.  Observations are scaled
    to the range ``[0, 1]`` for images.
    """

    def __init__(self, obs_shape: Tuple[int, ...], embed_size: int = 1024) -> None:
        super().__init__()
        self.embed_size = embed_size
        if len(obs_shape) == 3:
            c, h, w = obs_shape
            self.is_image = True
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=4, stride=2),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.SiLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.SiLU(inplace=True),
            )
            conv_out_dim = self._get_conv_out(obs_shape)
            self.fc = nn.Linear(conv_out_dim, embed_size)
        else:
            # Vector observation
            self.is_image = False
            self.fc = nn.Sequential(
                nn.Linear(obs_shape[0], embed_size),
                nn.SiLU(inplace=True),
                nn.Linear(embed_size, embed_size),
                nn.SiLU(inplace=True),
            )

    @torch.no_grad()
    def _get_conv_out(self, shape: Tuple[int, ...]) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(o.nelement() / o.size(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations.

        Args:
            x: A tensor of shape ``(B, *obs_shape)`` or ``(B, T, *obs_shape)``.

        Returns:
            A tensor of shape ``(B, T, embed_size)`` if a sequence was given,
            otherwise ``(B, embed_size)``.
        """
        if self.is_image:
            x = x.float() / 255.0
            if x.dim() == 5:
                b, t, c, h, w = x.shape
                x_flat = x.view(b * t, c, h, w)
                h = self.conv(x_flat)
                h = h.view(b * t, -1)
                h = self.fc(h)
                return h.view(b, t, -1)
            else:
                h = self.conv(x)
                h = h.view(h.size(0), -1)
                return self.fc(h)
        else:
            # Vector input
            if x.dim() == 3:
                b, t, d = x.shape
                h = x.view(b * t, d)
                h = self.fc(h)
                return h.view(b, t, -1)
            else:
                return self.fc(x)


class RSSMPrior(nn.Module):
    """
    Prior network of the recurrent state space model.

    Predicts the next latent state and deterministic belief given the current
    stochastic latent state and an action.  The design follows the TorchRL
    implementation but uses plain PyTorch modules.  A small MLP processes
    the concatenated latent and action before a GRUCell updates the belief.
    """

    def __init__(
            self,
            stoch_size: int,
            action_size: int,
            deter_size: int = 200,
            hidden_size: int | None = None,
            scale_lb: float = 0.1,
    ) -> None:
        super().__init__()
        self.stoch_size = stoch_size
        self.action_size = action_size
        self.deter_size = deter_size
        self.scale_lb = scale_lb
        hid = hidden_size or deter_size
        # MLP to project [z, a] into hidden space for the GRU
        self.action_state_projector = nn.Sequential(
            nn.Linear(stoch_size + action_size, hid),
            nn.ELU(inplace=True),
        )
        self.rnn = nn.GRUCell(hid, deter_size)
        # MLP to map belief to Gaussian parameters
        self.rnn_to_prior_projector = nn.Sequential(
            nn.Linear(deter_size, hid),
            nn.ELU(inplace=True),
            nn.Linear(hid, 2 * stoch_size),
        )

    def forward(
            self,
            stoch: torch.Tensor,
            belief: torch.Tensor,
            action: torch.Tensor,
    ) -> tuple[Any, Tensor, Tensor, Any]:
        """
        One step forward of the prior.

        Args:
            stoch: Current stochastic latent state ``(B, stoch_size)``.
            belief: Current deterministic belief ``(B, deter_size)``.
            action: Action taken at the current time step ``(B, action_size)``.

        Returns:
            A tuple ``(mean, std, next_stoch)``, where ``mean`` and ``std`` are
            the parameters of the predicted distribution of the next stochastic
            latent state and ``next_stoch`` is a sample drawn using the
            reparameterisation trick.
        """
        # Project concatenated [z, a] to GRU input
        x = torch.cat([stoch, action], dim=-1)
        h = self.action_state_projector(x)
        belief = self.rnn(h, belief)
        params = self.rnn_to_prior_projector(belief)
        mean, log_std = params.chunk(2, dim=-1)
        # Softplus to ensure positivity; clamp to lower bound
        std = F.softplus(log_std) + self.scale_lb
        next_stoch = mean + torch.randn_like(std) * std
        return mean, std, next_stoch, belief


class RSSMPosterior(nn.Module):
    """
    Posterior network of the recurrent state space model.

    Combines the current belief and the observation embedding to refine the
    stochastic latent state.  A small MLP produces the mean and standard
    deviation of the posterior distribution.
    """

    def __init__(
            self,
            deter_size: int,
            embed_size: int,
            stoch_size: int = 30,
            hidden_size: int | None = None,
            scale_lb: float = 0.1,
    ) -> None:
        super().__init__()
        hid = hidden_size or deter_size
        self.deter_size = deter_size
        self.embed_size = embed_size
        self.stoch_size = stoch_size
        self.scale_lb = scale_lb
        self.mlp = nn.Sequential(
            nn.Linear(deter_size + embed_size, hid),
            nn.ELU(inplace=True),
            nn.Linear(hid, 2 * stoch_size),
        )

    def forward(self, belief: torch.Tensor, embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.mlp(torch.cat([belief, embed], dim=-1))
        mean, log_std = params.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.scale_lb
        stoch = mean + torch.randn_like(std) * std
        return mean, std, stoch


class RSSM(nn.Module):
    """
    Container for prior and posterior models in the recurrent state space model.

    This class orchestrates the computation of prior and posterior sequences.  It
    exposes ``init_state``, ``observe`` and ``imagine`` methods for use in
    training and policy/value optimisation.
    """

    def __init__(
            self,
            action_size: int,
            embed_size: int,
            deter_size: int = 200,
            stoch_size: int = 30,
            scale_lb: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.embed_size = embed_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.prior = RSSMPrior(stoch_size, action_size, deter_size, scale_lb=scale_lb)
        self.posterior = RSSMPosterior(deter_size, embed_size, stoch_size, scale_lb=scale_lb)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        return {
            "deter": torch.zeros(batch_size, self.deter_size, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
        }

    def observe(
            self,
            embeds: torch.Tensor,
            actions: torch.Tensor,
            state: Dict[str, torch.Tensor],
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Compute posterior and prior sequences given embeddings and actions.

        Args:
            embeds: Observation embeddings of shape ``(B, T, embed_size)``.
            actions: Actions of shape ``(B, T, action_size)`` (one‑hot or
                continuous).
            state: Initial state dict with keys ``"deter"`` and ``"stoch"``.

        Returns:
            ``posteriors`` and ``priors``, each a list of state dictionaries of
            length ``T`` with keys ``deter``, ``stoch``, ``mean`` and ``std``.
        """
        b, t, _ = embeds.shape
        deter = state["deter"]
        stoch = state["stoch"]
        posteriors: List[Dict[str, torch.Tensor]] = []
        priors: List[Dict[str, torch.Tensor]] = []
        for i in range(t):
            a = actions[:, i]
            # Prior update
            prior_mean, prior_std, stoch_prior, deter = self.prior(stoch, deter, a)
            priors.append({
                "deter": deter,
                "stoch": stoch_prior,
                "mean": prior_mean,
                "std": prior_std,
            })
            # Posterior update uses embedding
            e = embeds[:, i]
            post_mean, post_std, stoch_post = self.posterior(deter, e)
            posteriors.append({
                "deter": deter,
                "stoch": stoch_post,
                "mean": post_mean,
                "std": post_std,
            })
            stoch = stoch_post
        return posteriors, priors

    def imagine(
            self,
            actions: torch.Tensor,
            state: Dict[str, torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Roll out prior states given actions and an initial state.

        Args:
            actions: Tensor of shape ``(B, T, action_size)``.
            state: Dict with keys ``"deter"`` and ``"stoch"``.

        Returns:
            List of prior state dicts of length ``T``.
        """
        b, t, _ = actions.shape
        deter = state["deter"]
        stoch = state["stoch"]
        priors: List[Dict[str, torch.Tensor]] = []
        for i in range(t):
            a = actions[:, i]
            mean, std, stoch, deter = self.prior(stoch, deter, a)
            priors.append({
                "deter": deter,
                "stoch": stoch,
                "mean": mean,
                "std": std,
            })
        return priors


class Decoder(nn.Module):
    """
    Observation decoder mapping latent features back to observations.

    For image observations the decoder uses a series of transposed
    convolutions.  For vector observations it uses a multi‑layer perceptron.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            deter_size: int,
            stoch_size: int,
            hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        self.is_image = len(obs_shape) == 3
        input_size = deter_size + stoch_size
        if self.is_image:
            c, h, w = obs_shape
            # Start from a small spatial map and upsample to the desired size
            self.fc = nn.Linear(input_size, 256 * 2 * 2)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(16, c, kernel_size=4, stride=2, padding=1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_size, obs_shape[0]),
            )

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, stoch], dim=-1)
        if self.is_image:
            h = self.fc(x)
            h = h.view(h.size(0), 256, 2, 2)
            return self.deconv(h)
        else:
            return self.mlp(x)


class RewardPredictor(nn.Module):
    def __init__(self, deter_size: int, stoch_size: int, units: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.mlp = MLP(deter_size + stoch_size, 1, units=units, depth=depth)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([deter, stoch], dim=-1)).squeeze(-1)


class ContinuePredictor(nn.Module):
    def __init__(self, deter_size: int, stoch_size: int, units: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.mlp = MLP(deter_size + stoch_size, 1, units=units, depth=depth)

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([deter, stoch], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_size: int,
                 embed_size: int = 1024, deter_size: int = 200, stoch_size: int = 30,
                 free_nats: float = 1.0, beta_pred: float = 1.0, beta_dyn: float = 0.5, beta_rep: float = 0.1,
                 units: int = 256, depth: int = 4) -> None:
        super().__init__()
        self.encoder = Encoder(obs_shape, embed_size)
        self.rssm = RSSM(action_size, embed_size, deter_size, stoch_size)
        self.decoder = Decoder(obs_shape, deter_size, stoch_size, embed_size)
        self.reward_predictor = RewardPredictor(deter_size, stoch_size, units=units, depth=depth)
        self.continue_predictor = ContinuePredictor(deter_size, stoch_size, units=units, depth=depth)
        self.free_nats = free_nats
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.kl_balance = 0.8  # common Dreamer balance
        self.deter_size = deter_size
        self.stoch_size = stoch_size

    def init_state(self, batch_size: int, device: torch.device | None = None) -> Dict[str, torch.Tensor]:
        return self.rssm.init_state(batch_size, device)

    @staticmethod
    def get_feat(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    @staticmethod
    def _kl_gauss(m1, s1, m2, s2):
        s1 = s1 + 1e-8
        s2 = s2 + 1e-8
        term = torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
        return term.sum(-1)

    def _kl_split(self, mean_post, std_post, mean_prior, std_prior):
        # dynamics: KL(sg(q) || p), representation: KL(q || sg(p))
        kl_dyn = self._kl_gauss(mean_post.detach(), std_post.detach(), mean_prior, std_prior)
        kl_rep = self._kl_gauss(mean_post, std_post, mean_prior.detach(), std_prior.detach())
        # free bits (per-step then mean)
        fb = self.free_nats
        return torch.clamp(kl_dyn, min=fb).mean(), torch.clamp(kl_rep, min=fb).mean()

    def imagine(self, actions: torch.Tensor, start_state: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, List[Dict[str, torch.Tensor]]]:
        priors = self.rssm.imagine(actions, start_state)
        feats = [self.get_feat(s) for s in priors]
        return torch.stack(feats, dim=1), priors

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, continues: torch.Tensor,
             state: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        embeds = self.encoder(obs)
        post, prior = self.rssm.observe(embeds, actions, state)

        deter_post = torch.stack([s["deter"] for s in post], dim=1)
        stoch_post = torch.stack([s["stoch"] for s in post], dim=1)

        b, t = deter_post.shape[:2]
        flat_deter = deter_post.reshape(b * t, -1)
        flat_stoch = stoch_post.reshape(b * t, -1)

        recon = self.decoder(flat_deter, flat_stoch)
        if recon.dim() == 4:
            recon = recon.view(b, t, *recon.shape[1:])
            recon_loss = F.mse_loss(recon, obs.float(), reduction="none").mean(dim=(2, 3, 4)).mean()
        else:
            recon = recon.view(b, t, -1)
            recon_loss = F.mse_loss(recon, obs.float(), reduction="none").mean(dim=2).mean()

        reward_pred = self.reward_predictor(flat_deter, flat_stoch).view(b, t)
        reward_loss = F.mse_loss(reward_pred, rewards, reduction="none").mean()

        cont_pred = self.continue_predictor(flat_deter, flat_stoch).view(b, t)
        cont_loss = F.binary_cross_entropy_with_logits(cont_pred, continues, reduction="none").mean()

        mean_post = torch.stack([s["mean"] for s in post], dim=1).view(b * t, -1)
        std_post = torch.stack([s["std"] for s in post], dim=1).view(b * t, -1)
        mean_prior = torch.stack([s["mean"] for s in prior], dim=1).view(b * t, -1)
        std_prior = torch.stack([s["std"] for s in prior], dim=1).view(b * t, -1)

        kl_dyn, kl_rep = self._kl_split(mean_post, std_post, mean_prior, std_prior)

        pred_loss = recon_loss + reward_loss + cont_loss  # continue folded into prediction
        total = self.beta_pred * pred_loss + self.beta_dyn * kl_dyn + self.beta_rep * kl_rep
        return total, post, prior
