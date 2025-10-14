"""
World model implementation for DreamerV3.

The world model is the key component of DreamerV3 that provides rich
latent representations and enables imagination through self‑supervised
learning.  DreamerV3 uses a recurrent state‑space model (RSSM) where a
deterministic recurrent state is combined with a stochastic latent state
to form a Markovian representation【145968576409203†L293-L344】.  Given an input
observation and an action, the model encodes the observation into a
stochastic latent, updates the deterministic hidden state, predicts the
next latent distribution (prior), and reconstructs the input as well as
predicting the reward and whether an episode continues【145968576409203†L293-L351】.

This module implements a simplified but faithful version of the world
model in PyTorch.  It supports both RGB image inputs and low‑dimensional
vector observations.  The model consists of the following components:

* Encoder: maps raw observations into feature vectors using a small
  convolutional network for images and an MLP for vectors.
* RecurrentStateSpaceModel: maintains a deterministic recurrent state and
  stochastic latent state; provides methods to perform both observation
  (posterior) and imagination (prior) updates【145968576409203†L293-L351】.
* Decoder: reconstructs observations from the latent and hidden state.
* RewardPredictor: predicts the immediate reward from the latent and
  hidden state.
* ContinuePredictor: predicts whether an episode continues (1 – done flag).

The world model exposes a forward method for computing losses from
sequences of observations, actions, rewards, and continuation flags, and
a method for rolling out imagined trajectories given start states and a
sequence of actions.  The losses include reconstruction error, KL
divergence between prior and posterior latents, reward prediction loss,
and continuation prediction loss, as described in the paper【145968576409203†L293-L365】.

Note: To keep the code concise and readable, certain architectural
choices (such as discrete latents and the symlog loss) from the full
DreamerV3 implementation are approximated here.  The encoder and
decoder architectures are modest and may need to be scaled up for
large‑scale experiments.
"""
from __future__ import annotations

from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    """Encode observations into feature vectors.

    For image inputs, we employ a simple convolutional stack similar to
    the one used in the Dreamer paper: four convolutional layers with
    increasing channel counts and stride 2.  For vector inputs, a two‑layer
    MLP is used.
    """

    def __init__(self, obs_shape: Tuple[int, ...], embed_size: int = 1024):
        super().__init__()
        self.obs_shape = obs_shape
        self.embed_size = embed_size
        if len(obs_shape) == 3:
            c, h, w = obs_shape
            self.is_image = True
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2),
                nn.ReLU(),
            )
            # compute output size
            with torch.no_grad():
                x = torch.zeros(1, *obs_shape)
                out = self.conv(x)
                self.conv_out_size = out.view(1, -1).size(1)
            self.fc = nn.Linear(int(self.conv_out_size), embed_size)
        else:
            self.is_image = False
            self.fc = nn.Sequential(
                nn.Linear(obs_shape[0], embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.is_image:
            x = obs.float() / 255.0
            # convert to NCHW
            if x.ndim == 5:  # sequence batch: (B, T, C, H, W)
                b, t, c, h, w = x.size()
                x = x.view(-1, c, h, w)
                out = self.conv(x)
                out = out.reshape(b, t, -1)
                out = self.fc(out)
                return out
            else:
                out = self.conv(x)
                out = out.reshape(out.size(0), -1)
                out = self.fc(out)
                return out
        else:
            return self.fc(obs.float())


class Decoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], deter_size: int, stoch_size: int, embed_size: int = 1024):
        super().__init__()
        self.obs_shape = obs_shape
        self.is_image = len(obs_shape) == 3
        input_size = deter_size + stoch_size
        if self.is_image:
            c, h, w = obs_shape

            # We will upsample by 2 four times: start_size * 2^4 == target_size
            # so start_h = h // 16, start_w = w // 16 (assumes divisibility).
            assert h % 16 == 0 and w % 16 == 0, \
                f"Decoder expects H and W divisible by 16, got {(h, w)}"
            start_h, start_w = h // 16, w // 16

            self.fc = nn.Linear(input_size, 256 * start_h * start_w)

            self.deconv = nn.Sequential(
                # (start_h, start_w) -> *2
                nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                # -> *2
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                # -> *2
                nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.ReLU(),
                # -> *2 (now at target H×W)
                nn.ConvTranspose2d(32, c, kernel_size=5, stride=2, padding=2, output_padding=1),
            )

            self._start_hw = (start_h, start_w)

        else:
            self.fc = nn.Sequential(
                nn.Linear(input_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, obs_shape[0]),
            )

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, stoch], dim=-1)
        if self.is_image:
            start_h, start_w = self._start_hw
            out = self.fc(x)
            out = out.view(out.size(0), 256, start_h, start_w)
            out = self.deconv(out)  # (B, C, H, W) exactly matching obs_shape
            return out
        else:
            return self.fc(x)


class RSSM(nn.Module):
    """Recurrent State‑Space Model (RSSM).

    This class maintains a deterministic hidden state `h_t` and a stochastic
    latent state `z_t` at each timestep.  Given a previous model state and
    an action, it updates the deterministic state through a GRU cell and
    outputs the prior distribution over the next stochastic state.  When
    provided with an encoded observation, it also produces the posterior
    distribution over the stochastic state (observation model).
    """

    def __init__(
            self,
            action_size: int,
            embed_size: int,
            deter_size: int = 200,
            stoch_size: int = 30,
    ):
        super().__init__()
        self.action_size = action_size
        self.embed_size = embed_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size

        # deterministic transition model
        self.gru = nn.GRUCell(stoch_size + action_size, deter_size)
        # prior: maps deterministic state to parameters of the stochastic latent
        self.prior_mean = nn.Linear(deter_size, stoch_size)
        self.prior_logstd = nn.Linear(deter_size, stoch_size)
        # posterior (observation) model: maps deterministic state and embed to posterior params
        self.post_mean = nn.Linear(deter_size + embed_size, stoch_size)
        self.post_logstd = nn.Linear(deter_size + embed_size, stoch_size)

    def init_state(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """Initialize deterministic and stochastic states with zeros."""
        return {
            "deter": torch.zeros(batch_size, self.deter_size, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
        }

    def _prior(self, deter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.prior_mean(deter)
        logstd = self.prior_logstd(deter)
        return mean, logstd

    def _posterior(self, deter: torch.Tensor, embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([deter, embed], dim=-1)
        mean = self.post_mean(h)
        logstd = self.post_logstd(h)
        return mean, logstd

    @staticmethod
    def _sample(mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logstd)
        eps = torch.randn_like(mean)
        return mean + eps * std

    def observe(self, embed: torch.Tensor, actions: torch.Tensor, prev_state: Dict[str, torch.Tensor]) -> Tuple[
        List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """Compute posterior and prior sequences given embeddings and actions.

        Args:
            embed: Tensor of shape (batch, time, embed_size).
            actions: Tensor of shape (batch, time, action_size), preprocessed
                actions from the environment.
            prev_state: Initial deterministic and stochastic states for the
                sequence.

        Returns:
            A pair (posterior_states, prior_states).  Each is a list of dicts
            containing the keys 'deter', 'stoch', 'mean', 'logstd'.  The lists
            have length equal to the sequence length.
        """
        batch, time, _ = embed.size()
        deter = prev_state["deter"]
        stoch = prev_state["stoch"]
        posterior_states: List[Dict[str, torch.Tensor]] = []
        prior_states: List[Dict[str, torch.Tensor]] = []
        for t in range(time):
            a = actions[:, t]
            x = embed[:, t]
            # Combine stochastic state and action to update deterministic state
            gru_input = torch.cat([stoch, a], dim=-1)
            deter = self.gru(gru_input, deter)
            # Prior predicts next latent from deterministic state
            mean_prior, logstd_prior = self._prior(deter)
            z_prior = self._sample(mean_prior, logstd_prior)
            prior_states.append({
                "deter": deter,
                "stoch": z_prior,
                "mean": mean_prior,
                "logstd": logstd_prior,
            })
            # Posterior uses observation
            mean_post, logstd_post = self._posterior(deter, x)
            z_post = self._sample(mean_post, logstd_post)
            posterior_states.append({
                "deter": deter,
                "stoch": z_post,
                "mean": mean_post,
                "logstd": logstd_post,
            })
            stoch = z_post
        return posterior_states, prior_states

    def imagine(self, actions: torch.Tensor, start_state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Roll out prior latent states given actions and an initial model state.

        This method is used for imagination during policy and value training.

        Args:
            actions: Tensor of shape (batch, time, action_size).
            start_state: Dict containing the initial deterministic and stochastic
                states.

        Returns:
            A list of prior state dicts of length equal to the time horizon.
        """
        batch, time, _ = actions.size()
        deter = start_state["deter"]
        stoch = start_state["stoch"]
        priors: List[Dict[str, torch.Tensor]] = []
        for t in range(time):
            a = actions[:, t]
            gru_input = torch.cat([stoch, a], dim=-1)
            deter = self.gru(gru_input, deter)
            mean, logstd = self._prior(deter)
            stoch = self._sample(mean, logstd)
            priors.append({
                "deter": deter,
                "stoch": stoch,
                "mean": mean,
                "logstd": logstd,
            })
        return priors


class RewardPredictor(nn.Module):
    """Predict immediate rewards from latent and hidden states."""

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int = 400):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(deter_size + stoch_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, stoch], dim=-1)
        return self.fc(x).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predict episode continuation flag (1 − done) from latent and hidden states."""

    def __init__(self, deter_size: int, stoch_size: int, hidden_size: int = 200):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(deter_size + stoch_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([deter, stoch], dim=-1)
        return self.fc(x).squeeze(-1)


class WorldModel(nn.Module):
    """Container class wrapping the RSSM, encoder, decoder and predictors.

    This class orchestrates the world model components and provides higher
    level interfaces for computing training losses and generating imagined
    trajectories.
    """

    def __init__(
            self,
            obs_shape: Tuple[int, ...],
            action_size: int,
            embed_size: int = 1024,
            deter_size: int = 200,
            stoch_size: int = 30,
            free_nats: float = 1.0,
            beta_pred: float = 1.0,
            beta_dyn: float = 1.0,
            beta_rep: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(obs_shape, embed_size)
        self.rssm = RSSM(action_size, embed_size, deter_size, stoch_size)
        self.decoder = Decoder(obs_shape, deter_size, stoch_size, embed_size)
        self.reward_predictor = RewardPredictor(deter_size, stoch_size)
        self.continue_predictor = ContinuePredictor(deter_size, stoch_size)
        self.embed_size = embed_size
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.free_nats = free_nats
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep

    def init_state(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        return self.rssm.init_state(batch_size, device)

    def get_feat(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine deterministic and stochastic parts into a feature vector."""
        return torch.cat([states["deter"], states["stoch"]], dim=-1)

    def imagine(self, actions: torch.Tensor, start_state: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """Generate imagined trajectories given actions.

        Args:
            actions: Tensor of shape (batch, time, action_size).
            start_state: Dict containing deterministic and stochastic state.

        Returns:
            A tuple (features, states) where features is a tensor of shape
            (batch, time, deter_size + stoch_size) and states is a list of
            state dicts at each time step.
        """
        priors = self.rssm.imagine(actions, start_state)
        feats = [self.get_feat(s) for s in priors]
        return torch.stack(feats, dim=1), priors

    def loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, continues: torch.Tensor,
             prev_state: Dict[str, torch.Tensor]) -> tuple[Tensor, list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Compute world model training loss.

        Args:
            obs: Tensor of shape (batch, time, *obs_shape) containing raw observations.
            actions: Tensor of shape (batch, time, action_size) containing actions.
            rewards: Tensor of shape (batch, time) with rewards.
            continues: Tensor of shape (batch, time) with continuation flags (1−done).
            prev_state: Initial model state dict.

        Returns:
            A tuple (loss, post, prior) where loss is a scalar tensor,
            post/prior are lists of state dicts from the posterior/prior sequences.
        """
        # Encode observations
        embed = self.encoder(obs)  # (batch, time, embed_size)
        # Run through RSSM
        post, prior = self.rssm.observe(embed, actions, prev_state)
        # Convert lists to tensors for convenience
        deter_post = torch.stack([s["deter"] for s in post], dim=1)
        stoch_post = torch.stack([s["stoch"] for s in post], dim=1)
        deter_prior = torch.stack([s["deter"] for s in prior], dim=1)
        stoch_prior = torch.stack([s["stoch"] for s in prior], dim=1)
        # Reconstruction
        if len(obs.shape) == 5:
            # image observations: (batch, time, C, H, W)
            recon = []
            batch, time, C, H, W = obs.size()
            for t in range(time):
                dec = self.decoder(
                    deter_post[:, t].reshape(-1, self.deter_size),
                    stoch_post[:, t].reshape(-1, self.stoch_size),
                )
                # dec is already (B, C, H, W); don't .view() it
                recon.append(dec)
            recon = torch.stack(recon, dim=1)
            recon_loss = F.mse_loss(recon, obs.float(), reduction="none").mean(dim=(2, 3, 4)).mean()
        else:
            # vector observations
            recon = []
            for t in range(obs.size(1)):
                dec = self.decoder(deter_post[:, t], stoch_post[:, t])
                recon.append(dec)
            recon = torch.stack(recon, dim=1)
            recon_loss = F.mse_loss(recon, obs.float(), reduction="none").mean(dim=2).mean()
        # Reward prediction
        reward_pred = []
        for t in range(len(post)):
            pred = self.reward_predictor(deter_post[:, t], stoch_post[:, t])
            reward_pred.append(pred)
        reward_pred = torch.stack(reward_pred, dim=1)
        reward_loss = F.mse_loss(reward_pred, rewards, reduction="none").mean()
        # Continue prediction
        cont_pred = []
        for t in range(len(post)):
            pred = self.continue_predictor(deter_post[:, t], stoch_post[:, t])
            cont_pred.append(pred)
        cont_pred = torch.stack(cont_pred, dim=1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_pred, continues, reduction="none").mean()
        # KL divergence between posterior and prior
        # Compute means and log stds
        mean_post = torch.stack([s["mean"] for s in post], dim=1)
        logstd_post = torch.stack([s["logstd"] for s in post], dim=1)
        mean_prior = torch.stack([s["mean"] for s in prior], dim=1)
        logstd_prior = torch.stack([s["logstd"] for s in prior], dim=1)
        # KL divergence per timestep and element
        kl_div = 0.5 * (
                (logstd_prior - logstd_post).exp() ** 2 +
                ((mean_post - mean_prior) / (logstd_prior.exp() + 1e-6)) ** 2 +
                2 * (logstd_prior - logstd_post) - 1
        )
        kl_div = kl_div.sum(dim=-1).mean()  # average over batch and time
        # Apply free nats (free bits) to prevent KL from shrinking too much【145968576409203†L389-L403】
        kl_loss = torch.max(kl_div, torch.tensor(self.free_nats, device=kl_div.device))
        # Combine losses with weights
        loss = self.beta_pred * (recon_loss + reward_loss + cont_loss) + self.beta_dyn * kl_loss
        # representation loss is omitted for simplicity; representation regularizer can be added here
        return loss, post, prior
