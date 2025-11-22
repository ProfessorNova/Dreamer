import torch
from torch import nn
from lib.transformer import EfficientTransformer
from lib.utils import LossRMSNormalizer


class ShortcutDynamics(nn.Module):
    def __init__(self, d_model, num_layers, num_heads,
                 head_dim, d_ff, num_register_tokens,
                 action_dim, num_action_tokens):
        super().__init__()
        self.d_model = d_model
        self.num_register_tokens = num_register_tokens
        self.num_action_tokens = num_action_tokens

        self.loss_norm = LossRMSNormalizer()

        self.register_tokens = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, d_model)
        )

        # Embed shortcut signal level tau and step size d as a single token
        self.shortcut_embed = nn.Linear(2, d_model)

        # Action embedding (Aggregation über Komponenten, wie im Paper)
        self.action_embed = nn.Linear(action_dim, num_action_tokens * d_model)

        self.backbone = EfficientTransformer(
            d_model, num_layers, num_heads, head_dim, d_ff
        )

        # Head to predict clean representations z1
        self.flow_head = nn.Linear(d_model, d_model)

    def _build_sequence(self, z_tilde, a, tau, d):
        """
        z_tilde: (B, T, S_z, D)
        a:       (B, T, action_dim)
        tau,d:   (B, T) in [0,1]
        returns seq: (B, T, S_total, D)
        """
        B, T, S_z, D = z_tilde.shape

        reg = self.register_tokens.expand(B, T, self.num_register_tokens, D)

        sd = torch.stack([tau, d], dim=-1)  # (B,T,2)
        sd_tok = self.shortcut_embed(sd).unsqueeze(2)  # (B,T,1,D)

        act = self.action_embed(a)  # (B,T,num_action_tokens*D)
        act = act.view(B, T, self.num_action_tokens, D)

        seq = torch.cat([z_tilde, reg, act, sd_tok], dim=2)
        return seq

    def forward(self, z_clean, a, tau, d):
        """
        z_clean: (B,T,S_z,D) samples from data
        a: (B,T,action_dim)
        tau,d: (B,T) noise schedule
        Returns:
          dict with 'z_pred', 'loss'
        """
        B, T, S_z, D = z_clean.shape

        # Sample noise and mix with clean reps (Eq. 6)
        z0 = torch.randn_like(z_clean)
        z_tilde = (1.0 - tau[..., None, None]) * z0 + tau[..., None, None] * z_clean

        seq = self._build_sequence(z_tilde, a, tau, d)
        h = self.backbone(seq)
        # Take only positions corresponding to z tokens for prediction
        h_z = h[:, :, :S_z, :]

        z_hat = self.flow_head(h_z)

        outputs = {"z_pred": z_hat, "loss": None}

        # Shortcut forcing loss (schematisch – Details aus Gl. 7)
        diff = z_hat - z_clean
        mse = torch.mean(diff ** 2)

        # Ramp weight w(tau) = 0.9 tau + 0.1
        w = 0.9 * tau + 0.1
        w = w[..., None, None]
        loss = (w * diff.pow(2)).mean()

        # Loss-RMS normalisieren
        loss = self.loss_norm("shortcut", loss)
        outputs["loss"] = loss

        return outputs
