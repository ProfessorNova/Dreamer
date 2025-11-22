# models/tokenizer.py
import torch
from torch import nn
from lib.transformer import EfficientTransformer, RMSNorm
from lib.utils import LossRMSNormalizer


class CausalTokenizer(nn.Module):
    def __init__(self, img_size, patch_size, d_model,
                 num_layers, num_heads, head_dim, d_ff,
                 num_latents, lpips_module=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.lpips = lpips_module  # optional
        self.loss_norm = LossRMSNormalizer()

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # Learned latent tokens per frame
        self.num_latents = num_latents
        self.latents = nn.Parameter(
            torch.randn(1, 1, num_latents, d_model)
        )

        # Encoder & Decoder (both causal in time via mask, hier vereinfacht)
        self.encoder = EfficientTransformer(
            d_model, num_layers, num_heads, head_dim, d_ff
        )
        self.decoder = EfficientTransformer(
            d_model, num_layers, num_heads, head_dim, d_ff
        )

        # Bottleneck projection
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh()
        )
        self.project_back = nn.Linear(d_model // 2, d_model)

        # Reconstruction head
        self.recon_head = nn.ConvTranspose3d(
            in_channels=d_model,
            out_channels=3,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

    def encode(self, video):
        """
        video: (B, T, C, H, W)
        returns z: (B, T, S_z, D)
        """
        B, T, C, H, W = video.shape
        x = video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.patch_embed(x)  # (B, D, T, H', W')
        B, D, T, H_, W_ = x.shape
        S = H_ * W_
        x = x.view(B, D, T, S).permute(0, 2, 3, 1)  # (B, T, S, D)

        latents = self.latents.expand(B, T, self.num_latents, self.d_model)
        x = torch.cat([x, latents], dim=2)  # patches + latents

        x = self.encoder(x)  # causal mask in impl.

        # Read out only latent tokens
        latents_out = x[:, :, -self.num_latents:, :]
        z = self.readout(latents_out)
        return z

    def decode(self, z):
        """
        z: (B, T, S_z, D_bottleneck) after readout
        """
        B, T, S_z, D_b = z.shape
        h = self.project_back(z)  # -> D_model
        # Concatenate dummy patch tokens to run transformer decoder
        # (hier vereinfachtes Setup)
        h = self.decoder(h)
        h = h.permute(0, 3, 1, 2)  # (B, D, T, S_z)
        # map S_z back to H',W' (angenommen passend gewählt)
        H_ = W_ = int(S_z ** 0.5)
        h = h.view(B, self.d_model, T, H_, W_)
        recon = self.recon_head(h)  # (B, 3, T, H, W)
        recon = recon.permute(0, 2, 1, 3, 4)
        return recon

    def forward(self, video, targets=None, mask_ratio_sampler=None):
        """
        If targets is not None, compute MAE loss (MSE + 0.2 * LPIPS),
        already normalized via LossRMSNormalizer.
        """
        # For Masked Autoencoding: drop random patches before encoder.
        # (Ich lasse die konkrete Maskierung als TODO.)
        z = self.encode(video)

        outputs = {"z": z, "recon": None, "loss": None}

        if targets is not None:
            recon = self.decode(z)
            outputs["recon"] = recon

            mse = torch.mean((recon - targets) ** 2)
            mse = self.loss_norm("tokenizer_mse", mse)

            if self.lpips is not None:
                lp = self.lpips(
                    recon.reshape(-1, 3, recon.size(-2), recon.size(-1)),
                    targets.reshape(-1, 3, targets.size(-2), targets.size(-1)),
                    normalize=True,
                )
                lp = lp.mean()
                lp = self.loss_norm("tokenizer_lpips", lp)
            else:
                lp = 0.0

            outputs["loss"] = mse + 0.2 * lp

        return outputs
