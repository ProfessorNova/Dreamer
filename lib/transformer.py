from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.RMSNorm):
    """Thin wrapper in case you later want custom defaults."""
    pass


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: (..., d_model)
        x_gated, x_linear = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x_gated) * x_linear)


def apply_rope(q, k, rope_cache):
    """Apply precomputed RoPE to q/k. Shapes: (B, H, T*S, Dh)."""
    # Implementation detail skipped here; standard RoPE pattern.
    # You would broadcast rope_cache to q/k and apply complex rotation.
    raise NotImplementedError


class SpaceTimeAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, kind: str):
        """
        kind: 'space' or 'time'
        """
        super().__init__()
        self.kind = kind
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        # Optional: parameters for QKNorm, logit soft-capping

    def forward(self, x, attn_mask=None, rope_cache=None):
        """
        x: (B, T, S, D) tokens.
        kind == 'space':   attend within each time step over spatial tokens
        kind == 'time':    attend along time at each spatial position
        """
        B, T, S, D = x.shape
        if self.kind == "space":
            x_ = x.view(B * T, S, D)
        else:
            x_ = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)

        q = self.q_proj(x_)
        k = self.k_proj(x_)
        v = self.v_proj(x_)

        q = q.view(B * (T if self.kind == "space" else S),
                   -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view_as(q)
        v = v.view_as(q)

        if rope_cache is not None:
            q, k = apply_rope(q, k, rope_cache)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(self.kind == "time")
        )
        out = out.transpose(1, 2).reshape(x_.shape[0], x_.shape[1], -1)
        out = self.o_proj(out)

        if self.kind == "space":
            out = out.view(B, T, S, D)
        else:
            out = out.view(B, S, T, D).permute(0, 2, 1, 3)

        return out


class EfficientBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, d_ff,
                 use_time_attn: bool):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.space_attn = SpaceTimeAttention(d_model, num_heads, head_dim,
                                             kind="space")
        self.time_attn = (SpaceTimeAttention(d_model, num_heads, head_dim,
                                             kind="time")
                          if use_time_attn else None)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, rope_cache=None):
        # x: (B, T, S, D)
        h = self.norm1(x)
        h = self.space_attn(h, rope_cache=rope_cache)
        if self.time_attn is not None:
            h = self.time_attn(h, rope_cache=rope_cache)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class EfficientTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, head_dim, d_ff,
                 temporal_every=4):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            use_time = (i % temporal_every == 0)
            blocks.append(
                EfficientBlock(d_model, num_heads, head_dim, d_ff,
                               use_time_attn=use_time)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, rope_cache=None):
        for blk in self.blocks:
            x = blk(x, rope_cache=rope_cache)
        return x
