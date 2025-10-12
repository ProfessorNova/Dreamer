import torch.nn as nn
import torch.nn.functional as F


class DreamerV3Block(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int | None = None,
                 norm: bool = True, residual: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim) if norm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim or in_dim)
        self.use_residual = residual and (out_dim is None or out_dim == in_dim)

    def forward(self, x):
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h if self.use_residual else h


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, units: int = 256, depth: int = 3,
                 norm: bool = True, zero_init: bool = False):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(DreamerV3Block(dim, hidden=units, out_dim=units, norm=norm, residual=True))
            dim = units
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(dim, out_dim)

        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.blocks(x))
