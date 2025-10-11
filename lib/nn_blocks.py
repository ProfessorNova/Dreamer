import torch.nn as nn


class MLP(nn.Module):
    """SiLU + LayerNorm MLP with configurable units/depth (DreamerV3-ish)."""

    def __init__(self, in_dim: int, out_dim: int, units: int = 256, depth: int = 16, norm=True):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, units)]
            if norm:
                layers += [nn.LayerNorm(units)]
            layers += [nn.SiLU()]
            last = units
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
