"""MLP encoder replacing LeWM's ViT for low-dimensional vector observations."""

import torch
from torch import nn


class VectorEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=256, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """Map ``(B, T, in_dim)`` to ``(B, T, out_dim)``."""
        return self.net(x.float())


__all__ = ["VectorEncoder"]

