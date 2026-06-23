"""Mission utility functions over latent trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def simplex_weights(values: np.ndarray | list[float]) -> np.ndarray:
    """Normalize non-negative mission-mode weights onto the probability simplex."""
    weights = np.asarray(values, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError("mission weights must be one-dimensional")
    if np.any(weights < 0):
        raise ValueError("mission weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("at least one mission weight must be positive")
    return weights / total


@dataclass
class LinearLatentUtility:
    """Terminal utility ``U(z_H) = w @ (W z_H + b)``."""

    W: np.ndarray
    b: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=np.float32)
        self.b = np.asarray(self.b, dtype=np.float32)
        self.weights = simplex_weights(self.weights)
        if self.W.ndim != 2:
            raise ValueError("W must be shaped (objectives, latent_dim)")
        if self.b.shape != (self.W.shape[0],):
            raise ValueError("b must have one entry per objective")
        if self.weights.shape != (self.W.shape[0],):
            raise ValueError("weights must have one entry per objective")

    def objective_values(self, latents: np.ndarray) -> np.ndarray:
        z = np.asarray(latents, dtype=np.float32)
        terminal = z[..., -1, :] if z.ndim >= 3 else z
        return terminal @ self.W.T + self.b

    def score(self, latents: np.ndarray) -> np.ndarray:
        values = self.objective_values(latents)
        return values @ self.weights

