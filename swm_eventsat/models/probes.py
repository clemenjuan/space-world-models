"""Linear probes from LeWM latent vectors to mission attributes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LinearProbe:
    """Affine probe ``y = x @ weight.T + bias`` fitted by ridge regression."""

    weight: np.ndarray
    bias: np.ndarray
    target_names: tuple[str, ...] = ()

    def predict(self, latents: np.ndarray) -> np.ndarray:
        x = np.asarray(latents, dtype=np.float32)
        flat = x.reshape(-1, x.shape[-1])
        y = flat @ self.weight.T + self.bias
        return y.reshape(*x.shape[:-1], self.weight.shape[0]).astype(np.float32)


def fit_linear_probe(
    latents: np.ndarray,
    targets: np.ndarray,
    ridge: float = 1e-4,
    target_names: Iterable[str] = (),
) -> LinearProbe:
    """Fit a multi-output linear probe with a bias term."""
    x = np.asarray(latents, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("latents and targets must include sample and feature dimensions")
    x2 = x.reshape(-1, x.shape[-1])
    y2 = y.reshape(-1, y.shape[-1])
    if x2.shape[0] != y2.shape[0]:
        raise ValueError(f"sample mismatch: {x2.shape[0]} latents vs {y2.shape[0]} targets")

    design = np.concatenate([x2, np.ones((x2.shape[0], 1), dtype=np.float64)], axis=1)
    reg = ridge * np.eye(design.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y2)
    weight = coef[:-1].T.astype(np.float32)
    bias = coef[-1].astype(np.float32)
    return LinearProbe(weight=weight, bias=bias, target_names=tuple(target_names))

