"""Linear probe utilities for AUTOPS EventSat latent attributes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .schema import AUTOPS_STATE_NAMES, WorldModelDataset

DEFAULT_ATTRIBUTE_NAMES = (
    "battery_margin",
    "storage_margin",
    "downlink_progress",
    "science_progress",
    "detection_progress",
    "communication_opportunity",
    "forced_mode_risk",
    "anomaly_safe",
)


@dataclass(frozen=True)
class ProbeFit:
    W: np.ndarray
    b: np.ndarray
    attribute_names: List[str]
    rmse: Dict[str, float]
    target_mean: np.ndarray
    target_std: np.ndarray


def _idx(names: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def build_attribute_targets(
    dataset: WorldModelDataset,
    attribute_names: Iterable[str] = DEFAULT_ATTRIBUTE_NAMES,
    state_names: Sequence[str] = AUTOPS_STATE_NAMES,
) -> np.ndarray:
    """Build AUTOPS-native mission attribute targets from state traces.

    Thermal margin and pointing accuracy are intentionally absent because AUTOPS
    does not simulate them explicitly in the EventSat implementation.
    """
    names = list(attribute_names)
    ix = _idx(state_names)
    state = dataset.state
    cap = np.maximum(state[..., ix["storage_capacity_mb"]], 1.0)
    stored = (
        state[..., ix["obc_data_mb"]]
        + state[..., ix["jetson_raw_mb"]]
        + state[..., ix["jetson_compressed_mb"]]
    )
    values = {
        "battery_margin": np.clip((state[..., ix["battery_soc"]] - 0.20) / 0.80, 0.0, 1.0),
        "storage_margin": np.clip(1.0 - stored / cap, 0.0, 1.0),
        "downlink_progress": state[..., ix["data_downlinked_mb"]],
        "science_progress": state[..., ix["total_observation_s"]] / 3600.0,
        "detection_progress": state[..., ix["total_detections"]],
        "communication_opportunity": (state[..., ix["ground_pass_active"]] > 0.5).astype(np.float32),
        "forced_mode_risk": dataset.forced_mode.astype(np.float32),
        "anomaly_safe": state[..., ix["health_nominal"]],
    }
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"unknown EventSat probe attributes: {missing}")
    return np.stack([values[name] for name in names], axis=-1).astype(np.float32)


def terminal_training_set(latents: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if latents.ndim != 3 or targets.ndim != 3:
        raise ValueError("latents and targets must be (E,T,D)/(E,T,K)")
    if latents.shape[:2] != targets.shape[:2]:
        raise ValueError("latents and targets must share episode/time axes")
    return latents.reshape(-1, latents.shape[-1]), targets.reshape(-1, targets.shape[-1])


def fit_ridge_probe(
    latents: np.ndarray,
    targets: np.ndarray,
    attribute_names: Iterable[str] = DEFAULT_ATTRIBUTE_NAMES,
    ridge: float = 1e-3,
    val_fraction: float = 0.2,
) -> ProbeFit:
    X, Y = terminal_training_set(latents.astype(np.float32), targets.astype(np.float32))
    n = X.shape[0]
    split = max(1, int(round(n * (1.0 - val_fraction))))
    split = min(split, n - 1) if n > 1 else n
    Xtr, Ytr = X[:split], Y[:split]
    Xv, Yv = X[split:], Y[split:]
    x_mean = Xtr.mean(axis=0, keepdims=True)
    x_std = Xtr.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    y_mean = Ytr.mean(axis=0, keepdims=True)
    y_std = Ytr.std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0
    Xn = (Xtr - x_mean) / x_std
    Yn = (Ytr - y_mean) / y_std
    Xa = np.concatenate([Xn, np.ones((Xn.shape[0], 1), dtype=np.float32)], axis=1)
    reg = ridge * np.eye(Xa.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Yn)
    Wn = coef[:-1].T
    bn = coef[-1]
    W = (Wn / x_std).astype(np.float32) * y_std.T
    b = (bn * y_std.reshape(-1) + y_mean.reshape(-1) - (W @ x_mean.reshape(-1))).astype(np.float32)
    if Xv.size:
        pred = Xv @ W.T + b
        err = np.sqrt(np.mean((pred - Yv) ** 2, axis=0))
    else:
        err = np.zeros(Y.shape[-1], dtype=np.float32)
    names = list(attribute_names)
    return ProbeFit(
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        attribute_names=names,
        rmse={name: float(err[i]) for i, name in enumerate(names)},
        target_mean=y_mean.reshape(-1).astype(np.float32),
        target_std=y_std.reshape(-1).astype(np.float32),
    )
