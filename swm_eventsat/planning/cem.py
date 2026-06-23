"""Categorical cross-entropy method for discrete action-sequence planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CEMConfig:
    horizon: int
    action_dim: int
    population: int = 256
    elite_frac: float = 0.10
    iterations: int = 4
    smoothing: float = 0.25
    min_prob: float = 1e-3


@dataclass
class CEMResult:
    action_sequence: np.ndarray
    score: float
    probabilities: np.ndarray
    elite_scores: np.ndarray


def _normalize_rows(probs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(probs, dtype=np.float64).copy()
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim == 1:
            out[:, ~mask_arr] = 0.0
        elif mask_arr.shape == out.shape:
            out[~mask_arr] = 0.0
        else:
            raise ValueError(f"mask shape {mask_arr.shape} incompatible with {out.shape}")
    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0
    if np.any(bad):
        out[bad] = 1.0
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 1:
                out[np.ix_(bad, ~mask_arr)] = 0.0
            elif mask_arr.shape == out.shape:
                out[bad] *= mask_arr[bad]
        row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum[:, 0] <= 0):
        raise ValueError("action mask leaves at least one timestep with no valid actions")
    return out / row_sum


def categorical_cem(
    score_fn: Callable[[np.ndarray], np.ndarray],
    config: CEMConfig,
    rng: np.random.Generator | None = None,
    initial_probs: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
) -> CEMResult:
    """Optimize a discrete action sequence with CEM.

    ``score_fn`` receives an integer array shaped ``(population, horizon)`` and
    returns one score per candidate. Higher is better.
    """
    rng = rng or np.random.default_rng()
    horizon = int(config.horizon)
    action_dim = int(config.action_dim)
    if horizon <= 0 or action_dim <= 1:
        raise ValueError("horizon must be positive and action_dim must exceed one")

    if initial_probs is None:
        probs = np.full((horizon, action_dim), 1.0 / action_dim, dtype=np.float64)
    else:
        probs = np.asarray(initial_probs, dtype=np.float64).reshape(horizon, action_dim)
    probs = _normalize_rows(probs, action_mask)

    n_elite = max(1, int(round(config.population * config.elite_frac)))
    best_seq = np.zeros(horizon, dtype=np.int64)
    best_score = -np.inf
    elite_scores = np.empty(0, dtype=np.float32)

    for _ in range(int(config.iterations)):
        samples = np.empty((int(config.population), horizon), dtype=np.int64)
        for t in range(horizon):
            samples[:, t] = rng.choice(action_dim, size=int(config.population), p=probs[t])
        scores = np.asarray(score_fn(samples), dtype=np.float64).reshape(-1)
        if scores.shape[0] != samples.shape[0]:
            raise ValueError("score_fn must return one score per sampled sequence")

        elite_idx = np.argsort(scores)[-n_elite:]
        elites = samples[elite_idx]
        elite_scores = scores[elite_idx].astype(np.float32)
        top = int(elite_idx[np.argmax(scores[elite_idx])])
        if scores[top] > best_score:
            best_score = float(scores[top])
            best_seq = samples[top].copy()

        counts = np.full((horizon, action_dim), float(config.min_prob), dtype=np.float64)
        for t in range(horizon):
            counts[t] += np.bincount(elites[:, t], minlength=action_dim)
        update = _normalize_rows(counts, action_mask)
        probs = (1.0 - float(config.smoothing)) * probs + float(config.smoothing) * update
        probs = _normalize_rows(probs, action_mask)

    return CEMResult(
        action_sequence=best_seq.astype(np.int64),
        score=best_score,
        probabilities=probs.astype(np.float32),
        elite_scores=elite_scores,
    )

