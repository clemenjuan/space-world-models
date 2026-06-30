"""CEM MPC over EventSat latent rollouts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol

import numpy as np

from swm_eventsat.models.probes import DEFAULT_ATTRIBUTE_NAMES
from swm_eventsat.schema import MODE_LIST


class LatentRolloutModel(Protocol):
    def rollout(self, history: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """Return latent rollout with shape (N,H,D) for action (N,H,7)."""


@dataclass
class PlannerResult:
    mode: str
    mode_index: int
    best_sequence: np.ndarray
    best_score: float
    diagnostics: Dict[str, float]


@dataclass
class CEMPlanner:
    model: LatentRolloutModel
    W: np.ndarray
    b: np.ndarray
    mode_weights: np.ndarray
    horizon: int = 12
    samples: int = 512
    elites: int = 64
    iterations: int = 4
    alpha: float = 0.7
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    previous_solution: Optional[np.ndarray] = None
    penalty_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def select_action(self, history: Dict[str, np.ndarray], first_mask: Optional[np.ndarray] = None) -> PlannerResult:
        probs = self._initial_probs()
        first_mask = np.ones(len(MODE_LIST), dtype=bool) if first_mask is None else first_mask.astype(bool)
        best_sequence = None
        best_score = -np.inf
        for _ in range(int(self.iterations)):
            seq = self._sample(probs)
            seq = self._apply_first_mask(seq, first_mask)
            action = encode_mode_sequences(seq)
            z = self.model.rollout(history, action)
            scores = score_latents(z, self.W, self.b, self.mode_weights, seq, self.penalty_fn)
            idx = int(np.argmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_sequence = seq[idx].copy()
            elite_count = max(1, min(int(self.elites), int(self.samples)))
            elite_idx = np.argpartition(scores, -elite_count)[-elite_count:]
            empirical = np.full_like(probs, 1e-5)
            for t in range(self.horizon):
                counts = np.bincount(seq[elite_idx, t], minlength=len(MODE_LIST)).astype(np.float64)
                empirical[t] += counts / max(1.0, counts.sum())
            empirical /= empirical.sum(axis=1, keepdims=True)
            probs = self.alpha * empirical + (1.0 - self.alpha) * probs
            probs /= probs.sum(axis=1, keepdims=True)
        if best_sequence is None:
            best_sequence = np.zeros(self.horizon, dtype=np.int64)
        self.previous_solution = best_sequence
        return PlannerResult(
            mode=MODE_LIST[int(best_sequence[0])],
            mode_index=int(best_sequence[0]),
            best_sequence=best_sequence,
            best_score=float(best_score),
            diagnostics={
                "candidate_count": float(self.samples),
                "cem_iterations": float(self.iterations),
            },
        )

    def _initial_probs(self) -> np.ndarray:
        if self.previous_solution is None:
            return np.full((self.horizon, len(MODE_LIST)), 1.0 / len(MODE_LIST), dtype=np.float64)
        shifted = np.concatenate([self.previous_solution[1:], self.previous_solution[-1:]])[: self.horizon]
        probs = np.full((self.horizon, len(MODE_LIST)), 0.04 / (len(MODE_LIST) - 1), dtype=np.float64)
        for t, idx in enumerate(shifted):
            probs[t, int(idx)] = 0.96
        return probs / probs.sum(axis=1, keepdims=True)

    def _sample(self, probs: np.ndarray) -> np.ndarray:
        seq = np.zeros((self.samples, self.horizon), dtype=np.int64)
        actions = np.arange(len(MODE_LIST), dtype=np.int64)
        for t in range(self.horizon):
            seq[:, t] = self.rng.choice(actions, size=self.samples, p=probs[t])
        return seq

    def _apply_first_mask(self, seq: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
        allowed = np.flatnonzero(first_mask)
        if allowed.size == 0:
            allowed = np.asarray([0], dtype=np.int64)
        bad = ~first_mask[seq[:, 0]]
        if np.any(bad):
            seq[bad, 0] = self.rng.choice(allowed, size=int(np.sum(bad)))
        return seq


def encode_mode_sequences(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.int64)
    out = np.zeros((*seq.shape, len(MODE_LIST)), dtype=np.float32)
    rows = np.indices(seq.shape)
    out[rows[0], rows[1], seq] = 1.0
    return out


def score_latents(
    z: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mode_weights: np.ndarray,
    seq: np.ndarray,
    penalty_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    terminal = z[:, -1, :]
    attrs = terminal @ W.T + b
    scores = attrs @ mode_weights
    if penalty_fn is not None:
        scores = scores - penalty_fn(z, seq)
    return scores.astype(np.float32)


def default_mode_weights(mode: str, attribute_names=DEFAULT_ATTRIBUTE_NAMES) -> np.ndarray:
    weights = {name: 0.0 for name in attribute_names}
    if mode == "safe":
        weights.update(battery_margin=0.45, storage_margin=0.20, forced_mode_risk=-0.20, anomaly_safe=0.15)
    elif mode == "downlink":
        weights.update(downlink_progress=0.45, storage_margin=0.25, battery_margin=0.15, forced_mode_risk=-0.10)
    else:
        # detection_progress is degenerate (zero detections in base-EventSat AO
        # traces — detection is an SSA concept), so its probe is not trustworthy.
        # Force its utility weight to 0 and fold the science share it used to carry
        # into science_progress. Restore if SSA detections enter the dataset.
        weights.update(science_progress=0.40, downlink_progress=0.25, battery_margin=0.20, storage_margin=0.15)
    return np.asarray([weights[name] for name in attribute_names], dtype=np.float32)
