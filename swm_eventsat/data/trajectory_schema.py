"""Canonical trajectory schema for EventSat scheduling datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EVENTSAT_MODE_LIST = (
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
)
EVENTSAT_MODE_TO_INDEX = {name: idx for idx, name in enumerate(EVENTSAT_MODE_LIST)}


@dataclass
class TrajectoryBatch:
    """In-memory form of a LeWM training/evaluation trajectory archive."""

    obs: np.ndarray
    action: np.ndarray
    state: np.ndarray | None = None
    reward: np.ndarray | None = None
    mode: np.ndarray | None = None
    resolved_mode: np.ndarray | None = None
    forced_mode: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"obs": self.obs, "action": self.action}
        for key in ("state", "reward", "mode", "resolved_mode", "forced_mode"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def validate_trajectory_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate the shared trajectory schema and return shape metadata."""
    if "obs" not in arrays or "action" not in arrays:
        raise ValueError("trajectory archive must contain obs and action arrays")
    obs = np.asarray(arrays["obs"])
    action = np.asarray(arrays["action"])
    if obs.ndim != 3:
        raise ValueError(f"obs must have shape (episode,time,dim), got {obs.shape}")
    if action.ndim != 3:
        raise ValueError(f"action must have shape (episode,time,dim), got {action.shape}")
    if obs.shape[:2] != action.shape[:2]:
        raise ValueError(f"obs/action episode-time mismatch: {obs.shape[:2]} vs {action.shape[:2]}")
    meta: dict[str, Any] = {
        "episodes": int(obs.shape[0]),
        "steps": int(obs.shape[1]),
        "obs_dim": int(obs.shape[2]),
        "action_dim": int(action.shape[2]),
    }
    for key in ("state", "reward", "mode", "resolved_mode", "forced_mode"):
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.shape[:2] != obs.shape[:2]:
            raise ValueError(f"{key} episode-time mismatch: {value.shape[:2]} vs {obs.shape[:2]}")
        meta[f"{key}_shape"] = tuple(int(v) for v in value.shape)
    if not np.isfinite(obs).all():
        raise ValueError("obs contains non-finite values")
    if not np.isfinite(action).all():
        raise ValueError("action contains non-finite values")
    return meta


def save_trajectory_npz(path: str | Path, batch: TrajectoryBatch) -> Path:
    """Save a trajectory batch with validation and metadata fields."""
    path = Path(path)
    arrays = batch.arrays()
    shape_meta = validate_trajectory_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        mode_names=np.asarray(batch.metadata.get("mode_names", EVENTSAT_MODE_LIST)),
        metadata=np.asarray({**batch.metadata, **shape_meta}, dtype=object),
    )
    return path


def load_trajectory_npz(path: str | Path) -> TrajectoryBatch:
    """Load and validate a trajectory archive."""
    blob = np.load(path, allow_pickle=True)
    arrays = {key: blob[key] for key in blob.files if key not in {"metadata", "mode_names"}}
    validate_trajectory_arrays(arrays)
    metadata: dict[str, Any] = {}
    if "metadata" in blob:
        raw = blob["metadata"]
        if raw.shape == ():
            metadata.update(dict(raw.item()))
    if "mode_names" in blob:
        metadata["mode_names"] = tuple(str(x) for x in blob["mode_names"].tolist())
    return TrajectoryBatch(
        obs=arrays["obs"].astype(np.float32),
        action=arrays["action"].astype(np.float32),
        state=arrays.get("state"),
        reward=arrays.get("reward"),
        mode=arrays.get("mode"),
        resolved_mode=arrays.get("resolved_mode"),
        forced_mode=arrays.get("forced_mode"),
        metadata=metadata,
    )

