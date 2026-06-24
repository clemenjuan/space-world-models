"""Canonical EventSat trajectory and AUTOPS world-model dataset schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

MODE_LIST = (
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
)
MODE_TO_INDEX = {name: idx for idx, name in enumerate(MODE_LIST)}
EVENTSAT_MODE_LIST = MODE_LIST
EVENTSAT_MODE_TO_INDEX = MODE_TO_INDEX

ACTION_NAMES = tuple(f"mode_{mode}" for mode in MODE_LIST)
AUTOPS_STATE_NAMES = (
    "battery_soc",
    "current_mode_idx",
    "in_sunlight",
    "ground_pass_active",
    "orbital_phase",
    "time_to_next_eclipse",
    "time_to_next_pass",
    "remaining_pass_duration",
    "following_gap_steps",
    "data_stored_mb",
    "obc_data_mb",
    "jetson_raw_mb",
    "jetson_compressed_mb",
    "data_downlinked_mb",
    "uncompressed_observations",
    "compression_progress",
    "undetected_observations",
    "detection_progress",
    "total_observation_s",
    "total_detections",
    "storage_capacity_mb",
    "jetson_capacity_mb",
    "daily_downlink_budget_mb",
    "achievable_downlink_mb",
    "health_nominal",
)
REQUIRED_WORLD_MODEL_KEYS = (
    "obs",
    "action",
    "state",
    "reward",
    "mode",
    "resolved_mode",
    "forced_mode",
    "episode_seed",
)


@dataclass
class TrajectoryBatch:
    """In-memory EventSat trajectory archive used by generators and adapters."""

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


@dataclass(frozen=True)
class WorldModelDataset:
    """Loaded AUTOPS EventSat world-model dataset."""

    path: Path
    obs: np.ndarray
    action: np.ndarray
    state: np.ndarray
    reward: np.ndarray
    mode: np.ndarray
    resolved_mode: np.ndarray
    forced_mode: np.ndarray
    episode_seed: np.ndarray
    metadata: Dict[str, Any]

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[1])

    @property
    def dataset_steps(self) -> int:
        return self.n_episodes * self.n_steps


def validate_trajectory_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate a trajectory archive and return shape metadata."""
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
    """Save an EventSat trajectory archive with validation and metadata."""
    path = Path(path)
    arrays = batch.arrays()
    shape_meta = validate_trajectory_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **arrays,
        mode_names=np.asarray(batch.metadata.get("mode_names", MODE_LIST)),
        metadata=np.asarray({**batch.metadata, **shape_meta}, dtype=object),
    )
    return path


def load_trajectory_npz(path: str | Path) -> TrajectoryBatch:
    """Load and validate a generic EventSat trajectory archive."""
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


def load_metadata(path: Path) -> Dict[str, Any]:
    candidates = [
        path.with_suffix(".metadata.json"),
        path.with_name(path.stem + ".metadata.json"),
        path.parent / "eventsat_world_model_v1.metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def load_world_model_dataset(path_like: str | Path, metadata_path: Optional[str | Path] = None) -> WorldModelDataset:
    """Load the AUTOPS EventSat v1 world-model dataset."""
    path = Path(path_like)
    blob = np.load(path)
    missing = [key for key in REQUIRED_WORLD_MODEL_KEYS if key not in blob]
    if missing:
        raise ValueError(f"dataset {path} missing required keys: {missing}")
    obs = blob["obs"].astype(np.float32)
    action = blob["action"].astype(np.float32)
    state = blob["state"].astype(np.float32)
    if obs.ndim != 3 or obs.shape[-1] != 25:
        raise ValueError(f"obs must be (E,T,25), got {obs.shape}")
    if action.ndim != 3 or action.shape[-1] != len(MODE_LIST):
        raise ValueError(f"action must be (E,T,{len(MODE_LIST)}), got {action.shape}")
    if state.ndim != 3:
        raise ValueError(f"state must be (E,T,S), got {state.shape}")
    if obs.shape[:2] != action.shape[:2] or obs.shape[:2] != state.shape[:2]:
        raise ValueError("obs/action/state episode and time axes must match")
    onehot = action.sum(axis=-1)
    if not np.allclose(onehot, 1.0, atol=1e-4):
        raise ValueError("action must be one-hot over EventSat modes")
    if not np.isfinite(obs).all() or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("dataset contains non-finite values")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8")) if metadata_path else load_metadata(path)
    return WorldModelDataset(
        path=path,
        obs=obs,
        action=action,
        state=state,
        reward=blob["reward"].astype(np.float32),
        mode=blob["mode"].astype(np.int64),
        resolved_mode=blob["resolved_mode"].astype(np.int64),
        forced_mode=blob["forced_mode"].astype(np.float32),
        episode_seed=blob["episode_seed"].astype(np.int64),
        metadata=metadata,
    )


__all__ = [
    "ACTION_NAMES",
    "AUTOPS_STATE_NAMES",
    "EVENTSAT_MODE_LIST",
    "EVENTSAT_MODE_TO_INDEX",
    "MODE_LIST",
    "MODE_TO_INDEX",
    "TrajectoryBatch",
    "WorldModelDataset",
    "load_metadata",
    "load_trajectory_npz",
    "load_world_model_dataset",
    "save_trajectory_npz",
    "validate_trajectory_arrays",
]
