"""Dataset schema for AUTOPS-generated EventSat world-model traces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
ACTION11_NAMES = (
    *(f"mode_{mode}" for mode in MODE_LIST),
    "data_priority_normal",
    "data_priority_urgent",
    "pipeline_compress_first",
    "pipeline_detect_first",
)
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
REQUIRED_KEYS = (
    "obs",
    "action",
    "state",
    "reward",
    "mode",
    "resolved_mode",
    "forced_mode",
    "episode_seed",
)


@dataclass(frozen=True)
class WorldModelDataset:
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
    path = Path(path_like)
    blob = np.load(path)
    missing = [key for key in REQUIRED_KEYS if key not in blob]
    if missing:
        raise ValueError(f"dataset {path} missing required keys: {missing}")
    obs = blob["obs"].astype(np.float32)
    action = blob["action"].astype(np.float32)
    state = blob["state"].astype(np.float32)
    if obs.ndim != 3 or obs.shape[-1] != 25:
        raise ValueError(f"obs must be (E,T,25), got {obs.shape}")
    if action.ndim != 3 or action.shape[-1] != 11:
        raise ValueError(f"action must be (E,T,11), got {action.shape}")
    if state.ndim != 3:
        raise ValueError(f"state must be (E,T,S), got {state.shape}")
    if obs.shape[:2] != action.shape[:2] or obs.shape[:2] != state.shape[:2]:
        raise ValueError("obs/action/state episode and time axes must match")
    onehot = action[..., :7].sum(axis=-1)
    if not np.allclose(onehot, 1.0, atol=1e-4):
        raise ValueError("action[..., :7] must be one-hot over EventSat modes")
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
