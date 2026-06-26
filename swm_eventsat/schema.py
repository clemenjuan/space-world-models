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

# SSA constellation modes (8D, distinct order from the 7D EventSat MODE_LIST;
# adds isl_share). Mirrors src.ssa.env.SSA_MODES in autops-agentic-framework.
SSA_MODE_LIST = (
    "charging",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
    "isl_share",
    "safe",
)
SSA_MODE_TO_INDEX = {name: idx for idx, name in enumerate(SSA_MODE_LIST)}
REQUIRED_SSA_WORLD_MODEL_KEYS = REQUIRED_WORLD_MODEL_KEYS + (
    "delivered_coverage",
    "onboard_coverage",
    "archive_records",
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


@dataclass(frozen=True)
class SSAWorldModelDataset:
    """Loaded AUTOPS SSA constellation world-model dataset.

    Per-satellite arrays carry a satellite axis ``S``; ``delivered_coverage`` /
    ``onboard_coverage`` / ``archive_records`` are collective (per episode-time).
    """

    path: Path
    obs: np.ndarray           # (E, T, S, 25)
    action: np.ndarray        # (E, T, S, 8) one-hot over SSA_MODE_LIST
    state: np.ndarray         # (E, T, S, 25)
    reward: np.ndarray        # (E, T, S)
    mode: np.ndarray          # (E, T, S)
    resolved_mode: np.ndarray  # (E, T, S)
    forced_mode: np.ndarray   # (E, T, S)
    episode_seed: np.ndarray  # (E,)
    sat_ids: tuple[str, ...]
    delivered_coverage: np.ndarray  # (E, T)
    onboard_coverage: np.ndarray    # (E, T)
    archive_records: np.ndarray     # (E, T)
    metadata: Dict[str, Any]

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[1])

    @property
    def n_satellites(self) -> int:
        return int(self.obs.shape[2])

    def flatten_satellites(self) -> "WorldModelDataset":
        """Collapse the satellite axis into the episode axis (IMAS shared WM).

        Returns a single-sat-shaped ``WorldModelDataset`` with ``(E*S, T, ·)``
        per-satellite arrays, so the existing LeWM trainer (which expects
        ``(episode, time, dim)`` and an 8D action) can consume each satellite
        trajectory as an independent transition stream.
        """
        e, t, s = self.obs.shape[0], self.obs.shape[1], self.obs.shape[2]

        def fold(arr: np.ndarray) -> np.ndarray:
            # (E, T, S, ...) -> (E, S, T, ...) -> (E*S, T, ...)
            moved = np.moveaxis(arr, 2, 1)
            return moved.reshape((e * s,) + moved.shape[2:])

        seed = np.repeat(self.episode_seed, s).astype(np.int64)
        return WorldModelDataset(
            path=self.path,
            obs=fold(self.obs),
            action=fold(self.action),
            state=fold(self.state),
            reward=fold(self.reward),
            mode=fold(self.mode),
            resolved_mode=fold(self.resolved_mode),
            forced_mode=fold(self.forced_mode),
            episode_seed=seed,
            metadata={**self.metadata, "flattened_from_ssa": True, "n_satellites": s},
        )


def load_ssa_world_model_dataset(
    path_like: str | Path, metadata_path: Optional[str | Path] = None
) -> SSAWorldModelDataset:
    """Load and validate the AUTOPS SSA constellation world-model dataset."""
    path = Path(path_like)
    blob = np.load(path, allow_pickle=False)
    missing = [key for key in REQUIRED_SSA_WORLD_MODEL_KEYS if key not in blob]
    if missing:
        raise ValueError(f"SSA dataset {path} missing required keys: {missing}")
    obs = blob["obs"].astype(np.float32)
    action = blob["action"].astype(np.float32)
    state = blob["state"].astype(np.float32)
    if obs.ndim != 4 or obs.shape[-1] != 25:
        raise ValueError(f"obs must be (E,T,S,25), got {obs.shape}")
    if action.ndim != 4 or action.shape[-1] != len(SSA_MODE_LIST):
        raise ValueError(f"action must be (E,T,S,{len(SSA_MODE_LIST)}), got {action.shape}")
    if state.ndim != 4 or state.shape[:3] != obs.shape[:3]:
        raise ValueError(f"state must be (E,T,S,·) matching obs, got {state.shape}")
    if obs.shape[:3] != action.shape[:3]:
        raise ValueError("obs/action episode-time-sat axes must match")
    onehot = action.sum(axis=-1)
    if not np.allclose(onehot, 1.0, atol=1e-4):
        raise ValueError("action must be one-hot over SSA modes")
    for arr in (obs, action, state):
        if not np.isfinite(arr).all():
            raise ValueError("dataset contains non-finite values")
    sat_ids = tuple(str(x) for x in blob["sat_ids"].tolist()) if "sat_ids" in blob else tuple(
        f"sat_{i}" for i in range(obs.shape[2])
    )
    metadata = (
        json.loads(Path(metadata_path).read_text(encoding="utf-8")) if metadata_path else load_metadata(path)
    )
    return SSAWorldModelDataset(
        path=path,
        obs=obs,
        action=action,
        state=state,
        reward=blob["reward"].astype(np.float32),
        mode=blob["mode"].astype(np.int64),
        resolved_mode=blob["resolved_mode"].astype(np.int64),
        forced_mode=blob["forced_mode"].astype(np.float32),
        episode_seed=blob["episode_seed"].astype(np.int64),
        sat_ids=sat_ids,
        delivered_coverage=blob["delivered_coverage"].astype(np.float32),
        onboard_coverage=blob["onboard_coverage"].astype(np.float32),
        archive_records=blob["archive_records"].astype(np.int64),
        metadata=metadata,
    )


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
    "SSA_MODE_LIST",
    "SSA_MODE_TO_INDEX",
    "SSAWorldModelDataset",
    "TrajectoryBatch",
    "WorldModelDataset",
    "load_metadata",
    "load_ssa_world_model_dataset",
    "load_trajectory_npz",
    "load_world_model_dataset",
    "save_trajectory_npz",
    "validate_trajectory_arrays",
]
