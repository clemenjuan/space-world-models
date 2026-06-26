"""Thin adapter from AUTOPS EventSat to the local LeWM trajectory schema."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from swm_eventsat.schema import (
    EVENTSAT_MODE_LIST,
    EVENTSAT_MODE_TO_INDEX,
    TrajectoryBatch,
    save_trajectory_npz,
)


DEFAULT_AUTOPS_ROOT = Path.home() / "autops-agentic-framework"


def ensure_autops_on_path(autops_root: str | Path = DEFAULT_AUTOPS_ROOT) -> Path:
    """Make the sibling AUTOPS repository importable."""
    root = Path(autops_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"AUTOPS repository not found: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def one_hot_mode(index: int, dim: int = len(EVENTSAT_MODE_LIST)) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def observation_to_vector(observation: Any, env: Any) -> np.ndarray:
    """Use AUTOPS' Gymnasium encoder for its canonical 25D EventSat vector."""
    from src.eventsat.gymnasium_wrapper import EventSatGymnasium

    wrapper = object.__new__(EventSatGymnasium)
    wrapper._env = env
    return wrapper._obs_to_vector(observation)


def state_from_info(info: dict[str, Any]) -> np.ndarray:
    """Map AUTOPS step info to a local EventSat state vector."""
    return np.asarray(
        [
            info.get("battery_soc", 0.0),
            info.get("obc_data_mb", 0.0),
            info.get("jetson_raw_mb", 0.0),
            info.get("jetson_compressed_mb", 0.0),
            info.get("data_downlinked_mb", 0.0),
            info.get("uncompressed_observations", 0.0),
            info.get("compression_progress", 0.0),
            info.get("undetected_observations", 0.0),
            info.get("detection_progress", 0.0),
            info.get("observation_hours", 0.0) * 3600.0,
            info.get("total_detections", 0.0),
            float(EVENTSAT_MODE_TO_INDEX.get(info.get("resolved_mode", "charging"), 0)),
            float(info.get("in_sunlight", 0.0)),
            float(info.get("ground_pass_active", 0.0)),
            info.get("time_to_next_eclipse", 0.0),
            info.get("time_to_next_pass", 0.0),
        ],
        dtype=np.float32,
    )


def heuristic_policy(env: Any, rng: np.random.Generator) -> int:
    """Small safe policy for AUTOPS trajectory export."""
    if env.battery_soc < 0.50:
        return EVENTSAT_MODE_TO_INDEX["charging"]
    if env._is_ground_pass_active() and env.obc_data_mb > 0.05:
        return EVENTSAT_MODE_TO_INDEX["communication"]
    if env.uncompressed_observations > 0:
        return EVENTSAT_MODE_TO_INDEX["payload_compress"]
    if env.undetected_observations > 0:
        return EVENTSAT_MODE_TO_INDEX["payload_detect"]
    if env.jetson_compressed_mb > 0.05 and env.obc_data_mb < 0.95 * env.storage_capacity_mb:
        return EVENTSAT_MODE_TO_INDEX["payload_send"]
    if env.battery_soc > 0.62 and env.data_stored_mb < 0.05 * env.storage_capacity_mb:
        return EVENTSAT_MODE_TO_INDEX["payload_observe"]
    return EVENTSAT_MODE_TO_INDEX["charging"]


def make_eventsat_env(
    autops_root: str | Path = DEFAULT_AUTOPS_ROOT,
    scenario_file: str = "configs/scenarios/eventsat.yaml",
    max_steps: int = 10080,
    step_duration_s: float = 60.0,
    **overrides: Any,
) -> Any:
    root = ensure_autops_on_path(autops_root)
    from src.eventsat.env import EventSatEnvironment

    config = {
        "constellation_size": 1,
        "step_duration_s": step_duration_s,
        "max_steps": int(max_steps),
        "scenario_file": str((root / scenario_file).resolve()),
        **overrides,
    }
    return EventSatEnvironment(config=config)


def rollout_autops_eventsat(
    n_episodes: int = 1,
    episode_len: int = 256,
    seed: int = 0,
    autops_root: str | Path = DEFAULT_AUTOPS_ROOT,
    policy: Callable[[Any, np.random.Generator], int] = heuristic_policy,
) -> TrajectoryBatch:
    """Roll AUTOPS EventSat and return a local LeWM trajectory batch."""
    obs_all, action_all, state_all = [], [], []
    reward_all, mode_all, resolved_all, forced_all = [], [], [], []

    for ep in range(int(n_episodes)):
        env = make_eventsat_env(autops_root=autops_root, max_steps=episode_len)
        rng = np.random.default_rng(seed + ep * 9973)
        observation = env.reset(seed=seed + ep)
        info = env.get_metrics()
        info.update(
            {
                "resolved_mode": env.current_mode,
                "requested_mode": env.current_mode,
                "forced": False,
                "in_sunlight": float(env._is_in_sunlight()),
                "ground_pass_active": float(env._is_ground_pass_active()),
            }
        )

        obs_ep, action_ep, state_ep = [], [], []
        reward_ep, mode_ep, resolved_ep, forced_ep = [], [], [], []

        for t in range(int(episode_len)):
            action_idx = int(policy(env, rng))
            mode_name = EVENTSAT_MODE_LIST[action_idx]
            obs_ep.append(observation_to_vector(observation, env))
            action_ep.append(one_hot_mode(action_idx))
            state_ep.append(state_from_info(info))
            mode_ep.append(action_idx)

            if t < int(episode_len) - 1:
                result = env.step({"eventsat_0": {"mode": mode_name}})
                observation = result.observation
                info = dict(result.info)
                reward_ep.append(float(sum(result.rewards.values())))
                resolved_name = str(info.get("resolved_mode", mode_name))
                resolved_idx = EVENTSAT_MODE_TO_INDEX.get(resolved_name, 0)
                resolved_ep.append(resolved_idx)
                forced_ep.append(float(info.get("forced", resolved_idx != action_idx)))
            else:
                reward_ep.append(0.0)
                resolved_ep.append(EVENTSAT_MODE_TO_INDEX.get(str(info.get("resolved_mode", mode_name)), 0))
                forced_ep.append(float(info.get("forced", 0.0)))

        obs_all.append(obs_ep)
        action_all.append(action_ep)
        state_all.append(state_ep)
        reward_all.append(reward_ep)
        mode_all.append(mode_ep)
        resolved_all.append(resolved_ep)
        forced_all.append(forced_ep)

    return TrajectoryBatch(
        obs=np.asarray(obs_all, dtype=np.float32),
        action=np.asarray(action_all, dtype=np.float32),
        state=np.asarray(state_all, dtype=np.float32),
        reward=np.asarray(reward_all, dtype=np.float32),
        mode=np.asarray(mode_all, dtype=np.int64),
        resolved_mode=np.asarray(resolved_all, dtype=np.int64),
        forced_mode=np.asarray(forced_all, dtype=np.float32),
        metadata={
            "source": "autops-agentic-framework",
            "mode_names": EVENTSAT_MODE_LIST,
            "policy": getattr(policy, "__name__", "callable"),
        },
    )


def export_autops_eventsat_npz(path: str | Path, **kwargs: Any) -> Path:
    """Roll AUTOPS EventSat and save a canonical trajectory archive."""
    return save_trajectory_npz(path, rollout_autops_eventsat(**kwargs))

