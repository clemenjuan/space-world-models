"""EventSat-Lite macro-action wrapper.

This curriculum keeps the same simplified EventSat physics and state vector, but
reduces the command space from seven low-level modes to four operational macro
actions:

- charge
- observe
- process_to_obc: compress, detect, or send depending on pipeline backlog
- downlink

The intent is to let the world model learn the causal data pipeline before we
ask a planner to discover the full 7-action sequencing problem.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from swm_eventsat.data.toy_eventsat_env import (
    MODE_LIST,
    MODE_TO_INDEX,
    OBS_DIM,
    STATE_NAMES,
    EventSatEnv,
)


LITE_MODE_LIST = (
    "charge",
    "observe",
    "process_to_obc",
    "downlink",
)
LITE_MODE_TO_INDEX = {mode: idx for idx, mode in enumerate(LITE_MODE_LIST)}
LITE_ACTION_DIM = len(LITE_MODE_LIST)
BASE_TO_LITE_MODE = {
    "charging": "charge",
    "payload_observe": "observe",
    "payload_compress": "process_to_obc",
    "payload_detect": "process_to_obc",
    "payload_send": "process_to_obc",
    "communication": "downlink",
    "safe": "charge",
}


class EventSatLiteEnv(gym.Env):
    """Four-action macro wrapper over :class:`EventSatEnv`."""

    metadata = {"render_modes": []}

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.base = EventSatEnv(*args, **kwargs)
        self.observation_space = self.base.observation_space
        self.action_space = spaces.Discrete(LITE_ACTION_DIM)

    def __getattr__(self, name: str) -> Any:
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    @property
    def data_stored_mb(self) -> float:
        return self.base.data_stored_mb

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.base.reset(seed=seed, options=options)
        return obs, self._lite_info(
            info,
            requested_lite="charge",
            base_command="charging",
        )

    def step(self, action: int):
        requested_idx = int(action)
        if requested_idx < 0 or requested_idx >= LITE_ACTION_DIM:
            requested_idx = LITE_MODE_TO_INDEX["charge"]
        requested_lite = LITE_MODE_LIST[requested_idx]
        base_command = self._macro_to_base_mode(requested_lite)
        obs, reward, terminated, truncated, info = self.base.step(MODE_TO_INDEX[base_command])
        return (
            obs,
            reward,
            terminated,
            truncated,
            self._lite_info(info, requested_lite=requested_lite, base_command=base_command),
        )

    def _macro_to_base_mode(self, requested_lite: str) -> str:
        if requested_lite == "charge":
            return "charging"
        if requested_lite == "observe":
            return "payload_observe"
        if requested_lite == "downlink":
            return "communication"
        if requested_lite != "process_to_obc":
            return "charging"

        if self.base.uncompressed_observations > 0:
            return "payload_compress"
        if self.base.undetected_observations > 0:
            return "payload_detect"
        if self.base.jetson_compressed_mb > 0.01:
            return "payload_send"
        return "charging"

    def _lite_info(self, info: dict[str, Any], requested_lite: str, base_command: str) -> dict[str, Any]:
        resolved_base = info["resolved_mode"]
        resolved_lite = BASE_TO_LITE_MODE.get(resolved_base, "charge")
        out = dict(info)
        out.update(
            {
                "lite_mode_names": LITE_MODE_LIST,
                "requested_lite_mode": requested_lite,
                "requested_lite_mode_idx": LITE_MODE_TO_INDEX[requested_lite],
                "resolved_lite_mode": resolved_lite,
                "resolved_lite_mode_idx": LITE_MODE_TO_INDEX[resolved_lite],
                "base_requested_mode": base_command,
                "base_requested_mode_idx": MODE_TO_INDEX[base_command],
                "base_resolved_mode": resolved_base,
                "base_resolved_mode_idx": MODE_TO_INDEX[resolved_base],
                "forced_lite_mode": requested_lite != resolved_lite,
            }
        )
        return out


def heuristic_eventsat_lite_policy(
    env: EventSatLiteEnv,
    rng: np.random.Generator | None = None,
    exploration: float = 0.0,
) -> int:
    """Macro-action operator policy for the Lite curriculum."""
    rng = rng or np.random.default_rng()
    if exploration > 0.0 and rng.random() < exploration:
        return int(rng.integers(0, LITE_ACTION_DIM))

    if env.battery_soc < 0.50:
        return LITE_MODE_TO_INDEX["charge"]
    if env.is_ground_pass_active() and env.obc_data_mb > 0.05:
        return LITE_MODE_TO_INDEX["downlink"]
    if (
        env.uncompressed_observations > 0
        or env.undetected_observations > 0
        or env.jetson_compressed_mb > 0.05
    ):
        return LITE_MODE_TO_INDEX["process_to_obc"]
    if env.battery_soc > 0.62 and env.data_stored_mb < 0.20 * env.storage_capacity_mb:
        return LITE_MODE_TO_INDEX["observe"]
    return LITE_MODE_TO_INDEX["charge"]


def balanced_eventsat_lite_policy(
    env: EventSatLiteEnv,
    rng: np.random.Generator | None = None,
    exploration: float = 0.18,
) -> int:
    """Data-collection policy with broader macro-action coverage."""
    rng = rng or np.random.default_rng()
    if rng.random() < exploration:
        return int(rng.integers(0, LITE_ACTION_DIM))

    candidates: list[tuple[int, float]] = [(LITE_MODE_TO_INDEX["charge"], 0.25)]
    if env.battery_soc < 0.45:
        candidates.append((LITE_MODE_TO_INDEX["charge"], 3.0))
    if env.is_ground_pass_active() and env.obc_data_mb > 0.05:
        candidates.append((LITE_MODE_TO_INDEX["downlink"], 4.0))
    if (
        env.uncompressed_observations > 0
        or env.undetected_observations > 0
        or env.jetson_compressed_mb > 0.05
    ):
        candidates.append((LITE_MODE_TO_INDEX["process_to_obc"], 3.0))
    if env.battery_soc > 0.52 and env.data_stored_mb < 0.75 * env.storage_capacity_mb:
        candidates.append((LITE_MODE_TO_INDEX["observe"], 2.0))

    actions = np.asarray([item[0] for item in candidates], dtype=np.int64)
    weights = np.asarray([item[1] for item in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    return int(rng.choice(actions, p=weights))


__all__ = [
    "BASE_TO_LITE_MODE",
    "LITE_ACTION_DIM",
    "LITE_MODE_LIST",
    "LITE_MODE_TO_INDEX",
    "OBS_DIM",
    "STATE_NAMES",
    "EventSatLiteEnv",
    "balanced_eventsat_lite_policy",
    "heuristic_eventsat_lite_policy",
]
