"""Action masks for EventSat scheduling planners."""
from __future__ import annotations

import numpy as np

from swm_eventsat.data.trajectory_schema import EVENTSAT_MODE_TO_INDEX


def first_action_mask(env: object, reserve_soc: float = 0.50, allow_unsafe: bool = False) -> np.ndarray:
    """Return a conservative 7-mode mask for the current EventSat-like state."""
    action_dim = len(EVENTSAT_MODE_TO_INDEX)
    if allow_unsafe:
        return np.ones(action_dim, dtype=bool)

    mask = np.zeros(action_dim, dtype=bool)
    mask[EVENTSAT_MODE_TO_INDEX["charging"]] = True
    min_soc = float(getattr(env, "min_soc", 0.2))
    battery_soc = float(getattr(env, "battery_soc", 1.0))
    if battery_soc <= min_soc + 0.03:
        mask[EVENTSAT_MODE_TO_INDEX["safe"]] = True
    if battery_soc < reserve_soc:
        return mask

    is_pass = getattr(env, "is_ground_pass_active", None) or getattr(env, "_is_ground_pass_active", None)
    pass_active = bool(is_pass()) if callable(is_pass) else False
    data_stored = float(getattr(env, "data_stored_mb", 0.0))
    obc = float(getattr(env, "obc_data_mb", 0.0))
    raw = int(getattr(env, "uncompressed_observations", 0))
    undetected = int(getattr(env, "undetected_observations", 0))
    compressed = float(getattr(env, "jetson_compressed_mb", 0.0))
    storage_capacity = float(getattr(env, "storage_capacity_mb", 4096.0))
    observation_size = float(getattr(env, "observation_size_mb", 9.41))

    mask[EVENTSAT_MODE_TO_INDEX["communication"]] = pass_active and obc > 0.01
    mask[EVENTSAT_MODE_TO_INDEX["payload_observe"]] = data_stored + observation_size <= storage_capacity
    mask[EVENTSAT_MODE_TO_INDEX["payload_compress"]] = raw > 0
    mask[EVENTSAT_MODE_TO_INDEX["payload_detect"]] = undetected > 0
    mask[EVENTSAT_MODE_TO_INDEX["payload_send"]] = compressed > 0.01 and obc < 0.98 * storage_capacity
    return mask

