"""Simplified nominal EventSat mode-scheduling environment.

This is the first small operations-world-model problem in this repo. It keeps
the EventSat-flavoured mode and data-pipeline logic from AUTOPS, but removes the
large experiment framework: one satellite, analytic sunlight/contact timing,
nominal health only, and a 7-way discrete mode command.

Observation is a normalized 25D vector mirroring the AUTOPS Gymnasium wrapper:
resources, orbital timing, flags, pipeline counters, and current mode one-hot.
Actions are commanded modes; the environment may resolve invalid commands to a
safe fallback such as charging.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


MODE_LIST = (
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
)
MODE_TO_INDEX = {mode: i for i, mode in enumerate(MODE_LIST)}

OBS_DIM = 25
ACTION_DIM = len(MODE_LIST)

STATE_NAMES = (
    "battery_soc",
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
    "current_mode_idx",
    "in_sunlight",
    "ground_pass_active",
    "time_to_next_eclipse_steps",
    "time_to_next_pass_steps",
)


class EventSatEnv(gym.Env):
    """Single-satellite nominal EventSat operations environment.

    The simulator is intentionally compact, but the causal structure is the
    important part:

    - charging changes battery depending on sunlight;
    - observation creates raw Jetson data;
    - compression is a multi-step raw -> compressed pipeline;
    - detection is a multi-step metadata pipeline;
    - payload_send moves compressed data to the OBC;
    - communication downlinks OBC data only during contact windows.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 512,
        step_duration_s: float = 60.0,
        randomize_phase: bool = True,
        pass_interval_orbits: float = 2.0,
        pass_duration_steps: int = 6,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.step_duration_s = float(step_duration_s)
        self.randomize_phase = bool(randomize_phase)

        # Orbit / timing, EventSat-ish 400 km SSO numbers.
        self.orbital_period_s = 5554.0
        self.orbital_period_steps = max(1, int(round(self.orbital_period_s / self.step_duration_s)))
        self.eclipse_fraction = 0.36
        self.pass_interval_steps = max(
            int(pass_duration_steps) + 1,
            int(round(pass_interval_orbits * self.orbital_period_steps)),
        )
        self.pass_duration_steps = max(1, int(pass_duration_steps))
        self.max_pass_steps_for_obs = 10.0

        # Power, storage, payload, and communications parameters from the
        # AUTOPS EventSat scenario, lightly rounded where needed.
        self.solar_generation_w = 24.0 * 0.70
        self.battery_capacity_wh = 70.0
        self.initial_soc = 0.8
        self.min_soc = 0.2
        self.charge_efficiency = 0.9
        self.observe_min_soc = 0.4
        self.compress_min_soc = 0.3
        self.detect_min_soc = 0.3
        self.send_min_soc = 0.3
        self.storage_capacity_mb = 4096.0
        self.jetson_capacity_mb = 249036.8
        self.observation_size_mb = 9.41
        self.compression_ratio = 5.11
        self.jetson_to_obc_rate_kbps = 8000.0
        self.downlink_rate_kbps = 50.0
        self.compression_steps = 2
        self.detection_steps = 5
        self.detection_metadata_mb = 0.01
        self.daily_downlink_budget_mb = 27.0

        self.consumption_w = {
            "charging": (4.72, 4.32),
            "communication": (33.65, 33.24),
            "payload_observe": (17.94, 17.55),
            "payload_compress": (12.77, 12.37),
            "payload_detect": (12.77, 12.37),
            "payload_send": (12.77, 12.37),
            "safe": (9.58, 9.58),
        }

        self.observation_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        self.current_step = 0
        self._phase_offset_steps = 0
        self._pass_offset_steps = 0
        self._reset_state()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.randomize_phase:
            self._phase_offset_steps = int(self.np_random.integers(0, self.orbital_period_steps))
            self._pass_offset_steps = int(self.np_random.integers(0, self.orbital_period_steps))
        else:
            self._phase_offset_steps = 0
            self._pass_offset_steps = int(round(0.18 * self.orbital_period_steps))
        self._reset_state()
        return self._observation(), self._info(
            requested_mode="charging",
            resolved_mode=self.current_mode,
            reward=0.0,
            action_info={},
        )

    def step(self, action: int):
        requested_idx = int(action)
        if requested_idx < 0 or requested_idx >= ACTION_DIM:
            requested_idx = MODE_TO_INDEX["charging"]
        requested_mode = MODE_LIST[requested_idx]
        resolved_mode = self._resolve_mode(requested_mode)

        in_sun = self.is_in_sunlight()
        pass_active = self.is_ground_pass_active()
        self._update_battery(resolved_mode, in_sun)
        reward, action_info = self._apply_mode_effects(resolved_mode, pass_active)

        self.current_mode = resolved_mode
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.max_steps
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(
                requested_mode=requested_mode,
                resolved_mode=resolved_mode,
                reward=reward,
                action_info=action_info,
            ),
        )

    def is_in_sunlight(self, step: int | None = None) -> bool:
        step = self.current_step if step is None else int(step)
        phase = self._orbital_phase(step)
        return phase < (1.0 - self.eclipse_fraction)

    def is_ground_pass_active(self, step: int | None = None) -> bool:
        step = self.current_step if step is None else int(step)
        rel = step - self._pass_offset_steps
        return rel >= 0 and (rel % self.pass_interval_steps) < self.pass_duration_steps

    @property
    def data_stored_mb(self) -> float:
        return self.jetson_raw_mb + self.jetson_compressed_mb + self.obc_data_mb

    def _reset_state(self) -> None:
        self.battery_soc = self.initial_soc
        self.jetson_raw_mb = 0.0
        self.jetson_compressed_mb = 0.0
        self.obc_data_mb = 0.0
        self.data_downlinked_mb = 0.0
        self.total_raw_captured_mb = 0.0
        self.uncompressed_observations = 0
        self.undetected_observations = 0
        self.compression_progress = 0
        self.detection_progress = 0
        self.total_detections = 0
        self.total_observation_s = 0.0
        self.current_mode = "charging"

    def _resolve_mode(self, requested_mode: str) -> str:
        if requested_mode not in MODE_TO_INDEX:
            return "charging"
        if self.battery_soc <= self.min_soc and requested_mode != "safe":
            return "safe"
        if requested_mode == "communication" and not self.is_ground_pass_active():
            return "charging"
        if requested_mode == "payload_observe" and self.battery_soc < self.observe_min_soc:
            return "charging"
        if requested_mode == "payload_compress" and self.battery_soc < self.compress_min_soc:
            return "charging"
        if requested_mode == "payload_detect" and self.battery_soc < self.detect_min_soc:
            return "charging"
        if requested_mode == "payload_send" and self.battery_soc < self.send_min_soc:
            return "charging"
        return requested_mode

    def _update_battery(self, mode: str, in_sun: bool) -> None:
        sun_w, eclipse_w = self.consumption_w[mode]
        consumption_w = sun_w if in_sun else eclipse_w
        generation_w = self.solar_generation_w if in_sun else 0.0
        delta_wh = (generation_w - consumption_w) * (self.step_duration_s / 3600.0)
        if delta_wh > 0:
            delta_wh *= self.charge_efficiency
        self.battery_soc = float(np.clip(self.battery_soc + delta_wh / self.battery_capacity_wh, 0.0, 1.0))

    def _apply_mode_effects(self, mode: str, pass_active: bool) -> tuple[float, dict[str, Any]]:
        info: dict[str, Any] = {"pass_active": bool(pass_active)}
        if mode != "payload_compress":
            self.compression_progress = 0
        if mode != "payload_detect":
            self.detection_progress = 0

        if mode == "payload_observe":
            self.total_observation_s += self.step_duration_s
            self.uncompressed_observations += 1
            self.jetson_raw_mb += self.observation_size_mb
            self.total_raw_captured_mb += self.observation_size_mb
            overflow = self.jetson_raw_mb > self.jetson_capacity_mb
            self.jetson_raw_mb = min(self.jetson_raw_mb, self.jetson_capacity_mb)
            info["storage_overflow"] = overflow

        elif mode == "payload_compress":
            had_data = self.uncompressed_observations > 0
            if had_data:
                self.compression_progress += 1
                if self.compression_progress >= self.compression_steps:
                    self.uncompressed_observations -= 1
                    self.jetson_raw_mb = max(0.0, self.jetson_raw_mb - self.observation_size_mb)
                    self.jetson_compressed_mb = min(
                        self.jetson_capacity_mb,
                        self.jetson_compressed_mb + self.observation_size_mb / self.compression_ratio,
                    )
                    self.undetected_observations += 1
                    self.compression_progress = 0
                    info["compression_completed"] = True
                else:
                    info["compression_in_progress"] = True
            info["had_data_to_compress"] = had_data

        elif mode == "payload_detect":
            had_data = self.undetected_observations > 0
            if had_data:
                self.detection_progress += 1
                if self.detection_progress >= self.detection_steps:
                    self.undetected_observations -= 1
                    space = max(0.0, self.storage_capacity_mb - self.obc_data_mb)
                    self.obc_data_mb += min(space, self.detection_metadata_mb)
                    self.total_detections += 1
                    self.detection_progress = 0
                    info["detection_completed"] = True
                else:
                    info["detection_in_progress"] = True
            info["had_data_to_detect"] = had_data

        elif mode == "payload_send":
            had_data = self.jetson_compressed_mb > 0.0
            transfer_mb = self._jetson_to_obc_capacity_mb()
            space = max(0.0, self.storage_capacity_mb - self.obc_data_mb)
            actual = min(self.jetson_compressed_mb, transfer_mb, space)
            self.jetson_compressed_mb -= actual
            self.obc_data_mb += actual
            info["had_data_to_send"] = had_data
            info["data_sent_mb"] = actual

        elif mode == "communication":
            actual = 0.0
            if pass_active:
                actual = min(self.obc_data_mb, self._downlink_capacity_mb())
                self.obc_data_mb -= actual
                self.data_downlinked_mb += actual
            info["data_downlinked_mb"] = actual

        reward = self._reward(mode, info)
        return reward, info

    def _reward(self, mode: str, action_info: dict[str, Any]) -> float:
        storage_ratio = self.data_stored_mb / self.storage_capacity_mb
        resource_penalty = 0.0
        if self.battery_soc < 0.3:
            resource_penalty -= (0.3 - self.battery_soc) / 0.3
        if storage_ratio > 0.8:
            resource_penalty -= (storage_ratio - 0.8) / 0.2

        action_reward = 0.0
        if mode == "payload_observe":
            action_reward = 1.0 - (0.5 if action_info.get("storage_overflow") else 0.0)
        elif mode == "payload_compress":
            action_reward = 0.5 if action_info.get("had_data_to_compress") else -0.1
        elif mode == "payload_detect":
            action_reward = 0.5 if action_info.get("had_data_to_detect") else -0.1
        elif mode == "payload_send":
            action_reward = 0.25 if action_info.get("had_data_to_send") else -0.1
        elif mode == "communication":
            action_reward = min(float(action_info.get("data_downlinked_mb", 0.0)), 5.0)
        elif mode == "charging":
            action_reward = -0.05
        elif mode == "safe":
            action_reward = -0.3

        obs_target_h = max(1e-6, (2.0 / 90.0) * (self.max_steps * self.step_duration_s / 86400.0))
        dl_target_mb = max(1e-6, (221.0 / 90.0) * (self.max_steps * self.step_duration_s / 86400.0))
        obs_gap = max(0.0, 1.0 - (self.total_observation_s / 3600.0) / obs_target_h)
        dl_gap = max(0.0, 1.0 - self.data_downlinked_mb / dl_target_mb)
        progress = min(1.0, self.current_step / max(1, self.max_steps))
        mission_penalty = -0.5 * (obs_gap + dl_gap) * progress
        return 0.01 * (resource_penalty + action_reward + mission_penalty)

    def _observation(self) -> np.ndarray:
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        vec[0] = self.battery_soc
        vec[1] = self.obc_data_mb / self.storage_capacity_mb
        vec[2] = self.jetson_raw_mb / self.jetson_capacity_mb
        vec[3] = self.jetson_compressed_mb / self.jetson_capacity_mb

        phase = self._orbital_phase(self.current_step)
        vec[4] = math.sin(phase * 2.0 * math.pi)
        vec[5] = math.cos(phase * 2.0 * math.pi)
        vec[6] = min(self._time_to_next_eclipse_steps() / self.orbital_period_steps, 1.0)
        vec[7] = min(self._time_to_next_pass_steps() / self.orbital_period_steps, 1.0)
        vec[8] = min(self._remaining_pass_steps() / self.max_pass_steps_for_obs, 1.0)
        vec[9] = self.current_step / max(1, self.max_steps)

        vec[10] = 1.0 if self.is_in_sunlight() else 0.0
        vec[11] = 1.0 if self.is_ground_pass_active() else 0.0
        vec[12] = 1.0

        vec[13] = min(self.uncompressed_observations / 10.0, 1.0)
        vec[14] = min(self.compression_progress / max(1, self.compression_steps), 1.0)
        vec[15] = min(self.undetected_observations / 10.0, 1.0)
        vec[16] = min(self.detection_progress / max(1, self.detection_steps), 1.0)
        vec[17] = min(self.data_downlinked_mb / self.daily_downlink_budget_mb, 2.0)

        vec[18 + MODE_TO_INDEX[self.current_mode]] = 1.0
        return np.nan_to_num(vec, nan=0.0, posinf=2.0, neginf=0.0).astype(np.float32)

    def _info(
        self,
        requested_mode: str,
        resolved_mode: str,
        reward: float,
        action_info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": self._state_vector(),
            "state_names": STATE_NAMES,
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "requested_mode_idx": MODE_TO_INDEX[requested_mode],
            "resolved_mode_idx": MODE_TO_INDEX[resolved_mode],
            "forced_mode": requested_mode != resolved_mode,
            "reward": float(reward),
            "in_sunlight": self.is_in_sunlight(),
            "ground_pass_active": self.is_ground_pass_active(),
            "data_stored_mb": self.data_stored_mb,
            **action_info,
        }

    def _state_vector(self) -> np.ndarray:
        return np.array(
            [
                self.battery_soc,
                self.obc_data_mb,
                self.jetson_raw_mb,
                self.jetson_compressed_mb,
                self.data_downlinked_mb,
                float(self.uncompressed_observations),
                float(self.compression_progress),
                float(self.undetected_observations),
                float(self.detection_progress),
                self.total_observation_s,
                float(self.total_detections),
                float(MODE_TO_INDEX[self.current_mode]),
                float(self.is_in_sunlight()),
                float(self.is_ground_pass_active()),
                float(self._time_to_next_eclipse_steps()),
                float(self._time_to_next_pass_steps()),
            ],
            dtype=np.float32,
        )

    def _orbital_phase(self, step: int) -> float:
        local = (int(step) + self._phase_offset_steps) % self.orbital_period_steps
        return local / self.orbital_period_steps

    def _time_to_next_eclipse_steps(self) -> int:
        local = (self.current_step + self._phase_offset_steps) % self.orbital_period_steps
        eclipse_start = int(round((1.0 - self.eclipse_fraction) * self.orbital_period_steps))
        if local < eclipse_start:
            return eclipse_start - local
        return self.orbital_period_steps - local + eclipse_start

    def _time_to_next_pass_steps(self) -> int:
        if self.is_ground_pass_active():
            return 0
        rel = self.current_step - self._pass_offset_steps
        if rel < 0:
            return -rel
        return self.pass_interval_steps - (rel % self.pass_interval_steps)

    def _remaining_pass_steps(self) -> int:
        rel = self.current_step - self._pass_offset_steps
        if rel < 0:
            return 0
        within = rel % self.pass_interval_steps
        if within >= self.pass_duration_steps:
            return 0
        return self.pass_duration_steps - within

    def _jetson_to_obc_capacity_mb(self) -> float:
        return (self.jetson_to_obc_rate_kbps / 8.0) * (self.step_duration_s / 1000.0)

    def _downlink_capacity_mb(self) -> float:
        return (self.downlink_rate_kbps / 8.0) * (self.step_duration_s / 1000.0)


def heuristic_eventsat_policy(
    env: EventSatEnv,
    rng: np.random.Generator | None = None,
    exploration: float = 0.0,
) -> int:
    """Small nominal operator policy for trajectory generation.

    It is deliberately simple: keep power healthy, exploit ground passes, and
    cycle observations through observe -> compress -> detect -> send -> downlink.
    A little random exploration gives the world model off-nominal-but-safe mode
    transitions without making the dataset mostly invalid commands.
    """
    rng = rng or np.random.default_rng()
    if exploration > 0.0 and rng.random() < exploration:
        # Exclude safe from nominal exploration; safety/anomaly data can be a
        # later experiment once the base dynamics are learned.
        return int(rng.integers(0, MODE_TO_INDEX["safe"]))

    if env.battery_soc < 0.50:
        return MODE_TO_INDEX["charging"]
    if env.is_ground_pass_active() and env.obc_data_mb > 0.05:
        return MODE_TO_INDEX["communication"]
    if env.uncompressed_observations > 0:
        return MODE_TO_INDEX["payload_compress"]
    if env.undetected_observations > 0:
        return MODE_TO_INDEX["payload_detect"]
    if env.jetson_compressed_mb > 0.05 and env.obc_data_mb < 0.95 * env.storage_capacity_mb:
        return MODE_TO_INDEX["payload_send"]
    if env.battery_soc > 0.62 and env.data_stored_mb < 0.05 * env.storage_capacity_mb:
        return MODE_TO_INDEX["payload_observe"]
    return MODE_TO_INDEX["charging"]
