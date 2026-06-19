"""Generate EventSat-Lite macro-action trajectories.

The generator deliberately mixes initial pipeline states so the model sees
observe, process, and downlink transitions often enough to learn them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.eventsat_env import MODE_LIST, MODE_TO_INDEX
from envs.eventsat_lite_env import (
    BASE_TO_LITE_MODE,
    LITE_ACTION_DIM,
    LITE_MODE_LIST,
    LITE_MODE_TO_INDEX,
    EventSatLiteEnv,
    balanced_eventsat_lite_policy,
    heuristic_eventsat_lite_policy,
)


SCENARIOS = (
    "empty_nominal",
    "raw_backlog",
    "compressed_backlog",
    "obc_backlog",
    "low_battery",
    "storage_pressure",
)


def _one_hot(index: int, dim: int = LITE_ACTION_DIM) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _apply_scenario(env: EventSatLiteEnv, scenario: str, rng: np.random.Generator) -> None:
    base = env.base
    if scenario == "empty_nominal":
        return
    if scenario == "raw_backlog":
        n = int(rng.integers(2, 9))
        base.uncompressed_observations = n
        base.jetson_raw_mb = float(n * base.observation_size_mb)
        base.total_raw_captured_mb = base.jetson_raw_mb
    elif scenario == "compressed_backlog":
        n = int(rng.integers(2, 12))
        base.jetson_compressed_mb = float(n * base.observation_size_mb / base.compression_ratio)
        base.undetected_observations = int(rng.integers(0, max(1, n // 2 + 1)))
    elif scenario == "obc_backlog":
        base.obc_data_mb = float(rng.uniform(8.0, 80.0))
        base.data_downlinked_mb = float(rng.uniform(0.0, 5.0))
    elif scenario == "low_battery":
        base.battery_soc = float(rng.uniform(0.30, 0.48))
    elif scenario == "storage_pressure":
        base.battery_soc = float(rng.uniform(0.62, 0.90))
        base.obc_data_mb = float(rng.uniform(0.55, 0.80) * base.storage_capacity_mb)
        base.jetson_compressed_mb = float(rng.uniform(100.0, 400.0))
        base.uncompressed_observations = int(rng.integers(1, 6))
        base.jetson_raw_mb = float(base.uncompressed_observations * base.observation_size_mb)


def _sample_action(env: EventSatLiteEnv, rng: np.random.Generator, policy: str, exploration: float) -> int:
    if policy == "random":
        return int(rng.integers(0, LITE_ACTION_DIM))
    if policy == "heuristic":
        return heuristic_eventsat_lite_policy(env, rng=rng, exploration=exploration)
    if policy == "balanced":
        return balanced_eventsat_lite_policy(env, rng=rng, exploration=exploration)
    raise ValueError(f"unknown EventSat-Lite policy: {policy!r}")


def generate(
    n_episodes: int = 128,
    episode_len: int = 256,
    out_path: str = "data/cache/eventsat_lite_trajectories.npz",
    seed: int = 0,
    policy: str = "balanced",
    exploration: float = 0.18,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    obs_all, act_all, state_all = [], [], []
    reward_all, mode_all, resolved_all, forced_all = [], [], [], []
    base_mode_all, base_resolved_all = [], []
    scenario_all = []

    for ep in range(int(n_episodes)):
        rng = np.random.default_rng(seed + ep * 9973)
        scenario = SCENARIOS[ep % len(SCENARIOS)]
        env = EventSatLiteEnv(max_steps=episode_len)
        obs, info = env.reset(seed=seed + ep)
        _apply_scenario(env, scenario, rng)
        obs = env.base._observation()
        info = env._lite_info(
            env.base._info(
                requested_mode="charging",
                resolved_mode=env.base.current_mode,
                reward=0.0,
                action_info={},
            ),
            requested_lite="charge",
            base_command="charging",
        )

        obs_ep, act_ep, state_ep = [], [], []
        reward_ep, mode_ep, resolved_ep, forced_ep = [], [], [], []
        base_mode_ep, base_resolved_ep = [], []

        for t in range(int(episode_len)):
            action = _sample_action(env, rng, policy=policy, exploration=exploration)
            lite_name = LITE_MODE_LIST[action]
            base_command = env._macro_to_base_mode(lite_name)
            base_resolved = env.base._resolve_mode(base_command)
            resolved_lite = BASE_TO_LITE_MODE.get(base_resolved, "charge")
            resolved_idx = LITE_MODE_TO_INDEX[resolved_lite]

            obs_ep.append(obs)
            act_ep.append(_one_hot(action))
            state_ep.append(info["state"])
            mode_ep.append(action)
            resolved_ep.append(resolved_idx)
            forced_ep.append(float(resolved_idx != action))
            base_mode_ep.append(MODE_TO_INDEX[base_command])
            base_resolved_ep.append(MODE_TO_INDEX[base_resolved])

            if t < episode_len - 1:
                obs, reward, _, _, info = env.step(action)
                reward_ep.append(float(reward))
            else:
                reward_ep.append(0.0)

        obs_all.append(obs_ep)
        act_all.append(act_ep)
        state_all.append(state_ep)
        reward_all.append(reward_ep)
        mode_all.append(mode_ep)
        resolved_all.append(resolved_ep)
        forced_all.append(forced_ep)
        base_mode_all.append(base_mode_ep)
        base_resolved_all.append(base_resolved_ep)
        scenario_all.append(scenario)

    np.savez(
        out_path,
        obs=np.asarray(obs_all, dtype=np.float32),
        action=np.asarray(act_all, dtype=np.float32),
        state=np.asarray(state_all, dtype=np.float32),
        reward=np.asarray(reward_all, dtype=np.float32),
        mode=np.asarray(mode_all, dtype=np.int64),
        resolved_mode=np.asarray(resolved_all, dtype=np.int64),
        forced_mode=np.asarray(forced_all, dtype=np.float32),
        base_mode=np.asarray(base_mode_all, dtype=np.int64),
        base_resolved_mode=np.asarray(base_resolved_all, dtype=np.int64),
        mode_names=np.asarray(LITE_MODE_LIST),
        base_mode_names=np.asarray(MODE_LIST),
        scenario=np.asarray(scenario_all),
        policy=np.asarray(policy),
        exploration=np.asarray(float(exploration), dtype=np.float32),
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=128)
    parser.add_argument("--episode-len", type=int, default=256)
    parser.add_argument("--out", default="data/cache/eventsat_lite_trajectories.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("balanced", "heuristic", "random"), default="balanced")
    parser.add_argument("--exploration", type=float, default=0.18)
    args = parser.parse_args()
    print(
        "wrote",
        generate(
            n_episodes=args.n_episodes,
            episode_len=args.episode_len,
            out_path=args.out,
            seed=args.seed,
            policy=args.policy,
            exploration=args.exploration,
        ),
    )
