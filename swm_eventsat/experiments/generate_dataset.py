"""Roll out nominal EventSat mode-dynamics episodes and save .npz trajectories."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swm_eventsat.data.toy_eventsat_env import ACTION_DIM, MODE_LIST, MODE_TO_INDEX, EventSatEnv, heuristic_eventsat_policy


def _one_hot(index: int, dim: int = ACTION_DIM) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _sample_action(env: EventSatEnv, rng: np.random.Generator, policy: str, exploration: float) -> int:
    if policy == "random":
        # Keep the first nominal dataset out of safe-mode/anomaly territory.
        return int(rng.integers(0, ACTION_DIM - 1))
    if policy != "heuristic":
        raise ValueError(f"unknown EventSat rollout policy: {policy!r}")
    return heuristic_eventsat_policy(env, rng=rng, exploration=exploration)


def generate(
    n_episodes: int = 64,
    episode_len: int = 256,
    out_path: str = "data/cache/eventsat_trajectories.npz",
    seed: int = 0,
    policy: str = "heuristic",
    exploration: float = 0.08,
):
    """Generate EventSat operations trajectories.

    The saved schema matches the generated generators:

    - obs:    (episode, time, 25) normalized EventSat observation vector
    - action: (episode, time, 7) one-hot commanded mode
    - state:  (episode, time, 16) raw diagnostic state
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    obs_all, act_all, state_all = [], [], []
    reward_all, mode_all = [], []
    resolved_all, forced_all = [], []

    for ep in range(int(n_episodes)):
        rng = np.random.default_rng(seed + ep * 9973)
        env = EventSatEnv(max_steps=episode_len)
        obs, info = env.reset(seed=seed + ep)

        obs_ep, act_ep, state_ep = [], [], []
        reward_ep, mode_ep, resolved_ep, forced_ep = [], [], [], []
        for t in range(int(episode_len)):
            action = _sample_action(env, rng, policy=policy, exploration=exploration)
            resolved_mode = env._resolve_mode(MODE_LIST[action])
            resolved_idx = MODE_TO_INDEX[resolved_mode]
            obs_ep.append(obs)
            act_ep.append(_one_hot(action))
            state_ep.append(info["state"])
            mode_ep.append(action)
            resolved_ep.append(resolved_idx)
            forced_ep.append(float(resolved_idx != action))

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

    np.savez(
        out_path,
        obs=np.asarray(obs_all, dtype=np.float32),
        action=np.asarray(act_all, dtype=np.float32),
        state=np.asarray(state_all, dtype=np.float32),
        reward=np.asarray(reward_all, dtype=np.float32),
        mode=np.asarray(mode_all, dtype=np.int64),
        resolved_mode=np.asarray(resolved_all, dtype=np.int64),
        forced_mode=np.asarray(forced_all, dtype=np.float32),
        mode_names=np.asarray(MODE_LIST),
        policy=np.asarray(policy),
        exploration=np.asarray(float(exploration), dtype=np.float32),
    )
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=64)
    p.add_argument("--episode-len", type=int, default=256)
    p.add_argument("--out", default="data/cache/eventsat_trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    p.add_argument("--exploration", type=float, default=0.08)
    args = p.parse_args()
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
