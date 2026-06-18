"""Roll out or augment OdEnv episodes and save trajectories to .npz."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.od_env import OdEnv


def _geometry_rollout(episode_len):
    env = OdEnv(max_steps=episode_len)
    obs, info = env.reset(seed=0)
    del obs
    time_ep = [info["geometry"]["time_s"]]
    station_state_ep = [info["geometry"]["station_state_eci"]]
    basis_ep = [info["geometry"]["topocentric_basis_eci"]]
    for _ in range(episode_len - 1):
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        time_ep.append(info["geometry"]["time_s"])
        station_state_ep.append(info["geometry"]["station_state_eci"])
        basis_ep.append(info["geometry"]["topocentric_basis_eci"])
    return (
        np.asarray(time_ep, dtype=np.float32).squeeze(-1),
        np.asarray(station_state_ep, dtype=np.float32),
        np.asarray(basis_ep, dtype=np.float32),
    )


def augment_geometry(source_path, out_path):
    blob = np.load(source_path)
    payload = {key: blob[key] for key in blob.files}
    if "obs" not in payload or payload["obs"].ndim != 3:
        raise ValueError("source dataset must contain obs shaped (episode, time, feature)")
    episodes, episode_len = payload["obs"].shape[:2]
    time_s, station_state, basis = _geometry_rollout(episode_len)
    payload["time_s"] = np.broadcast_to(time_s, (episodes, episode_len)).copy()
    payload["station_state_eci"] = np.broadcast_to(
        station_state, (episodes, episode_len, station_state.shape[-1])
    ).copy()
    payload["topocentric_basis_eci"] = np.broadcast_to(
        basis, (episodes, episode_len, basis.shape[-2], basis.shape[-1])
    ).copy()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **payload)
    return out_path


def generate(
    n_episodes=64,
    episode_len=256,
    out_path="data/cache/od_trajectories.npz",
    seed=0,
    save_geometry=False,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    obs_all, act_all, state_all = [], [], []
    time_all, station_state_all, basis_all = [], [], []
    for ep in range(n_episodes):
        env = OdEnv(max_steps=episode_len)
        obs, info = env.reset(seed=seed + ep)
        obs_ep, act_ep, state_ep = [obs], [np.zeros(3, np.float32)], [info["state"]]
        time_ep = [info["geometry"]["time_s"]]
        station_state_ep = [info["geometry"]["station_state_eci"]]
        basis_ep = [info["geometry"]["topocentric_basis_eci"]]
        for _ in range(episode_len - 1):
            a = np.zeros(3, dtype=np.float32)
            obs, _, _, _, info = env.step(a)
            obs_ep.append(obs)
            act_ep.append(a)
            state_ep.append(info["state"])
            time_ep.append(info["geometry"]["time_s"])
            station_state_ep.append(info["geometry"]["station_state_eci"])
            basis_ep.append(info["geometry"]["topocentric_basis_eci"])
        obs_all.append(obs_ep)
        act_all.append(act_ep)
        state_all.append(state_ep)
        time_all.append(time_ep)
        station_state_all.append(station_state_ep)
        basis_all.append(basis_ep)

    payload = {
        "obs": np.asarray(obs_all, dtype=np.float32),
        "action": np.asarray(act_all, dtype=np.float32),
        "state": np.asarray(state_all, dtype=np.float32),
    }
    if save_geometry:
        payload.update(
            time_s=np.asarray(time_all, dtype=np.float32).squeeze(-1),
            station_state_eci=np.asarray(station_state_all, dtype=np.float32),
            topocentric_basis_eci=np.asarray(basis_all, dtype=np.float32),
        )
    np.savez(out_path, **payload)
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=64)
    p.add_argument("--episode-len", type=int, default=256)
    p.add_argument("--out", default="data/cache/od_trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-geometry", action="store_true")
    p.add_argument("--source", default="", help="Existing OD .npz to copy and augment with geometry.")
    args = p.parse_args()
    if args.source:
        print("wrote", augment_geometry(args.source, args.out))
    else:
        print(
            "wrote",
            generate(
                args.n_episodes,
                args.episode_len,
                args.out,
                args.seed,
                save_geometry=args.save_geometry,
            ),
        )
