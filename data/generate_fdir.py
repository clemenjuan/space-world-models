"""Roll out FdirEnv NOMINAL episodes (no fault, action 0) and save to .npz."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.fdir_env import FdirEnv


def generate(n_episodes=64, episode_len=256, out_path="data/cache/fdir_trajectories.npz", seed=0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    obs_all, act_all, state_all = [], [], []
    # Nominal discrete action is 0; one-hot it so the model receives a (T, 4)
    # float action and the existing OdWindowDataset (which reads .shape[-1])
    # works unchanged. The one-hot is identical every step in nominal rollouts.
    nominal_onehot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for ep in range(n_episodes):
        env = FdirEnv(fault_mode=None, max_steps=episode_len)
        obs, info = env.reset(seed=seed + ep)
        obs_ep, act_ep, state_ep = [obs], [nominal_onehot.copy()], [info["state"]]
        for _ in range(episode_len - 1):
            # Discrete nominal action 0 (does not affect dynamics this step).
            obs, _, _, _, info = env.step(0)
            obs_ep.append(obs)
            act_ep.append(nominal_onehot.copy())
            state_ep.append(info["state"])
        obs_all.append(obs_ep)
        act_all.append(act_ep)
        state_all.append(state_ep)
    np.savez(
        out_path,
        obs=np.asarray(obs_all, dtype=np.float32),
        action=np.asarray(act_all, dtype=np.float32),
        state=np.asarray(state_all, dtype=np.float32),
    )
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=64)
    p.add_argument("--episode-len", type=int, default=256)
    p.add_argument("--out", default="data/cache/fdir_trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print("wrote", generate(args.n_episodes, args.episode_len, args.out, args.seed))
