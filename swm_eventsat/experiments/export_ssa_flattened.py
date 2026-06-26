"""Flatten an SSA constellation dataset into a single-sat-shaped LeWM npz.

The IMAS shared world-model baseline treats each satellite trajectory as an
independent transition stream. This stages the ``ssa_world_model_v1`` export
(``(E,T,S,·)``) into the ``(E*S,T,·)`` layout the existing
``WindowedTrajectoryDataset`` / ``train_world_model`` pipeline consumes, with an
8D SSA action instead of the 7D EventSat action.

Example:
    python -m swm_eventsat.experiments.export_ssa_flattened \
        --dataset ~/autops-agentic-framework/data/world_model/ssa_v1/ssa_world_model_v1.npz \
        --out data/cache/ssa_imas_flat.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from swm_eventsat.schema import load_ssa_world_model_dataset


def run(dataset: str | Path, out: str | Path) -> Path:
    ds = load_ssa_world_model_dataset(dataset)
    flat = ds.flatten_satellites()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        obs=flat.obs.astype(np.float32),
        action=flat.action.astype(np.float32),
        state=flat.state.astype(np.float32),
        reward=flat.reward.astype(np.float32),
        mode=flat.mode.astype(np.int64),
        resolved_mode=flat.resolved_mode.astype(np.int64),
        forced_mode=flat.forced_mode.astype(np.float32),
        episode_seed=flat.episode_seed.astype(np.int64),
    )
    print(
        f"wrote {out_path} obs={flat.obs.shape} action={flat.action.shape} "
        f"(from {ds.n_episodes} episodes x {ds.n_satellites} satellites)"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="ssa_world_model_v1.npz path")
    parser.add_argument("--out", default="data/cache/ssa_imas_flat.npz")
    args = parser.parse_args()
    run(args.dataset, args.out)


if __name__ == "__main__":
    main()
