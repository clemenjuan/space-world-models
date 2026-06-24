"""AUTOPS bridge CLI for EventSat trajectory export and planner smoke runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from swm_eventsat.data.autops_adapter import (
    DEFAULT_AUTOPS_ROOT,
    export_autops_eventsat_npz,
    heuristic_policy,
    make_eventsat_env,
)
from swm_eventsat.schema import EVENTSAT_MODE_LIST
from swm_eventsat.experiments.write_autops_results import write_minimal_results


def _scaled_downlink_target_mb(steps: int, step_duration_s: float = 60.0) -> float:
    episode_days = (float(steps) * float(step_duration_s)) / 86400.0
    return max(1e-6, 221.0 * (episode_days / 90.0))


def run_policy_episode(
    *,
    steps: int,
    seed: int,
    autops_root: str | Path = DEFAULT_AUTOPS_ROOT,
) -> dict[str, float]:
    """Run one AUTOPS EventSat episode through the adapter policy hook."""
    env = make_eventsat_env(autops_root=autops_root, max_steps=steps)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    rewards: list[float] = []
    forced: list[float] = []
    modes: list[int] = []
    last_info: dict[str, Any] = {}

    for _ in range(int(steps)):
        action_idx = int(heuristic_policy(env, rng))
        modes.append(action_idx)
        result = env.step({"eventsat_0": {"mode": EVENTSAT_MODE_LIST[action_idx]}})
        rewards.append(float(sum(result.rewards.values())))
        last_info = dict(result.info)
        forced.append(float(last_info.get("forced", False)))
        if result.done:
            break

    downlinked = float(last_info.get("data_downlinked_mb", 0.0))
    max_downlink = float(last_info.get("max_achievable_downlink_mb", 0.0))
    obs_hours = float(last_info.get("observation_hours", 0.0))
    switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    n = max(1, len(rewards))
    forced_rate = float(np.mean(forced)) if forced else 0.0
    return {
        "utility": downlinked / _scaled_downlink_target_mb(steps),
        "data_downlink_efficiency": downlinked / max_downlink if max_downlink > 0 else 0.0,
        "observation_hours": obs_hours,
        "downlinked_mb": downlinked,
        "final_battery_soc": float(last_info.get("battery_soc", 0.0)),
        "operator_load": forced_rate,
        "constraint_violation_rate": forced_rate,
        "commanding_effort": switches / n,
        "mean_latency_s": 0.0,
        "candidate_count": 1.0,
        "cem_iterations": 0.0,
        "policy_loaded": 1.0,
        "total_reward": float(np.sum(rewards)),
    }


def run_policy_episodes(*, episodes: int, steps: int, seed: int, autops_root: str | Path) -> list[dict[str, float]]:
    return [
        run_policy_episode(steps=steps, seed=seed + ep, autops_root=autops_root)
        for ep in range(int(episodes))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-out", default="")
    parser.add_argument("--results-dir", default="data/results/eventsat_lewm_mpc_bridge")
    parser.add_argument("--experiment-id", default="eventsat_lewm_mpc_bridge")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autops-root", default=str(DEFAULT_AUTOPS_ROOT))
    args = parser.parse_args()
    if args.dataset_out:
        path = export_autops_eventsat_npz(
            args.dataset_out,
            n_episodes=args.episodes,
            episode_len=args.steps,
            seed=args.seed,
            autops_root=args.autops_root,
        )
        print(f"wrote dataset {path}")
    rows = run_policy_episodes(
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        autops_root=args.autops_root,
    )
    results = write_minimal_results(
        args.results_dir,
        experiment_id=args.experiment_id,
        episode_metrics=rows,
        config={
            "planner": "adapter_heuristic_smoke",
            "intended_planner": "LeWM+CEM",
            "steps": int(args.steps),
            "seed": int(args.seed),
        },
    )
    print(f"wrote results {results}")


if __name__ == "__main__":
    main()
