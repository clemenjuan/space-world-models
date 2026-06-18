"""Evaluate FDIR detector behavior across fault modes and seeds.

This produces a compact JSON artifact consumed by scripts/build_fdir_results_board.py.
The experiment calibrates thresholds on nominal episodes, then evaluates detection
rate, pre-fault false alarms, and detection delay over several state-level fault
scenarios.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/cache/fdir_trajectories.npz"
OUT = ROOT / "data/figures/fdir_detection_benchmark.json"


SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "nominal_holdout",
        "label": "Nominal holdout",
        "fault_mode": None,
        "fault_channel": None,
        "params": {},
        "expected": "false-alarm control",
    },
    {
        "id": "spike_voltage_obvious",
        "label": "Voltage spike, obvious",
        "fault_mode": "spike",
        "fault_channel": "solar_array_voltage",
        "params": {"spike_magnitude": 5.0, "spike_duration": 20},
        "expected": "easy impulse",
    },
    {
        "id": "spike_voltage_mild",
        "label": "Voltage spike, mild",
        "fault_mode": "spike",
        "fault_channel": "solar_array_voltage",
        "params": {"spike_magnitude": 0.5, "spike_duration": 20},
        "expected": "small impulse",
    },
    {
        "id": "drift_voltage_slow",
        "label": "Voltage drift, slow",
        "fault_mode": "drift",
        "fault_channel": "solar_array_voltage",
        "params": {"drift_rate": 0.015},
        "expected": "gradual power drift",
    },
    {
        "id": "drift_voltage_fast",
        "label": "Voltage drift, fast",
        "fault_mode": "drift",
        "fault_channel": "solar_array_voltage",
        "params": {"drift_rate": 0.05},
        "expected": "clear power drift",
    },
    {
        "id": "stuck_voltage",
        "label": "Voltage stuck-at",
        "fault_mode": "stuck_at",
        "fault_channel": "solar_array_voltage",
        "params": {},
        "expected": "loss of power dynamics",
    },
    {
        "id": "drift_wheel_x",
        "label": "Wheel-X speed drift",
        "fault_mode": "drift",
        "fault_channel": "rw_speed_x",
        "params": {"drift_rate": 2.5},
        "expected": "actuator momentum drift",
    },
    {
        "id": "stuck_wheel_x",
        "label": "Wheel-X stuck-at",
        "fault_mode": "stuck_at",
        "fault_channel": "rw_speed_x",
        "params": {},
        "expected": "loss of wheel dynamics",
    },
    {
        "id": "drift_panel_temp",
        "label": "Panel temperature drift",
        "fault_mode": "drift",
        "fault_channel": "panel_temp",
        "params": {"drift_rate": 0.035},
        "expected": "thermal ramp",
    },
]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    x = float(value)
    return x if math.isfinite(x) else None


def _percentile(values: np.ndarray, q: float) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    if flat.size == 0:
        return float("nan")
    return float(np.percentile(flat, q))


def _threshold(values: np.ndarray, percentile: float) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    p = _percentile(flat, percentile)
    return {
        "mean": mean,
        "std": std,
        "mean_plus_3std": mean + 3.0 * std,
        "percentile": percentile,
        "percentile_value": p,
        "threshold": max(mean + 3.0 * std, p),
    }


def _alarm_edges(scores: np.ndarray, threshold: float, min_consecutive: int) -> list[int]:
    above = np.asarray(scores, dtype=float).ravel() > threshold
    edges: list[int] = []
    run = 0
    fired = False
    for i, hot in enumerate(above):
        if hot:
            run += 1
            if run >= min_consecutive and not fired:
                edges.append(i)
                fired = True
        else:
            run = 0
            fired = False
    return edges


def _first_delay(
    scores: np.ndarray,
    score_steps: np.ndarray,
    threshold: float,
    fault_step: int,
    min_consecutive: int,
) -> int | None:
    for idx in _alarm_edges(scores, threshold, min_consecutive):
        step = int(score_steps[idx])
        if step >= fault_step:
            return step - fault_step
    return None


def _pre_alarm_count(
    scores: np.ndarray,
    score_steps: np.ndarray,
    threshold: float,
    fault_step: int,
    min_consecutive: int,
) -> int:
    return sum(
        1
        for idx in _alarm_edges(scores, threshold, min_consecutive)
        if int(score_steps[idx]) < fault_step
    )


def _normalize(obs: np.ndarray, actions: np.ndarray, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    obs_norm = (obs - stats["obs_mean"]) / stats["obs_std"]
    act_norm = (actions - stats["act_mean"]) / stats["act_std"]
    return obs_norm.astype(np.float32), act_norm.astype(np.float32)


def _one_hot_nominal(n: int, action_dim: int) -> np.ndarray:
    actions = np.zeros((n, action_dim), dtype=np.float32)
    actions[:, 0] = 1.0
    return actions


def _rollout_scenario(
    scenario: dict[str, Any],
    *,
    seed: int,
    episode_len: int,
    fault_step: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from envs.fdir_env import FdirEnv

    kwargs = {
        "max_steps": episode_len,
        "fault_mode": scenario["fault_mode"],
        "fault_channel": scenario["fault_channel"],
        "fault_step": fault_step,
    }
    kwargs.update(scenario.get("params") or {})
    env = FdirEnv(**kwargs)
    obs, info = env.reset(seed=seed)
    obs_seq = [obs]
    state_seq = [info["state"]]
    for _ in range(episode_len - 1):
        obs, _, _, _, info = env.step(0)
        obs_seq.append(obs)
        state_seq.append(info["state"])
    actions = _one_hot_nominal(episode_len, action_dim)
    return np.asarray(obs_seq, dtype=np.float32), actions, np.asarray(state_seq, dtype=np.float32)


def _load_model() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import hydra
    from omegaconf import OmegaConf

    from scripts.build_fdir_results_board import _find_runs, _latest_checkpoint_run, _strip_model_prefix

    runs = _find_runs()
    run = _latest_checkpoint_run(runs)
    if run is None:
        raise FileNotFoundError("No readable FDIR checkpoint found under stable-pretraining runs.")
    run_dir = Path(run["run_dir"])
    cfg = OmegaConf.load(run_dir / "hparams.yaml")
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = _strip_model_prefix(checkpoint.get("state_dict", checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model_meta = {
        "run_id": run["run_id"],
        "run_name": run["name"],
        "checkpoint": run["checkpoint"],
        "history_size": int(cfg.history_size),
        "num_preds": int(cfg.num_preds),
        "window": int(cfg.data.window),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return model, model_meta, run


def _score_lewm(model: Any, obs_norm: np.ndarray, act_norm: np.ndarray, history_size: int) -> np.ndarray:
    from models.surprise import surprise_scores

    with torch.no_grad():
        obs_t = torch.tensor(obs_norm, dtype=torch.float32)
        act_t = torch.tensor(act_norm, dtype=torch.float32)
        return surprise_scores(model, obs_t, act_t, history_size=history_size).cpu().numpy()


def _z_energy(obs_norm: np.ndarray) -> np.ndarray:
    return np.mean(np.square(obs_norm), axis=-1)


def _episode_row(
    *,
    scenario: dict[str, Any],
    seed: int,
    lewm_scores: np.ndarray,
    z_scores: np.ndarray,
    lewm_steps: np.ndarray,
    z_steps: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    fault_step: int,
    min_consecutive: int,
    detection_window: int,
) -> dict[str, Any]:
    lewm_delay = _first_delay(
        lewm_scores, lewm_steps, thresholds["lewm"]["threshold"], fault_step, min_consecutive
    )
    z_delay = _first_delay(
        z_scores, z_steps, thresholds["zscore"]["threshold"], fault_step, min_consecutive
    )
    lewm_post_mask = (lewm_steps >= fault_step) & (lewm_steps < fault_step + detection_window)
    lewm_pre_mask = lewm_steps < fault_step
    z_post_mask = (z_steps >= fault_step) & (z_steps < fault_step + detection_window)
    z_pre_mask = z_steps < fault_step
    lewm_pre = lewm_scores[lewm_pre_mask]
    lewm_post = lewm_scores[lewm_post_mask]
    z_pre = z_scores[z_pre_mask]
    z_post = z_scores[z_post_mask]
    return {
        "scenario_id": scenario["id"],
        "scenario_label": scenario["label"],
        "seed": int(seed),
        "fault_step": int(fault_step),
        "lewm_detected": bool(lewm_delay is not None),
        "lewm_detected_within_window": bool(lewm_delay is not None and lewm_delay < detection_window),
        "lewm_delay": lewm_delay,
        "lewm_pre_alarm_count": _pre_alarm_count(
            lewm_scores, lewm_steps, thresholds["lewm"]["threshold"], fault_step, min_consecutive
        ),
        "lewm_pre_mean": float(lewm_pre.mean()) if lewm_pre.size else None,
        "lewm_post_mean": float(lewm_post.mean()) if lewm_post.size else None,
        "lewm_max_post": float(lewm_post.max()) if lewm_post.size else None,
        "zscore_detected": bool(z_delay is not None),
        "zscore_detected_within_window": bool(z_delay is not None and z_delay < detection_window),
        "zscore_delay": z_delay,
        "zscore_pre_alarm_count": _pre_alarm_count(
            z_scores, z_steps, thresholds["zscore"]["threshold"], fault_step, min_consecutive
        ),
        "zscore_pre_mean": float(z_pre.mean()) if z_pre.size else None,
        "zscore_post_mean": float(z_post.mean()) if z_post.size else None,
        "zscore_max_post": float(z_post.max()) if z_post.size else None,
    }


def _summarize_rows(scenario: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0

    def alarm_rate(key: str) -> float:
        return float(np.mean([row[key] > 0 for row in rows])) if rows else 0.0

    def median_delay(key: str) -> float | None:
        delays = [_finite_float(row[key]) for row in rows]
        delays = [d for d in delays if d is not None]
        return float(np.median(delays)) if delays else None

    def mean_value(key: str) -> float | None:
        values = [_finite_float(row[key]) for row in rows]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    lewm_pre = mean_value("lewm_pre_mean")
    lewm_post = mean_value("lewm_post_mean")
    z_pre = mean_value("zscore_pre_mean")
    z_post = mean_value("zscore_post_mean")
    return {
        "id": scenario["id"],
        "label": scenario["label"],
        "fault_mode": scenario["fault_mode"] or "none",
        "fault_channel": scenario["fault_channel"] or "none",
        "params": scenario.get("params") or {},
        "expected": scenario.get("expected", ""),
        "episodes": n,
        "lewm_detect_rate": rate("lewm_detected_within_window"),
        "lewm_any_detect_rate": rate("lewm_detected"),
        "lewm_false_pre_rate": alarm_rate("lewm_pre_alarm_count"),
        "lewm_median_delay": median_delay("lewm_delay"),
        "lewm_pre_mean": lewm_pre,
        "lewm_post_mean": lewm_post,
        "lewm_post_pre_ratio": (lewm_post / lewm_pre) if lewm_pre and lewm_post is not None else None,
        "zscore_detect_rate": rate("zscore_detected_within_window"),
        "zscore_any_detect_rate": rate("zscore_detected"),
        "zscore_false_pre_rate": alarm_rate("zscore_pre_alarm_count"),
        "zscore_median_delay": median_delay("zscore_delay"),
        "zscore_pre_mean": z_pre,
        "zscore_post_mean": z_post,
        "zscore_post_pre_ratio": (z_post / z_pre) if z_pre and z_post is not None else None,
    }


def evaluate(
    *,
    dataset_path: Path,
    output_path: Path,
    seeds: int,
    nominal_episodes: int,
    fault_step: int,
    threshold_percentile: float,
    min_consecutive: int,
    detection_window: int,
) -> dict[str, Any]:
    model, model_meta, _run = _load_model()
    blob = np.load(dataset_path)
    obs = blob["obs"].astype(np.float32)
    actions = blob["action"].astype(np.float32)
    state = blob["state"].astype(np.float32)
    episode_len = int(obs.shape[1])
    fault_step = min(int(fault_step), episode_len - 2)

    stats = {
        "obs_mean": obs.reshape(-1, obs.shape[-1]).mean(axis=0),
        "obs_std": obs.reshape(-1, obs.shape[-1]).std(axis=0),
        "act_mean": actions.reshape(-1, actions.shape[-1]).mean(axis=0),
        "act_std": actions.reshape(-1, actions.shape[-1]).std(axis=0),
    }
    stats["obs_std"][stats["obs_std"] < 1e-8] = 1.0
    stats["act_std"][stats["act_std"] < 1e-8] = 1.0

    n_cal = min(int(nominal_episodes), obs.shape[0])
    nominal_obs_norm, nominal_act_norm = _normalize(obs[:n_cal], actions[:n_cal], stats)
    lewm_nominal = _score_lewm(model, nominal_obs_norm, nominal_act_norm, model_meta["history_size"])
    z_nominal = _z_energy(nominal_obs_norm)
    thresholds = {
        "lewm": _threshold(lewm_nominal, threshold_percentile),
        "zscore": _threshold(z_nominal, threshold_percentile),
    }

    lewm_steps = np.arange(model_meta["history_size"], episode_len)
    z_steps = np.arange(episode_len)
    scenario_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None

    for scenario in SCENARIOS:
        scenario_obs = []
        scenario_actions = []
        scenario_states = []
        for i in range(seeds):
            seed = 9000 + i
            obs_seq, act_seq, state_seq = _rollout_scenario(
                scenario,
                seed=seed,
                episode_len=episode_len,
                fault_step=fault_step,
                action_dim=actions.shape[-1],
            )
            scenario_obs.append(obs_seq)
            scenario_actions.append(act_seq)
            scenario_states.append(state_seq)

        scenario_obs_arr = np.asarray(scenario_obs, dtype=np.float32)
        scenario_act_arr = np.asarray(scenario_actions, dtype=np.float32)
        scenario_obs_norm, scenario_act_norm = _normalize(scenario_obs_arr, scenario_act_arr, stats)
        scenario_lewm = _score_lewm(model, scenario_obs_norm, scenario_act_norm, model_meta["history_size"])
        scenario_z = _z_energy(scenario_obs_norm)
        scenario_rows: list[dict[str, Any]] = []
        for i in range(seeds):
            row = _episode_row(
                scenario=scenario,
                seed=9000 + i,
                lewm_scores=scenario_lewm[i],
                z_scores=scenario_z[i],
                lewm_steps=lewm_steps,
                z_steps=z_steps,
                thresholds=thresholds,
                fault_step=fault_step,
                min_consecutive=min_consecutive,
                detection_window=detection_window,
            )
            scenario_rows.append(row)
            rows.append(row)

        summary = _summarize_rows(scenario, scenario_rows)
        scenario_summaries.append(summary)

        if scenario["id"] == "spike_voltage_mild":
            rep_idx = 0
            representative = {
                "scenario_id": scenario["id"],
                "scenario_label": scenario["label"],
                "seed": 9000,
                "fault_step": fault_step,
                "time_step": z_steps.astype(int).tolist(),
                "score_time_step": lewm_steps.astype(int).tolist(),
                "lewm": scenario_lewm[rep_idx].astype(float).tolist(),
                "zscore": scenario_z[rep_idx].astype(float).tolist(),
                "obs": {
                    "solar_array_voltage": scenario_obs_arr[rep_idx, :, 0].astype(float).tolist(),
                    "panel_temp": scenario_obs_arr[rep_idx, :, 2].astype(float).tolist(),
                    "rw_speed_x": scenario_obs_arr[rep_idx, :, 4].astype(float).tolist(),
                    "bus_current": scenario_obs_arr[rep_idx, :, 7].astype(float).tolist(),
                },
            }

    lewm_rates = [s["lewm_detect_rate"] for s in scenario_summaries if s["id"] != "nominal_holdout"]
    z_rates = [s["zscore_detect_rate"] for s in scenario_summaries if s["id"] != "nominal_holdout"]
    lewm_delays = [
        s["lewm_median_delay"]
        for s in scenario_summaries
        if s["id"] != "nominal_holdout" and s["lewm_median_delay"] is not None
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path.relative_to(ROOT)),
        "model": model_meta,
        "config": {
            "seeds": seeds,
            "nominal_calibration_episodes": n_cal,
            "fault_step": fault_step,
            "threshold_percentile": threshold_percentile,
            "threshold_rule": "max(mean + 3 std, percentile)",
            "min_consecutive": min_consecutive,
            "detection_window": detection_window,
        },
        "thresholds": thresholds,
        "nominal": {
            "episodes": int(obs.shape[0]),
            "steps": episode_len,
            "obs_dim": int(obs.shape[-1]),
            "action_dim": int(actions.shape[-1]),
            "state_dim": int(state.shape[-1]),
            "lewm_threshold_exceed_fraction": float(
                np.mean(lewm_nominal > thresholds["lewm"]["threshold"])
            ),
            "zscore_threshold_exceed_fraction": float(
                np.mean(z_nominal > thresholds["zscore"]["threshold"])
            ),
        },
        "headline": {
            "scenario_count": len([s for s in scenario_summaries if s["id"] != "nominal_holdout"]),
            "episodes_per_scenario": seeds,
            "lewm_mean_detect_rate": float(np.mean(lewm_rates)) if lewm_rates else None,
            "zscore_mean_detect_rate": float(np.mean(z_rates)) if z_rates else None,
            "lewm_median_delay": float(np.median(lewm_delays)) if lewm_delays else None,
        },
        "scenario_summaries": scenario_summaries,
        "rows": rows,
        "representative": representative,
        "caveats": [
            "This benchmark measures unsupervised detection only; it does not classify or recover from faults.",
            "Thresholds are calibrated on nominal training episodes, so rates are diagnostic rather than a held-out certification claim.",
            "The z-score baseline is intentionally simple and strong for large marginal shifts; LeWM is expected to matter most for subtle temporal faults.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--nominal-episodes", type=int, default=48)
    parser.add_argument("--fault-step", type=int, default=100)
    parser.add_argument("--threshold-percentile", type=float, default=99.5)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--detection-window", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(
        dataset_path=args.dataset,
        output_path=args.out,
        seeds=args.seeds,
        nominal_episodes=args.nominal_episodes,
        fault_step=args.fault_step,
        threshold_percentile=args.threshold_percentile,
        min_consecutive=args.min_consecutive,
        detection_window=args.detection_window,
    )
    headline = payload["headline"]
    print(
        "wrote "
        f"{args.out.relative_to(ROOT)}; "
        f"{headline['scenario_count']} fault scenarios x {headline['episodes_per_scenario']} seeds, "
        f"LeWM mean detect={headline['lewm_mean_detect_rate']:.2f}, "
        f"z-score mean detect={headline['zscore_mean_detect_rate']:.2f}"
    )


if __name__ == "__main__":
    main()
