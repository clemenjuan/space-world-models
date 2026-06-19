"""Rank EventSat-Lite action traces with the world model and delta decoder."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from envs.eventsat_env import STATE_NAMES
from envs.eventsat_lite_env import (
    LITE_ACTION_DIM,
    LITE_MODE_LIST,
    LITE_MODE_TO_INDEX,
    EventSatLiteEnv,
    heuristic_eventsat_lite_policy,
)
from eventsat_world_model_utils import (
    FIGURES,
    STATE_INDEX,
    decode_latents,
    fit_normalizers,
    load_eventsat_model,
    load_state_decoder,
    normalize_action,
    normalize_obs,
    relpath,
    safety_flags,
    state_summary,
    write_json,
)


DATASET = ROOT / "data/cache/eventsat_lite_trajectories.npz"
DECODER = FIGURES / "eventsat_lite_delta_decoder.pt"
OUT = FIGURES / "eventsat_lite_trace_ranking.json"
MODEL_DATASET = "data/cache/eventsat_lite_trajectories.npz"


def _one_hot(index: int) -> np.ndarray:
    out = np.zeros(LITE_ACTION_DIM, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _one_hot_sequence(actions: np.ndarray) -> np.ndarray:
    return np.stack([_one_hot(int(a)) for a in actions], axis=0)


def _mode_hist(actions: np.ndarray) -> dict[str, list[Any]]:
    counts = np.bincount(actions.astype(int), minlength=LITE_ACTION_DIM)
    return {"mode": list(LITE_MODE_LIST), "count": counts.astype(int).tolist()}


def _macro_to_base_idx(action: int, env: EventSatLiteEnv) -> int:
    from envs.eventsat_env import MODE_TO_INDEX

    base_mode = env._macro_to_base_mode(LITE_MODE_LIST[int(action)])
    return MODE_TO_INDEX[base_mode]


def rollout_lite_action_sequence(actions: np.ndarray, seed: int) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    env = EventSatLiteEnv(max_steps=actions.shape[0])
    obs, info = env.reset(seed=seed)
    obs_rows, action_rows, state_rows, reward_rows = [], [], [], []
    mode_rows, base_mode_rows, forced_rows = [], [], []

    for t, action in enumerate(actions):
        obs_rows.append(obs)
        action_rows.append(_one_hot(int(action)))
        state_rows.append(info["state"])
        mode_rows.append(int(action))
        base_mode_rows.append(_macro_to_base_idx(int(action), env))
        forced_rows.append(0.0)
        if t < actions.shape[0] - 1:
            obs, reward, _, _, info = env.step(int(action))
            forced_rows[-1] = float(info["forced_lite_mode"])
            reward_rows.append(float(reward))
        else:
            reward_rows.append(0.0)

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "action": np.asarray(action_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "mode": np.asarray(mode_rows, dtype=np.int64),
        "base_mode": np.asarray(base_mode_rows, dtype=np.int64),
        "forced_mode": np.asarray(forced_rows, dtype=np.float32),
        "env_params": {"storage_capacity_mb": float(env.storage_capacity_mb)},
    }


def make_trace_actions(name: str, steps: int, seed: int) -> np.ndarray:
    if name == "charge_only":
        return np.full(steps, LITE_MODE_TO_INDEX["charge"], dtype=np.int64)
    if name == "observe_only":
        return np.full(steps, LITE_MODE_TO_INDEX["observe"], dtype=np.int64)
    if name == "process_only":
        return np.full(steps, LITE_MODE_TO_INDEX["process_to_obc"], dtype=np.int64)
    if name == "downlink_only":
        return np.full(steps, LITE_MODE_TO_INDEX["downlink"], dtype=np.int64)
    if name == "observe_process_no_downlink":
        pattern = np.asarray(
            [
                LITE_MODE_TO_INDEX["observe"],
                LITE_MODE_TO_INDEX["process_to_obc"],
                LITE_MODE_TO_INDEX["process_to_obc"],
                LITE_MODE_TO_INDEX["process_to_obc"],
                LITE_MODE_TO_INDEX["charge"],
            ],
            dtype=np.int64,
        )
        return np.resize(pattern, steps).astype(np.int64)
    if name == "heuristic":
        env = EventSatLiteEnv(max_steps=steps)
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        actions = []
        for t in range(steps):
            action = heuristic_eventsat_lite_policy(env, rng=rng, exploration=0.0)
            actions.append(action)
            if t < steps - 1:
                obs, _, _, _, info = env.step(action)
        return np.asarray(actions, dtype=np.int64)
    if name == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, LITE_ACTION_DIM, size=steps, dtype=np.int64)
    raise ValueError(f"unknown trace name {name!r}")


def _window_stack(array: np.ndarray, window: int) -> np.ndarray:
    return np.stack([array[start : start + window] for start in range(array.shape[0] - window + 1)]).astype(np.float32)


def _predict_deltas(
    model: Any,
    decoder: Any,
    artifact: dict[str, Any],
    rollout: dict[str, Any],
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    history_size: int,
    num_preds: int,
    window: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    obs_norm = normalize_obs(rollout["obs"], normalizers)
    act_norm = normalize_action(rollout["action"], normalizers)
    obs_windows = _window_stack(obs_norm, window)
    act_windows = _window_stack(act_norm, window)
    latents, step_rows = [], []
    with torch.no_grad():
        for start in range(0, obs_windows.shape[0], batch_size):
            end = min(start + batch_size, obs_windows.shape[0])
            batch = {
                "obs": torch.from_numpy(obs_windows[start:end]).to(device),
                "action": torch.from_numpy(act_windows[start:end]).to(device),
            }
            encoded = model.encode(batch)
            emb = encoded["emb"]
            act_emb = encoded["act_emb"]
            pred = model.predict(emb[:, :history_size], act_emb[:, :history_size])
            target = emb[:, num_preds:]
            m = min(pred.size(1), target.size(1))
            latents.append(pred[:, m - 1].cpu().numpy().astype(np.float32))
            step_rows.append(np.arange(start, end, dtype=np.int64) + num_preds + m - 1)
    decoded = decode_latents(
        decoder,
        artifact,
        np.concatenate(latents, axis=0),
        batch_size=batch_size,
        device=device,
    )
    return np.concatenate(step_rows, axis=0), decoded


def _true_score(summary: dict[str, float], flags: dict[str, Any]) -> float:
    return float(
        3.0 * summary["final_downlinked_mb"]
        + 0.5 * summary["observation_min"]
        + 2.0 * summary["detections"]
        - 0.02 * summary["final_stored_mb"]
        - 25.0 * flags["low_soc_steps"]
        - 5.0 * flags["invalid_communication_steps"]
        - 0.5 * flags["storage_pressure_steps"]
    )


def _predicted_score(
    decoded: np.ndarray,
    artifact: dict[str, Any],
    rollout: dict[str, Any],
    steps: np.ndarray,
) -> dict[str, float]:
    names = artifact["target_names"]
    idx = {name: i for i, name in enumerate(names)}
    target_actions = rollout["mode"][steps]
    target_state = rollout["state"][steps]
    downlink = float(np.clip(decoded[:, idx["delta_data_downlinked_mb"]], 0.0, None).sum())
    observations = float(
        (
            np.clip(decoded[:, idx["event_observation"]], 0.0, 1.0)
            * (target_actions == LITE_MODE_TO_INDEX["observe"])
        ).sum()
    )
    detections = float(
        (
            np.clip(decoded[:, idx["event_detection"]], 0.0, 1.0)
            * (target_actions == LITE_MODE_TO_INDEX["process_to_obc"])
        ).sum()
    )
    stored0 = float(
        rollout["state"][0, STATE_INDEX["obc_data_mb"]]
        + rollout["state"][0, STATE_INDEX["jetson_raw_mb"]]
        + rollout["state"][0, STATE_INDEX["jetson_compressed_mb"]]
    )
    stored_delta = (
        decoded[:, idx["delta_obc_data_mb"]]
        + decoded[:, idx["delta_jetson_raw_mb"]]
        + decoded[:, idx["delta_jetson_compressed_mb"]]
    )
    pred_stored = stored0 + np.cumsum(stored_delta)
    pred_soc = float(rollout["state"][0, STATE_INDEX["battery_soc"]]) + np.cumsum(decoded[:, idx["delta_battery_soc"]])
    capacity = float(rollout["env_params"]["storage_capacity_mb"])
    storage_pressure = float(np.sum(pred_stored > 0.85 * capacity))
    low_soc = float(np.sum(pred_soc < 0.25))
    invalid_downlink = float(
        np.sum(
            (target_actions == LITE_MODE_TO_INDEX["downlink"])
            & (target_state[:, STATE_INDEX["ground_pass_active"]] < 0.5)
        )
    )
    final_stored = float(max(0.0, pred_stored[-1])) if pred_stored.size else stored0
    score = (
        3.0 * downlink
        + 0.5 * observations
        + 2.0 * detections
        - 0.02 * final_stored
        - 0.5 * storage_pressure
        - 25.0 * low_soc
        - 5.0 * invalid_downlink
    )
    return {
        "score": float(score),
        "predicted_downlinked_mb": downlink,
        "predicted_observation_events": observations,
        "predicted_detection_events": detections,
        "predicted_final_stored_mb": final_stored,
        "predicted_storage_pressure_steps": storage_pressure,
        "predicted_low_soc_steps": low_soc,
        "predicted_invalid_downlink_steps": invalid_downlink,
    }


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ra = _rank(a)
    rb = _rank(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.backends.nnpack.enabled = False
    device = torch.device(args.device)
    model, cfg, run = load_eventsat_model(device=device, dataset=MODEL_DATASET)
    decoder, artifact = load_state_decoder(Path(args.decoder), device=device)
    dataset_path = Path(cfg.data.path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    normalizers = fit_normalizers(dataset_path)
    trace_names = args.traces.split(",") if args.traces else [
        "heuristic",
        "observe_process_no_downlink",
        "observe_only",
        "downlink_only",
        "process_only",
        "charge_only",
        "random",
    ]

    rows = []
    for i, trace_name in enumerate(trace_names):
        actions = make_trace_actions(trace_name, args.steps, args.seed + i * 101)
        rollout = rollout_lite_action_sequence(actions, seed=args.seed)
        steps, decoded = _predict_deltas(
            model,
            decoder,
            artifact,
            rollout,
            normalizers,
            history_size=int(cfg.history_size),
            num_preds=int(cfg.num_preds),
            window=int(cfg.data.window),
            batch_size=args.batch_size,
            device=device,
        )
        summary = state_summary(rollout["state"], rollout["forced_mode"])
        flags = safety_flags(
            state=rollout["state"],
            mode=rollout["base_mode"],
            forced_mode=rollout["forced_mode"],
            storage_capacity_mb=rollout["env_params"]["storage_capacity_mb"],
        )
        predicted = _predicted_score(decoded, artifact, rollout, steps)
        row = {
            "trace": trace_name,
            "hist": _mode_hist(actions),
            "true_summary": summary,
            "true_safety_flags": flags,
            "true_score": _true_score(summary, flags),
            "predicted_score": predicted["score"],
            **predicted,
        }
        rows.append(row)

    true_scores = [row["true_score"] for row in rows]
    predicted_scores = [row["predicted_score"] for row in rows]
    rows_by_pred = sorted(rows, key=lambda row: row["predicted_score"], reverse=True)
    rows_by_true = sorted(rows, key=lambda row: row["true_score"], reverse=True)
    payload = {
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "mode_names": list(LITE_MODE_LIST),
        "run": run,
        "decoder": {
            "artifact": relpath(Path(args.decoder)),
            "created_at": artifact.get("created_at"),
            "target_names": artifact.get("target_names", []),
        },
        "metric": {
            "spearman_predicted_vs_true": _spearman(predicted_scores, true_scores),
            "top_predicted_trace": rows_by_pred[0]["trace"],
            "top_true_trace": rows_by_true[0]["trace"],
            "top_match": rows_by_pred[0]["trace"] == rows_by_true[0]["trace"],
        },
        "traces": rows,
        "ranking": {
            "predicted": [row["trace"] for row in rows_by_pred],
            "true": [row["trace"] for row in rows_by_true],
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder", default=str(DECODER))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traces", default="")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    payload = evaluate(args)
    out_path = Path(args.out)
    write_json(out_path, payload)
    print(
        f"wrote {relpath(out_path)} spearman={payload['metric']['spearman_predicted_vs_true']:.3f} "
        f"top_pred={payload['metric']['top_predicted_trace']} top_true={payload['metric']['top_true_trace']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
