"""Evaluate a supplied EventSat action trace with the LeWM plus state decoder.

This script does not choose actions. It answers: if an external source supplies
these 7-mode commands, what does the world model predict, what happens in the
simplified simulator, and where are the safety flags?
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from swm_eventsat.data.toy_eventsat_env import ACTION_DIM, MODE_LIST, MODE_TO_INDEX, STATE_NAMES
from swm_eventsat.models.checkpoint_io import (
    DATASET,
    FIGURES,
    STATE_INDEX,
    decode_latents,
    fit_normalizers,
    load_eventsat_model,
    load_state_decoder,
    mode_histogram,
    normalize_action,
    normalize_obs,
    one_hot_sequence,
    relpath,
    rollout_action_sequence,
    rollout_heuristic,
    safety_flags,
    state_summary,
    write_json,
)


OUT = FIGURES / "eventsat_action_trace_evaluation.json"
DECODER = FIGURES / "eventsat_state_decoder.pt"
TARGET_NAMES = tuple(STATE_NAMES) + ("reward", "forced_mode")


def _mode_to_idx(value: Any) -> int:
    if isinstance(value, str):
        if value not in MODE_TO_INDEX:
            raise ValueError(f"unknown EventSat mode {value!r}")
        return MODE_TO_INDEX[value]
    idx = int(value)
    if idx < 0 or idx >= ACTION_DIM:
        raise ValueError(f"mode index out of range: {idx}")
    return idx


def load_actions(path: Path) -> tuple[np.ndarray, str]:
    if path.suffix.lower() == ".npz":
        blob = np.load(path)
        if "mode" in blob:
            arr = blob["mode"]
        elif "actions" in blob:
            arr = blob["actions"]
        elif "action" in blob:
            arr = blob["action"]
        else:
            raise ValueError(f"{path} must contain mode, actions, or action")
        if arr.ndim >= 2 and arr.shape[-1] == ACTION_DIM:
            arr = np.argmax(arr, axis=-1)
        return np.asarray(arr, dtype=np.int64).reshape(-1), f"file:{relpath(path)}"

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("actions", "action", "mode", "modes", "action_sequence"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"{path} JSON must contain actions, action, mode, modes, or action_sequence")
    if not isinstance(payload, list):
        raise ValueError(f"{path} JSON action payload must be a list")
    return np.asarray([_mode_to_idx(item) for item in payload], dtype=np.int64), f"file:{relpath(path)}"


def _resize_actions(actions: np.ndarray, steps: int | None) -> np.ndarray:
    if steps is None or steps <= 0:
        return actions.astype(np.int64)
    steps = int(steps)
    if actions.shape[0] >= steps:
        return actions[:steps].astype(np.int64)
    if actions.shape[0] == 0:
        raise ValueError("action trace is empty")
    pad = np.full(steps - actions.shape[0], int(actions[-1]), dtype=np.int64)
    return np.concatenate([actions.astype(np.int64), pad], axis=0)


def _policy_actions(policy: str, steps: int, seed: int, exploration: float) -> tuple[np.ndarray, str]:
    if policy == "heuristic":
        rollout = rollout_heuristic(steps=steps, seed=seed, exploration=exploration)
        return np.asarray(rollout["mode"], dtype=np.int64), "policy:scripted_heuristic"
    if policy == "always_communicate":
        return np.full(steps, MODE_TO_INDEX["communication"], dtype=np.int64), "policy:always_communicate_bad_trace"
    if policy == "always_observe":
        return np.full(steps, MODE_TO_INDEX["payload_observe"], dtype=np.int64), "policy:always_observe_bad_trace"
    if policy == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, ACTION_DIM, size=steps, dtype=np.int64), "policy:seeded_random"
    raise ValueError(f"unknown policy {policy!r}")


def _window_stack(array: np.ndarray, window: int) -> np.ndarray:
    return np.stack([array[start : start + window] for start in range(array.shape[0] - window + 1)]).astype(np.float32)


def _regression_metrics(pred: np.ndarray, target: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    residual = pred - target
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    return {
        "rmse": {name: float(rmse[i]) for i, name in enumerate(names)},
        "mae": {name: float(mae[i]) for i, name in enumerate(names)},
    }


def _summary_delta(predicted: dict[str, float], true: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(predicted) & set(true))
    return {key: float(predicted[key] - true[key]) for key in keys if isinstance(predicted[key], (int, float))}


def _predict_teacher_forced(
    model: Any,
    decoder: Any,
    decoder_artifact: dict[str, Any],
    rollout: dict[str, Any],
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    history_size: int,
    num_preds: int,
    window: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    obs_norm = normalize_obs(np.asarray(rollout["obs"], dtype=np.float32), normalizers)
    act_norm = normalize_action(np.asarray(rollout["action"], dtype=np.float32), normalizers)
    obs_windows = _window_stack(obs_norm, window)
    act_windows = _window_stack(act_norm, window)

    pred_latents: list[np.ndarray] = []
    target_steps: list[np.ndarray] = []
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
            pred_latents.append(pred[:, m - 1].cpu().numpy().astype(np.float32))
            offsets = np.arange(start, end, dtype=np.int64) + num_preds + m - 1
            target_steps.append(offsets)

    pred_latent = np.concatenate(pred_latents, axis=0)
    steps = np.concatenate(target_steps, axis=0)
    decoded = decode_latents(decoder, decoder_artifact, pred_latent, batch_size=batch_size, device=device)
    true_y = np.concatenate(
        [
            np.asarray(rollout["state"], dtype=np.float32)[steps],
            np.asarray(rollout["reward"], dtype=np.float32)[steps, None],
            np.asarray(rollout["forced_mode"], dtype=np.float32)[steps, None],
        ],
        axis=1,
    )
    return {
        "steps": steps,
        "decoded": decoded,
        "true_y": true_y,
        "metrics": _regression_metrics(decoded, true_y, TARGET_NAMES),
    }


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


def _compact_timeline(steps: np.ndarray, decoded: np.ndarray, rollout: dict[str, Any], max_points: int) -> dict[str, Any]:
    idx = _downsample_indices(steps.shape[0], max_points)
    true_state = np.asarray(rollout["state"], dtype=np.float32)[steps[idx]]
    pred_state = decoded[idx, : len(STATE_NAMES)]
    return {
        "step": steps[idx].astype(int).tolist(),
        "true_soc": true_state[:, STATE_INDEX["battery_soc"]].astype(float).tolist(),
        "pred_soc": pred_state[:, STATE_INDEX["battery_soc"]].astype(float).tolist(),
        "true_downlinked_mb": true_state[:, STATE_INDEX["data_downlinked_mb"]].astype(float).tolist(),
        "pred_downlinked_mb": pred_state[:, STATE_INDEX["data_downlinked_mb"]].astype(float).tolist(),
        "true_stored_mb": (
            true_state[:, STATE_INDEX["obc_data_mb"]]
            + true_state[:, STATE_INDEX["jetson_raw_mb"]]
            + true_state[:, STATE_INDEX["jetson_compressed_mb"]]
        )
        .astype(float)
        .tolist(),
        "pred_stored_mb": (
            pred_state[:, STATE_INDEX["obc_data_mb"]]
            + pred_state[:, STATE_INDEX["jetson_raw_mb"]]
            + pred_state[:, STATE_INDEX["jetson_compressed_mb"]]
        )
        .astype(float)
        .tolist(),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.backends.nnpack.enabled = False
    device = torch.device(args.device)
    if args.actions:
        actions, source = load_actions(Path(args.actions))
        actions = _resize_actions(actions, args.steps)
    else:
        steps = int(args.steps or 10080)
        actions, source = _policy_actions(args.policy, steps=steps, seed=args.seed, exploration=args.exploration)

    rollout = rollout_action_sequence(actions, seed=args.seed)
    model, cfg, run = load_eventsat_model(device=device)
    decoder, decoder_artifact = load_state_decoder(Path(args.decoder), device=device)
    dataset_path = Path(cfg.data.path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    normalizers = fit_normalizers(dataset_path)
    history_size = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    window = int(cfg.data.window)

    prediction = _predict_teacher_forced(
        model=model,
        decoder=decoder,
        decoder_artifact=decoder_artifact,
        rollout=rollout,
        normalizers=normalizers,
        history_size=history_size,
        num_preds=num_preds,
        window=window,
        batch_size=args.batch_size,
        device=device,
    )
    decoded = np.asarray(prediction["decoded"], dtype=np.float32)
    pred_state = decoded[:, : len(STATE_NAMES)]
    pred_forced = decoded[:, len(STATE_NAMES) + 1]
    steps = np.asarray(prediction["steps"], dtype=np.int64)

    true_summary = state_summary(np.asarray(rollout["state"], dtype=np.float32), np.asarray(rollout["forced_mode"], dtype=np.float32))
    predicted_summary = state_summary(pred_state, pred_forced)
    true_flags = safety_flags(
        state=np.asarray(rollout["state"], dtype=np.float32),
        mode=np.asarray(rollout["mode"], dtype=np.int64),
        resolved_mode=np.asarray(rollout["resolved_mode"], dtype=np.int64),
        forced_mode=np.asarray(rollout["forced_mode"], dtype=np.float32),
        storage_capacity_mb=float(rollout["env_params"]["storage_capacity_mb"]),
    )
    pred_flags = safety_flags(
        state=pred_state,
        mode=np.asarray(rollout["mode"], dtype=np.int64)[steps],
        forced_mode=pred_forced,
        storage_capacity_mb=float(rollout["env_params"]["storage_capacity_mb"]),
    )

    payload = {
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action_source": source,
        "world_model_role": "fast evaluator for externally supplied action trace; no action choice in this script",
        "prediction_mode": "teacher_forced_one_step_latent_prediction",
        "steps": int(actions.shape[0]),
        "seed": int(args.seed),
        "mode_names": list(MODE_LIST),
        "hist": mode_histogram(actions),
        "run": run,
        "decoder": {
            "artifact": relpath(Path(args.decoder)),
            "target_names": decoder_artifact.get("target_names", list(TARGET_NAMES)),
            "created_at": decoder_artifact.get("created_at"),
        },
        "true_summary": true_summary,
        "predicted_summary": predicted_summary,
        "summary_delta_pred_minus_true": _summary_delta(predicted_summary, true_summary),
        "decoder_error_metrics": prediction["metrics"],
        "safety_flags": {
            "true": true_flags,
            "predicted": pred_flags,
        },
        "environment": rollout["env_params"],
        "timeline": _compact_timeline(steps, decoded, rollout, args.max_points),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="")
    parser.add_argument(
        "--policy",
        choices=("heuristic", "always_communicate", "always_observe", "random"),
        default="heuristic",
    )
    parser.add_argument("--steps", type=int, default=10080)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.0)
    parser.add_argument("--decoder", default=str(DECODER))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    payload = evaluate(args)
    out_path = Path(args.out)
    write_json(out_path, payload)
    print(
        f"wrote {relpath(out_path)} action_source={payload['action_source']} "
        f"unsafe={payload['safety_flags']['true']['unsafe']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
