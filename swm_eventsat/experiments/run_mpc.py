"""Run a toy LeWM-MPC controller in the simplified EventSat environment.

The world model still predicts dynamics, not actions. This script adds the
controller layer: generate candidate action sequences, roll them through the
frozen LeWM, decode predicted mission state/reward, score candidates, execute
the first action in the real simplified simulator, and repeat.
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import deque
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

from swm_eventsat.data.toy_eventsat_env import ACTION_DIM, MODE_LIST, MODE_TO_INDEX, STATE_NAMES, EventSatEnv, heuristic_eventsat_policy
from swm_eventsat.models.checkpoint_io import (
    FIGURES,
    STATE_INDEX,
    decode_latents,
    fit_normalizers,
    latent_rollout,
    load_eventsat_model,
    load_state_decoder,
    mode_histogram,
    one_hot,
    relpath,
    rollout_heuristic,
    safety_flags,
    state_summary,
    write_json,
)


OUT = FIGURES / "eventsat_lewm_mpc_week.json"
DECODER = FIGURES / "eventsat_state_decoder.pt"


def safe_first_action_mask(env: EventSatEnv, reserve_soc: float = 0.50, allow_unsafe: bool = False) -> np.ndarray:
    """Return allowed first actions for the current real simulator state."""
    if allow_unsafe:
        return np.ones(ACTION_DIM, dtype=bool)

    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[MODE_TO_INDEX["charging"]] = True
    mask[MODE_TO_INDEX["safe"]] = env.battery_soc <= env.min_soc + 0.03

    if env.battery_soc < reserve_soc:
        return mask

    mask[MODE_TO_INDEX["communication"]] = env.is_ground_pass_active() and env.obc_data_mb > 0.01
    mask[MODE_TO_INDEX["payload_observe"]] = (
        env.battery_soc >= env.observe_min_soc
        and env.data_stored_mb + env.observation_size_mb <= env.storage_capacity_mb
    )
    mask[MODE_TO_INDEX["payload_compress"]] = env.battery_soc >= env.compress_min_soc and env.uncompressed_observations > 0
    mask[MODE_TO_INDEX["payload_detect"]] = env.battery_soc >= env.detect_min_soc and env.undetected_observations > 0
    mask[MODE_TO_INDEX["payload_send"]] = (
        env.battery_soc >= env.send_min_soc
        and env.jetson_compressed_mb > 0.01
        and env.obc_data_mb < 0.98 * env.storage_capacity_mb
    )
    return mask


def _heuristic_sequence(env: EventSatEnv, horizon: int, rng: np.random.Generator) -> np.ndarray:
    sim = copy.deepcopy(env)
    seq: list[int] = []
    for _ in range(horizon):
        action = int(heuristic_eventsat_policy(sim, rng=rng, exploration=0.0))
        seq.append(action)
        sim.step(action)
    return np.asarray(seq, dtype=np.int64)


def generate_candidate_sequences(
    env: EventSatEnv,
    horizon: int,
    n_random: int,
    rng: np.random.Generator,
    reserve_soc: float = 0.50,
    allow_unsafe: bool = False,
) -> np.ndarray:
    """Generate MPC candidates, including heuristic and repeated-mode plans."""
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    allowed = np.flatnonzero(safe_first_action_mask(env, reserve_soc=reserve_soc, allow_unsafe=allow_unsafe))
    if allowed.size == 0:
        allowed = np.asarray([MODE_TO_INDEX["charging"]], dtype=np.int64)

    rows: list[np.ndarray] = []
    rows.append(_heuristic_sequence(env, horizon, rng))
    for mode_idx in range(ACTION_DIM):
        if allow_unsafe or mode_idx in allowed:
            rows.append(np.full(horizon, mode_idx, dtype=np.int64))

    future_pool = np.arange(ACTION_DIM if allow_unsafe else ACTION_DIM - 1, dtype=np.int64)
    for _ in range(int(n_random)):
        seq = rng.choice(future_pool, size=horizon, replace=True).astype(np.int64)
        seq[0] = int(rng.choice(allowed))
        rows.append(seq)

    candidates = np.unique(np.stack(rows, axis=0), axis=0)
    return candidates.astype(np.int64)


def _switch_count(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] <= 1:
        return np.zeros(actions.shape[0], dtype=np.float32)
    return np.sum(actions[:, 1:] != actions[:, :-1], axis=1).astype(np.float32)


def score_candidates(
    decoded: np.ndarray,
    candidates: np.ndarray,
    env: EventSatEnv,
    reserve_soc: float = 0.50,
) -> np.ndarray:
    """Score decoded candidate rollouts in mission terms."""
    state = decoded[:, :, : len(STATE_NAMES)]
    reward = decoded[:, :, len(STATE_NAMES)]
    forced = decoded[:, :, len(STATE_NAMES) + 1]

    final = state[:, -1, :]
    soc = state[:, :, STATE_INDEX["battery_soc"]]
    obc = state[:, :, STATE_INDEX["obc_data_mb"]]
    raw = state[:, :, STATE_INDEX["jetson_raw_mb"]]
    comp = state[:, :, STATE_INDEX["jetson_compressed_mb"]]
    stored = obc + raw + comp
    ground = state[:, :, STATE_INDEX["ground_pass_active"]]

    downlink_gain = final[:, STATE_INDEX["data_downlinked_mb"]] - env.data_downlinked_mb
    obs_gain_min = (final[:, STATE_INDEX["total_observation_s"]] - env.total_observation_s) / 60.0
    detection_gain = final[:, STATE_INDEX["total_detections"]] - env.total_detections
    low_soc_penalty = np.sum(np.clip(0.30 - soc, 0.0, None), axis=1)
    reserve_penalty = np.clip(reserve_soc - soc[:, 0], 0.0, None)
    storage_pressure = np.sum(np.clip(stored / env.storage_capacity_mb - 0.80, 0.0, None), axis=1)
    invalid_comm = np.sum((candidates == MODE_TO_INDEX["communication"]) & (ground < 0.5), axis=1)
    switch_penalty = _switch_count(candidates)
    forced_penalty = np.sum(np.clip(forced, 0.0, 1.0), axis=1)

    score = (
        100.0 * np.sum(reward, axis=1)
        + 2.5 * np.maximum(downlink_gain, 0.0)
        + 0.45 * np.maximum(obs_gain_min, 0.0)
        + 2.0 * np.maximum(detection_gain, 0.0)
        - 55.0 * low_soc_penalty
        - 8.0 * reserve_penalty
        - 20.0 * storage_pressure
        - 4.0 * invalid_comm
        - 2.0 * forced_penalty
        - 0.05 * switch_penalty
        - 0.002 * np.maximum(stored[:, -1], 0.0)
    )
    if env.battery_soc < reserve_soc:
        score += np.where(candidates[:, 0] == MODE_TO_INDEX["charging"], 3.0, -3.0)
    return score.astype(np.float32)


def _downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


def _rollout_timeline(state: np.ndarray, actions: np.ndarray, rewards: np.ndarray, max_points: int) -> dict[str, Any]:
    idx = _downsample_indices(state.shape[0], max_points)
    stored = (
        state[:, STATE_INDEX["obc_data_mb"]]
        + state[:, STATE_INDEX["jetson_raw_mb"]]
        + state[:, STATE_INDEX["jetson_compressed_mb"]]
    )
    return {
        "step": idx.astype(int).tolist(),
        "mode": actions[idx].astype(int).tolist(),
        "mode_label": [MODE_LIST[int(i)] for i in actions[idx]],
        "soc": state[idx, STATE_INDEX["battery_soc"]].astype(float).tolist(),
        "stored_mb": stored[idx].astype(float).tolist(),
        "downlinked_mb": state[idx, STATE_INDEX["data_downlinked_mb"]].astype(float).tolist(),
        "reward": rewards[idx].astype(float).tolist(),
    }


def _comparison(mpc_summary: dict[str, float], baseline_summary: dict[str, float]) -> dict[str, float]:
    keys = (
        "final_downlinked_mb",
        "observation_min",
        "detections",
        "final_stored_mb",
        "min_soc",
        "forced_rate",
    )
    return {key: float(mpc_summary.get(key, 0.0) - baseline_summary.get(key, 0.0)) for key in keys}


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.backends.nnpack.enabled = False
    torch.set_num_threads(max(1, min(torch.get_num_threads(), args.torch_threads)))
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    model, cfg, run_info = load_eventsat_model(device=device)
    decoder, decoder_artifact = load_state_decoder(Path(args.decoder), device=device)
    dataset_path = Path(cfg.data.path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    normalizers = fit_normalizers(dataset_path)
    history_size = int(cfg.history_size)

    env = EventSatEnv(max_steps=args.steps)
    obs, info = env.reset(seed=args.seed)
    obs_hist = deque([obs.copy() for _ in range(history_size)], maxlen=history_size)
    act_hist = deque([one_hot(MODE_TO_INDEX["charging"]) for _ in range(history_size)], maxlen=history_size)

    obs_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    action_rows: list[int] = []
    resolved_rows: list[int] = []
    forced_rows: list[float] = []
    reward_rows: list[float] = []
    best_score_rows: list[float] = []
    candidate_count_rows: list[int] = []

    for t in range(int(args.steps)):
        candidates = generate_candidate_sequences(
            env=env,
            horizon=args.horizon,
            n_random=args.candidates,
            rng=rng,
            reserve_soc=args.reserve_soc,
            allow_unsafe=args.allow_unsafe_plans,
        )
        pred_latents = latent_rollout(
            model=model,
            obs_context=np.asarray(obs_hist, dtype=np.float32),
            action_context=np.asarray(act_hist, dtype=np.float32),
            candidate_actions=candidates,
            normalizers=normalizers,
            history_size=history_size,
            device=device,
        )
        decoded = decode_latents(decoder, decoder_artifact, pred_latents, batch_size=args.decode_batch_size, device=device)
        scores = score_candidates(decoded, candidates, env=env, reserve_soc=args.reserve_soc)
        best_idx = int(np.argmax(scores))
        action = int(candidates[best_idx, 0])
        resolved_mode = env._resolve_mode(MODE_LIST[action])
        resolved_idx = MODE_TO_INDEX[resolved_mode]

        obs_rows.append(obs.copy())
        state_rows.append(info["state"].copy())
        action_rows.append(action)
        resolved_rows.append(resolved_idx)
        forced_rows.append(float(resolved_idx != action))
        best_score_rows.append(float(scores[best_idx]))
        candidate_count_rows.append(int(candidates.shape[0]))

        if t < int(args.steps) - 1:
            obs, reward, _, _, info = env.step(action)
            reward_rows.append(float(reward))
            obs_hist.append(obs.copy())
            act_hist.append(one_hot(action))
        else:
            reward_rows.append(0.0)

        if args.log_every > 0 and (t + 1) % args.log_every == 0:
            print(
                f"step {t + 1}/{args.steps} mode={MODE_LIST[action]} "
                f"soc={env.battery_soc:.3f} stored={env.data_stored_mb:.2f} "
                f"downlinked={env.data_downlinked_mb:.2f}",
                flush=True,
            )

    state = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.int64)
    resolved = np.asarray(resolved_rows, dtype=np.int64)
    forced = np.asarray(forced_rows, dtype=np.float32)
    rewards = np.asarray(reward_rows, dtype=np.float32)
    baseline = rollout_heuristic(steps=args.steps, seed=args.seed, exploration=0.0)

    mpc_summary = state_summary(state, forced)
    baseline_summary = state_summary(np.asarray(baseline["state"], dtype=np.float32), np.asarray(baseline["forced_mode"], dtype=np.float32))
    mpc_flags = safety_flags(
        state=state,
        mode=actions,
        resolved_mode=resolved,
        forced_mode=forced,
        storage_capacity_mb=float(env.storage_capacity_mb),
    )
    baseline_flags = safety_flags(
        state=np.asarray(baseline["state"], dtype=np.float32),
        mode=np.asarray(baseline["mode"], dtype=np.int64),
        resolved_mode=np.asarray(baseline["resolved_mode"], dtype=np.int64),
        forced_mode=np.asarray(baseline["forced_mode"], dtype=np.float32),
        storage_capacity_mb=float(env.storage_capacity_mb),
    )

    payload = {
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "mode_names": list(MODE_LIST),
        "action_source": "LeWM-MPC controller: world model scores candidate action sequences; first action is executed closed-loop",
        "world_model_role": "planner evaluator inside MPC, not a direct policy network",
        "controller": {
            "horizon": int(args.horizon),
            "random_candidates_per_step": int(args.candidates),
            "candidate_count_mean": float(np.mean(candidate_count_rows)),
            "candidate_count_min": int(np.min(candidate_count_rows)),
            "candidate_count_max": int(np.max(candidate_count_rows)),
            "reserve_soc": float(args.reserve_soc),
            "allow_unsafe_plans": bool(args.allow_unsafe_plans),
            "score": "predicted reward + downlink/observation/detection gains - low SoC/storage/invalid/forced/switching penalties",
        },
        "run": run_info,
        "decoder": {
            "artifact": relpath(Path(args.decoder)),
            "created_at": decoder_artifact.get("created_at"),
        },
        "summary": {
            "lewm_mpc": mpc_summary,
            "heuristic": baseline_summary,
            "delta_lewm_mpc_minus_heuristic": _comparison(mpc_summary, baseline_summary),
        },
        "safety_flags": {
            "lewm_mpc": mpc_flags,
            "heuristic": baseline_flags,
        },
        "hist": {
            "lewm_mpc": mode_histogram(actions),
            "heuristic": mode_histogram(np.asarray(baseline["mode"], dtype=np.int64)),
        },
        "timeline": {
            "lewm_mpc": _rollout_timeline(state, actions, rewards, args.max_points),
            "heuristic": _rollout_timeline(
                np.asarray(baseline["state"], dtype=np.float32),
                np.asarray(baseline["mode"], dtype=np.int64),
                np.asarray(baseline["reward"], dtype=np.float32),
                args.max_points,
            ),
        },
        "best_score": {
            "mean": float(np.mean(best_score_rows)),
            "min": float(np.min(best_score_rows)),
            "max": float(np.max(best_score_rows)),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10080)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--reserve-soc", type=float, default=0.50)
    parser.add_argument("--allow-unsafe-plans", action="store_true")
    parser.add_argument("--decoder", default=str(DECODER))
    parser.add_argument("--decode-batch-size", type=int, default=4096)
    parser.add_argument("--max-points", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    payload = run(args)
    out_path = Path(args.out)
    write_json(out_path, payload)
    delta = payload["summary"]["delta_lewm_mpc_minus_heuristic"]
    print(
        f"wrote {relpath(out_path)} "
        f"downlink_delta={delta['final_downlinked_mb']:.3g}MB "
        f"unsafe={payload['safety_flags']['lewm_mpc']['unsafe']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
