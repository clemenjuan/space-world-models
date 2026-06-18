"""Run a one-week EventSat rollout and score LeWM one-step latent inference."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.eventsat_env import ACTION_DIM, MODE_LIST, STATE_NAMES, EventSatEnv, heuristic_eventsat_policy


DATASET = ROOT / "data/cache/eventsat_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/eventsat_week_inference.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_metric(sidecar: dict[str, Any], summary: dict[str, Any], key: str) -> float | None:
    value = sidecar.get("summary", {}).get(key)
    if value is not None:
        return _maybe_float(value)
    metric = summary.get("metrics", {}).get(key, {})
    return _maybe_float(metric.get("last"))


def _find_eventsat_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUN_ROOT.exists():
        return runs
    for summary_path in RUN_ROOT.glob("*/*/*/summary.json"):
        run_dir = summary_path.parent
        sidecar = _read_json(run_dir / "sidecar.json")
        summary = _read_json(summary_path)
        hparams = sidecar.get("hparams", {})
        if hparams.get("data.path") != "data/cache/eventsat_trajectories.npz":
            continue
        model_target = hparams.get("model._target_")
        if model_target not in (None, "models.od_jepa.ODJEPA"):
            continue
        checkpoint = sidecar.get("checkpoint_path")
        if not checkpoint:
            ckpt = run_dir / "checkpoints/last.ckpt"
            checkpoint = str(ckpt) if ckpt.exists() else ""
        if not checkpoint or not Path(checkpoint).exists():
            continue
        val_pred = _extract_metric(sidecar, summary, "validate/pred_loss_epoch")
        runs.append(
            {
                "run_id": sidecar.get("run_id") or summary.get("run_id") or run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": checkpoint,
                "status": sidecar.get("status", "unknown"),
                "epoch": _extract_metric(sidecar, summary, "epoch"),
                "val_pred_loss": val_pred,
                "fit_pred_loss": _extract_metric(sidecar, summary, "fit/pred_loss"),
                "name": hparams.get("wandb.config.name") or sidecar.get("run_id") or run_dir.name,
            }
        )
    runs.sort(
        key=lambda run: (
            run.get("val_pred_loss") is None,
            run.get("val_pred_loss") if run.get("val_pred_loss") is not None else float("inf"),
            -(run.get("epoch") or -1),
        )
    )
    return runs


def _one_hot(index: int, dim: int = ACTION_DIM) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def _rollout(steps: int, seed: int, exploration: float) -> dict[str, np.ndarray]:
    env = EventSatEnv(max_steps=steps)
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    mode_rows: list[int] = []
    resolved_rows: list[int] = []
    forced_rows: list[float] = []

    for t in range(steps):
        action = heuristic_eventsat_policy(env, rng=rng, exploration=exploration)
        resolved_mode = env._resolve_mode(MODE_LIST[action])
        resolved_idx = MODE_LIST.index(resolved_mode)

        obs_rows.append(obs)
        action_rows.append(_one_hot(action))
        state_rows.append(info["state"])
        mode_rows.append(action)
        resolved_rows.append(resolved_idx)
        forced_rows.append(float(resolved_idx != action))

        if t < steps - 1:
            obs, reward, _, _, info = env.step(action)
            reward_rows.append(float(reward))
        else:
            reward_rows.append(0.0)

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "action": np.asarray(action_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "mode": np.asarray(mode_rows, dtype=np.int64),
        "resolved_mode": np.asarray(resolved_rows, dtype=np.int64),
        "forced_mode": np.asarray(forced_rows, dtype=np.float32),
        "env_params": {
            "obc_capacity_mb": float(env.storage_capacity_mb),
            "jetson_capacity_mb": float(env.jetson_capacity_mb),
            "observation_size_mb": float(env.observation_size_mb),
            "compression_ratio": float(env.compression_ratio),
            "compressed_observation_mb": float(env.observation_size_mb / env.compression_ratio),
            "detection_metadata_mb": float(env.detection_metadata_mb),
            "jetson_to_obc_rate_kbps": float(env.jetson_to_obc_rate_kbps),
            "downlink_rate_kbps": float(env.downlink_rate_kbps),
            "downlink_capacity_mb_per_step": float(env._downlink_capacity_mb()),
            "pass_interval_steps": int(env.pass_interval_steps),
            "pass_duration_steps": int(env.pass_duration_steps),
            "storage_capacity_mb": float(env.storage_capacity_mb),
        },
    }


def _normalizers(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("obs", "action"):
        flat = blob[key].reshape(-1, blob[key].shape[-1]).astype(np.float32)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0
        out[key] = (mean.astype(np.float32), std.astype(np.float32))
    return out


def _windows(array: np.ndarray, window: int) -> np.ndarray:
    starts = np.arange(array.shape[0] - window + 1)
    return np.stack([array[start : start + window] for start in starts]).astype(np.float32)


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
    return out or state_dict


def _load_model(run: dict[str, Any], device: torch.device):
    cfg = OmegaConf.load(Path(run["run_dir"]) / "hparams.yaml")
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(run["checkpoint"], map_location="cpu")
    state = _strip_model_prefix(ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"warning: missing model keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected model keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model, cfg


def _score_rollout(
    model,
    rollout: dict[str, np.ndarray],
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    history_size: int,
    num_preds: int,
    window: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    obs_mean, obs_std = normalizers["obs"]
    act_mean, act_std = normalizers["action"]
    obs = (rollout["obs"] - obs_mean) / obs_std
    action = (rollout["action"] - act_mean) / act_std
    obs_windows = _windows(obs, window)
    action_windows = _windows(action, window)

    mse_rows: list[np.ndarray] = []
    persistence_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, obs_windows.shape[0], batch_size):
            end = min(start + batch_size, obs_windows.shape[0])
            batch = {
                "obs": torch.from_numpy(obs_windows[start:end]).to(device),
                "action": torch.from_numpy(action_windows[start:end]).to(device),
            }
            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]
            pred = model.predict(emb[:, :history_size], act_emb[:, :history_size])
            target = emb[:, num_preds:]
            m = min(pred.size(1), target.size(1))
            pred_last = pred[:, m - 1]
            target_last = target[:, m - 1]
            persistence = emb[:, history_size - 1]
            mse_rows.append((pred_last - target_last).pow(2).mean(dim=-1).cpu().numpy())
            persistence_rows.append((persistence - target_last).pow(2).mean(dim=-1).cpu().numpy())

    starts = np.arange(obs_windows.shape[0], dtype=np.int64)
    target_step = starts + history_size
    return {
        "step": target_step,
        "mse": np.concatenate(mse_rows).astype(np.float32),
        "persistence_mse": np.concatenate(persistence_rows).astype(np.float32),
    }


def _downsample(array: np.ndarray, max_points: int) -> np.ndarray:
    if array.size <= max_points:
        return array
    idx = np.linspace(0, array.size - 1, num=max_points, dtype=np.int64)
    return array[idx]


def _mode_means(scores: dict[str, np.ndarray], resolved: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, name in enumerate(MODE_LIST):
        mask = resolved[scores["step"]] == idx
        if np.any(mask):
            out[name] = float(scores["mse"][mask].mean())
    return out


def _summary(state: np.ndarray, forced: np.ndarray) -> dict[str, float]:
    s = {name: state[:, i] for i, name in enumerate(STATE_NAMES)}
    data_stored = s["obc_data_mb"] + s["jetson_raw_mb"] + s["jetson_compressed_mb"]
    return {
        "final_soc": float(s["battery_soc"][-1]),
        "final_stored_mb": float(data_stored[-1]),
        "final_downlinked_mb": float(s["data_downlinked_mb"][-1]),
        "observation_min": float(s["total_observation_s"][-1] / 60.0),
        "detections": float(s["total_detections"][-1]),
        "forced_rate": float(forced.mean()),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runs = _find_eventsat_runs()
    if not runs:
        raise RuntimeError("no completed EventSat runs with checkpoints found")
    run = runs[0]
    device = torch.device(args.device)
    model, cfg = _load_model(run, device)
    rollout = _rollout(args.steps, args.seed, args.exploration)
    dataset_path = Path(cfg.data.path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    normalizers = _normalizers(dataset_path)
    scores = _score_rollout(
        model=model,
        rollout=rollout,
        normalizers=normalizers,
        history_size=int(cfg.history_size),
        num_preds=int(cfg.num_preds),
        window=int(cfg.data.window),
        batch_size=args.batch_size,
        device=device,
    )

    mse = scores["mse"]
    persistence = scores["persistence_mse"]
    improvement = persistence / np.maximum(mse, 1e-12)
    persistence_over_model_mean = float(persistence.mean() / max(float(mse.mean()), 1e-12))
    model_over_persistence_mean = float(mse.mean() / max(float(persistence.mean()), 1e-12))
    sample_idx = np.linspace(0, rollout["obs"].shape[0] - 1, num=min(args.max_points, rollout["obs"].shape[0]), dtype=np.int64)
    score_idx = np.linspace(0, mse.shape[0] - 1, num=min(args.max_points, mse.shape[0]), dtype=np.int64)
    state = rollout["state"]
    s = {name: state[:, i] for i, name in enumerate(STATE_NAMES)}
    stored = s["obc_data_mb"] + s["jetson_raw_mb"] + s["jetson_compressed_mb"]
    env_params = rollout["env_params"]
    observations = float(s["total_observation_s"][-1] / 60.0)
    compressed_payload_generated_mb = observations * env_params["compressed_observation_mb"]
    metadata_generated_mb = float(s["total_detections"][-1] * env_params["detection_metadata_mb"])
    data_accounting = {
        "obc_capacity_mb": env_params["obc_capacity_mb"],
        "final_obc_mb": float(s["obc_data_mb"][-1]),
        "final_stored_mb": float(stored[-1]),
        "downlinked_mb": float(s["data_downlinked_mb"][-1]),
        "observations": observations,
        "compressed_payload_generated_mb": float(compressed_payload_generated_mb),
        "metadata_generated_mb": metadata_generated_mb,
        "generated_to_obc_mb_est": float(compressed_payload_generated_mb + metadata_generated_mb),
    }

    return {
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(cfg.data.path)),
        "steps": int(args.steps),
        "step_duration_s": 60.0,
        "seed": int(args.seed),
        "exploration": float(args.exploration),
        "run": run,
        "history_size": int(cfg.history_size),
        "num_preds": int(cfg.num_preds),
        "window": int(cfg.data.window),
        "mode_names": list(MODE_LIST),
        "action_source": "scripted heuristic_eventsat_policy; no learned policy is used in this rollout",
        "learning_target": "ODJEPA latent dynamics: predict next latent observation embedding from recent observations and one-hot actions",
        "action_rules": [
            "if battery_soc < 0.50: charging",
            "elif ground_pass_active and obc_data_mb > 0.05: communication",
            "elif uncompressed_observations > 0: payload_compress",
            "elif undetected_observations > 0: payload_detect",
            "elif jetson_compressed_mb > 0.05 and OBC has space: payload_send",
            "elif battery_soc > 0.62 and stored data < 5% capacity: payload_observe",
            "else: charging",
        ],
        "environment": env_params,
        "data_accounting": data_accounting,
        "metrics": {
            "mse_mean": float(mse.mean()),
            "mse_median": float(np.median(mse)),
            "mse_p95": float(np.percentile(mse, 95)),
            "mse_max": float(mse.max()),
            "persistence_mse_mean": float(persistence.mean()),
            "persistence_mse_median": float(np.median(persistence)),
            "persistence_over_model_mean": persistence_over_model_mean,
            "model_over_persistence_mean": model_over_persistence_mean,
            "improvement_mean": float(improvement.mean()),
            "improvement_median": float(np.median(improvement)),
            "mode_mse_mean": _mode_means(scores, rollout["resolved_mode"]),
        },
        "summary": _summary(rollout["state"], rollout["forced_mode"]),
        "score_step": scores["step"][score_idx].astype(int).tolist(),
        "mse": mse[score_idx].astype(float).tolist(),
        "persistence_mse": persistence[score_idx].astype(float).tolist(),
        "time_step": sample_idx.astype(int).tolist(),
        "time_hour": (sample_idx.astype(float) / 60.0).tolist(),
        "soc": s["battery_soc"][sample_idx].astype(float).tolist(),
        "obc_mb": s["obc_data_mb"][sample_idx].astype(float).tolist(),
        "raw_mb": s["jetson_raw_mb"][sample_idx].astype(float).tolist(),
        "compressed_mb": s["jetson_compressed_mb"][sample_idx].astype(float).tolist(),
        "stored_mb": stored[sample_idx].astype(float).tolist(),
        "downlinked_mb": s["data_downlinked_mb"][sample_idx].astype(float).tolist(),
        "uncompressed_obs": s["uncompressed_observations"][sample_idx].astype(float).tolist(),
        "undetected_obs": s["undetected_observations"][sample_idx].astype(float).tolist(),
        "compression_progress": s["compression_progress"][sample_idx].astype(float).tolist(),
        "detection_progress": s["detection_progress"][sample_idx].astype(float).tolist(),
        "in_sunlight": s["in_sunlight"][sample_idx].astype(float).tolist(),
        "ground_pass": s["ground_pass_active"][sample_idx].astype(float).tolist(),
        "reward": rollout["reward"][sample_idx].astype(float).tolist(),
        "cum_reward": np.cumsum(rollout["reward"])[sample_idx].astype(float).tolist(),
        "mode": rollout["mode"][sample_idx].astype(int).tolist(),
        "mode_label": [MODE_LIST[i] for i in rollout["mode"][sample_idx]],
        "resolved_mode": rollout["resolved_mode"][sample_idx].astype(int).tolist(),
        "resolved_label": [MODE_LIST[i] for i in rollout["resolved_mode"][sample_idx]],
        "hist": {
            "mode": list(MODE_LIST),
            "count": np.bincount(rollout["mode"], minlength=len(MODE_LIST)).astype(int).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10080)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exploration", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    payload = evaluate(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    try:
        display_path = out_path.relative_to(ROOT)
    except ValueError:
        display_path = out_path
    print(f"wrote {display_path}")
    print(
        "week mse="
        f"{metrics['mse_mean']:.6g}, persistence={metrics['persistence_mse_mean']:.6g}, "
        f"persist/model(mean)={metrics['persistence_over_model_mean']:.2f}x"
    )


if __name__ == "__main__":
    main()
