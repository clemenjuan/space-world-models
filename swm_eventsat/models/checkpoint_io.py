"""Shared helpers for EventSat LeWM decoder, evaluator, and MPC scripts."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swm_eventsat.data.toy_eventsat_env import ACTION_DIM, MODE_LIST, MODE_TO_INDEX, STATE_NAMES, EventSatEnv, heuristic_eventsat_policy


DATASET = ROOT / "data/cache/eventsat_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
FIGURES = ROOT / "data/figures"
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_NAMES)}
KEY_STATE_NAMES = (
    "battery_soc",
    "obc_data_mb",
    "jetson_raw_mb",
    "jetson_compressed_mb",
    "data_downlinked_mb",
)


class EventSatStateDecoder(nn.Module):
    """Small MLP probe from LeWM latent embeddings to EventSat state/reward."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 2, output_dim: int = 17):
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(int(depth)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def extract_metric(sidecar: dict[str, Any], summary: dict[str, Any], key: str) -> float | None:
    value = sidecar.get("summary", {}).get(key)
    if value is not None:
        return maybe_float(value)
    metric = summary.get("metrics", {}).get(key, {})
    return maybe_float(metric.get("last"))


def one_hot(index: int, dim: int = ACTION_DIM) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def one_hot_sequence(actions: np.ndarray | list[int], dim: int = ACTION_DIM) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    out = np.zeros((arr.shape[0], dim), dtype=np.float32)
    valid = (arr >= 0) & (arr < dim)
    out[np.arange(arr.shape[0])[valid], arr[valid]] = 1.0
    return out


def mode_histogram(actions: np.ndarray | list[int]) -> dict[str, list[Any]]:
    arr = np.asarray(actions, dtype=np.int64).reshape(-1)
    counts = np.bincount(arr, minlength=len(MODE_LIST))[: len(MODE_LIST)]
    return {"mode": list(MODE_LIST), "count": counts.astype(int).tolist()}


def find_eventsat_runs(run_root: Path = RUN_ROOT, dataset: str = "data/cache/eventsat_trajectories.npz") -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not run_root.exists():
        return runs
    for summary_path in run_root.glob("*/*/*/summary.json"):
        run_dir = summary_path.parent
        sidecar = read_json(run_dir / "sidecar.json")
        summary = read_json(summary_path)
        hparams = sidecar.get("hparams", {})
        if hparams.get("data.path") != dataset:
            continue
        model_target = hparams.get("model._target_")
        if model_target not in (None, "core.models.vector_jepa.VectorJEPA", "models.od_jepa.ODJEPA"):
            continue
        checkpoint = sidecar.get("checkpoint_path")
        if not checkpoint:
            ckpt = run_dir / "checkpoints/last.ckpt"
            checkpoint = str(ckpt) if ckpt.exists() else ""
        if not checkpoint or not Path(checkpoint).exists():
            continue
        runs.append(
            {
                "run_id": sidecar.get("run_id") or summary.get("run_id") or run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": checkpoint,
                "status": sidecar.get("status", "unknown"),
                "epoch": extract_metric(sidecar, summary, "epoch"),
                "val_pred_loss": extract_metric(sidecar, summary, "validate/pred_loss_epoch")
                or extract_metric(sidecar, summary, "validate/pred_loss"),
                "fit_pred_loss": extract_metric(sidecar, summary, "fit/pred_loss"),
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


def best_eventsat_run(dataset: str = "data/cache/eventsat_trajectories.npz") -> dict[str, Any]:
    runs = find_eventsat_runs(dataset=dataset)
    if not runs:
        raise RuntimeError(f"no EventSat VectorJEPA run with a checkpoint found for {dataset}")
    return runs[0]


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
    return out or state_dict


def load_eventsat_model(
    run: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    dataset: str = "data/cache/eventsat_trajectories.npz",
) -> tuple[Any, Any, dict[str, Any]]:
    torch.backends.nnpack.enabled = False
    device = torch.device(device)
    run = best_eventsat_run(dataset=dataset) if run is None else run
    cfg = OmegaConf.load(Path(run["run_dir"]) / "hparams.yaml")
    if cfg.model.get("_target_") == "models.od_jepa.ODJEPA":
        cfg.model._target_ = "core.models.vector_jepa.VectorJEPA"
    if cfg.model.encoder.get("_target_") == "models.od_encoder.OdEncoder":
        cfg.model.encoder._target_ = "core.models.vector_encoder.VectorEncoder"
    for section in ("predictor", "action_encoder", "projector", "pred_proj"):
        target = cfg.model[section].get("_target_")
        if isinstance(target, str) and target.startswith("module."):
            cfg.model[section]._target_ = target.replace("module.", "core.models.components.", 1)
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    state = _strip_model_prefix(checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg, run


def fit_normalizers(path: Path = DATASET) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    blob = np.load(path)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("obs", "action"):
        flat = blob[key].reshape(-1, blob[key].shape[-1]).astype(np.float32)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std[std < 1e-8] = 1.0
        out[key] = (mean.astype(np.float32), std.astype(np.float32))
    return out


def normalize_obs(obs: np.ndarray, normalizers: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mean, std = normalizers["obs"]
    return ((obs.astype(np.float32) - mean) / std).astype(np.float32)


def normalize_action(action: np.ndarray, normalizers: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mean, std = normalizers["action"]
    return ((action.astype(np.float32) - mean) / std).astype(np.float32)


def env_params(env: EventSatEnv) -> dict[str, Any]:
    return {
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
        "min_soc": float(env.min_soc),
        "observe_min_soc": float(env.observe_min_soc),
        "compress_min_soc": float(env.compress_min_soc),
        "detect_min_soc": float(env.detect_min_soc),
        "send_min_soc": float(env.send_min_soc),
    }


def rollout_action_sequence(actions: np.ndarray | list[int], seed: int = 7) -> dict[str, np.ndarray | dict[str, Any]]:
    mode = np.asarray(actions, dtype=np.int64).reshape(-1)
    env = EventSatEnv(max_steps=int(mode.shape[0]))
    obs, info = env.reset(seed=seed)

    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    resolved_rows: list[int] = []
    forced_rows: list[float] = []

    for t, action in enumerate(mode):
        action = int(action)
        if action < 0 or action >= ACTION_DIM:
            action = MODE_TO_INDEX["charging"]
        resolved_mode = env._resolve_mode(MODE_LIST[action])
        resolved_idx = MODE_TO_INDEX[resolved_mode]
        obs_rows.append(obs)
        action_rows.append(one_hot(action))
        state_rows.append(info["state"])
        resolved_rows.append(resolved_idx)
        forced_rows.append(float(resolved_idx != action))
        if t < mode.shape[0] - 1:
            obs, reward, _, _, info = env.step(action)
            reward_rows.append(float(reward))
        else:
            reward_rows.append(0.0)

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "action": np.asarray(action_rows, dtype=np.float32),
        "state": np.asarray(state_rows, dtype=np.float32),
        "reward": np.asarray(reward_rows, dtype=np.float32),
        "mode": mode.astype(np.int64),
        "resolved_mode": np.asarray(resolved_rows, dtype=np.int64),
        "forced_mode": np.asarray(forced_rows, dtype=np.float32),
        "env_params": env_params(env),
    }


def rollout_heuristic(steps: int, seed: int = 7, exploration: float = 0.0) -> dict[str, np.ndarray | dict[str, Any]]:
    env = EventSatEnv(max_steps=int(steps))
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    actions: list[int] = []
    for _ in range(int(steps)):
        actions.append(int(heuristic_eventsat_policy(env, rng=rng, exploration=exploration)))
        if len(actions) < int(steps):
            obs, _, _, _, info = env.step(actions[-1])
    return rollout_action_sequence(np.asarray(actions, dtype=np.int64), seed=seed)


def state_summary(state: np.ndarray, forced: np.ndarray | None = None) -> dict[str, float]:
    s = {name: state[:, i] for i, name in enumerate(STATE_NAMES)}
    stored = s["obc_data_mb"] + s["jetson_raw_mb"] + s["jetson_compressed_mb"]
    out = {
        "final_soc": float(s["battery_soc"][-1]),
        "min_soc": float(np.min(s["battery_soc"])),
        "final_obc_mb": float(s["obc_data_mb"][-1]),
        "final_stored_mb": float(stored[-1]),
        "max_stored_mb": float(np.max(stored)),
        "final_downlinked_mb": float(s["data_downlinked_mb"][-1]),
        "observation_min": float(s["total_observation_s"][-1] / 60.0),
        "detections": float(s["total_detections"][-1]),
    }
    if forced is not None:
        out["forced_rate"] = float(np.mean(forced))
        out["forced_steps"] = float(np.sum(forced > 0.0))
    return out


def safety_flags(
    state: np.ndarray,
    mode: np.ndarray,
    resolved_mode: np.ndarray | None = None,
    forced_mode: np.ndarray | None = None,
    storage_capacity_mb: float = 4096.0,
) -> dict[str, Any]:
    s = {name: state[:, i] for i, name in enumerate(STATE_NAMES)}
    mode = np.asarray(mode, dtype=np.int64).reshape(-1)
    stored = s["obc_data_mb"] + s["jetson_raw_mb"] + s["jetson_compressed_mb"]
    invalid_comm = (mode == MODE_TO_INDEX["communication"]) & (s["ground_pass_active"] < 0.5)
    payload = np.isin(
        mode,
        [
            MODE_TO_INDEX["payload_observe"],
            MODE_TO_INDEX["payload_compress"],
            MODE_TO_INDEX["payload_detect"],
            MODE_TO_INDEX["payload_send"],
        ],
    )
    low_payload_soc = payload & (s["battery_soc"] < 0.3)
    forced_steps = int(np.sum(forced_mode > 0.0)) if forced_mode is not None else None
    unresolved = None
    if resolved_mode is not None:
        unresolved = int(np.sum(np.asarray(resolved_mode, dtype=np.int64).reshape(-1) != mode))
    return {
        "min_soc": float(np.min(s["battery_soc"])),
        "low_soc_steps": int(np.sum(s["battery_soc"] < 0.25)),
        "invalid_communication_steps": int(np.sum(invalid_comm)),
        "payload_below_min_soc_steps": int(np.sum(low_payload_soc)),
        "storage_pressure_steps": int(np.sum(stored > 0.85 * float(storage_capacity_mb))),
        "storage_over_capacity_steps": int(np.sum(stored > float(storage_capacity_mb))),
        "forced_mode_steps": forced_steps if forced_steps is not None else unresolved,
        "unsafe": bool(
            np.any(s["battery_soc"] < 0.25)
            or np.any(invalid_comm)
            or np.any(stored > 0.85 * float(storage_capacity_mb))
            or np.any(stored > float(storage_capacity_mb))
            or np.any(low_payload_soc)
        ),
    }


def decode_latents(
    decoder: EventSatStateDecoder,
    artifact: dict[str, Any],
    latents: np.ndarray,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    device = torch.device(device)
    feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(artifact["feature_std"], dtype=np.float32)
    target_mean = np.asarray(artifact["target_mean"], dtype=np.float32)
    target_std = np.asarray(artifact["target_std"], dtype=np.float32)
    flat = latents.reshape(-1, latents.shape[-1]).astype(np.float32)
    outs: list[np.ndarray] = []
    decoder.to(device)
    decoder.eval()
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            end = min(start + batch_size, flat.shape[0])
            xb = (flat[start:end] - feature_mean) / feature_std
            yb = decoder(torch.from_numpy(xb).to(device)).cpu().numpy()
            outs.append(yb * target_std + target_mean)
    decoded = np.concatenate(outs, axis=0)
    return decoded.reshape(*latents.shape[:-1], decoded.shape[-1])


def load_state_decoder(
    path: Path = FIGURES / "eventsat_state_decoder.pt",
    device: str | torch.device = "cpu",
) -> tuple[EventSatStateDecoder, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    decoder = EventSatStateDecoder(
        input_dim=int(artifact["input_dim"]),
        hidden_dim=int(artifact["hidden_dim"]),
        depth=int(artifact["depth"]),
        output_dim=int(artifact["output_dim"]),
    )
    decoder.load_state_dict(artifact["state_dict"])
    decoder.to(torch.device(device))
    decoder.eval()
    artifact = dict(artifact)
    for key in ("feature_mean", "feature_std", "target_mean", "target_std"):
        artifact[key] = artifact[key].detach().cpu().numpy() if torch.is_tensor(artifact[key]) else np.asarray(artifact[key])
    return decoder, artifact


def latent_rollout(
    model: Any,
    obs_context: np.ndarray,
    action_context: np.ndarray,
    candidate_actions: np.ndarray,
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]],
    history_size: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Autoregressively roll latent state for candidate action sequences.

    `candidate_actions` is (candidates, horizon) of mode indices. The last
    context action is replaced by the first candidate action, matching the
    training alignment where action[t] moves observation[t] toward t+1.
    """
    device = torch.device(device)
    actions = np.asarray(candidate_actions, dtype=np.int64)
    if actions.ndim == 1:
        actions = actions[None, :]
    n_candidates, horizon = actions.shape
    obs_norm = normalize_obs(obs_context[-history_size:], normalizers)
    act_norm = normalize_action(action_context[-history_size:], normalizers)
    with torch.no_grad():
        batch = {
            "obs": torch.from_numpy(obs_norm[None]).to(device),
            "action": torch.from_numpy(act_norm[None]).to(device),
        }
        encoded = model.encode(batch)
        emb_hist = encoded["emb"].repeat(n_candidates, 1, 1)
        act_hist = torch.from_numpy(np.repeat(act_norm[None], n_candidates, axis=0)).to(device)
        first_actions = normalize_action(one_hot_sequence(actions[:, 0]), normalizers)
        act_hist[:, -1, :] = torch.from_numpy(first_actions).to(device)

        pred_rows: list[torch.Tensor] = []
        for t in range(horizon):
            act_emb = model.action_encoder(act_hist[:, -history_size:])
            pred = model.predict(emb_hist[:, -history_size:], act_emb)[:, -1:]
            pred_rows.append(pred[:, 0])
            emb_hist = torch.cat([emb_hist, pred], dim=1)
            if t + 1 < horizon:
                next_actions = normalize_action(one_hot_sequence(actions[:, t + 1]), normalizers)
                act_hist = torch.cat([act_hist, torch.from_numpy(next_actions[:, None]).to(device)], dim=1)
        return torch.stack(pred_rows, dim=1).cpu().numpy().astype(np.float32)
