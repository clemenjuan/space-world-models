# Train a geometry-aware OD decoder baseline.
#
# This is the first target-construction sanity check for OD state decoding. It does
# not use LeWM. It asks whether raw tracking measurements plus timestamp/station
# ECI geometry can decode the hidden ECI state on held-out episodes.
#
# Usage:
#     uv run python data/generate_dataset.py --save-geometry --out data/cache/od_trajectories_geometry.npz
#     uv run python scripts/train_od_geometry_decoder.py --dataset data/cache/od_trajectories_geometry.npz --mode raw
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "data/cache/od_trajectories_geometry.npz"
DEFAULT_OUT = ROOT / "data/figures/od_geometry_decoder_raw.pt"
DEFAULT_METRICS_OUT = ROOT / "data/figures/od_geometry_decoder_raw_metrics.json"


class GeometryDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 2, output_dim: int = 6):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


@dataclass
class FeatureBundle:
    x: np.ndarray
    y: np.ndarray
    episode: np.ndarray
    time_min: np.ndarray
    target_index: np.ndarray
    window: int
    step_feature_dim: int


def _timer() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _angle_features(obs: np.ndarray) -> np.ndarray:
    rng = obs[..., 0:1]
    az = obs[..., 1:2]
    el = obs[..., 2:3]
    rr = obs[..., 3:4]
    return np.concatenate(
        [rng, np.sin(az), np.cos(az), np.sin(el), np.cos(el), rr], axis=-1
    ).astype(np.float32)


def _safe_unit(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, eps)


def _station_enu_basis(station_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = _safe_unit(station_state[..., :3])
    east = _safe_unit(station_state[..., 3:])
    north = _safe_unit(np.cross(up, east))
    east = _safe_unit(np.cross(north, up))
    return east.astype(np.float32), north.astype(np.float32), up.astype(np.float32)


def _eci_measurement_features(
    obs: np.ndarray,
    station_state: np.ndarray,
    topocentric_basis: np.ndarray | None = None,
) -> np.ndarray:
    rng = obs[..., 0:1]
    az = obs[..., 1:2]
    el = obs[..., 2:3]
    rr = obs[..., 3:4]
    if topocentric_basis is None:
        east, north, up = _station_enu_basis(station_state)
    else:
        east = topocentric_basis[..., 0, :].astype(np.float32)
        north = topocentric_basis[..., 1, :].astype(np.float32)
        up = topocentric_basis[..., 2, :].astype(np.float32)
    cos_el = np.cos(el)
    los = (
        cos_el * np.sin(az) * east
        + cos_el * np.cos(az) * north
        + np.sin(el) * up
    ).astype(np.float32)
    measured_pos = station_state[..., :3] + rng * los
    radial_velocity = station_state[..., 3:] + rr * los
    return np.concatenate([los, measured_pos, radial_velocity], axis=-1).astype(np.float32)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


def _require_geometry(dataset: dict[str, np.ndarray]) -> None:
    missing = [key for key in ("time_s", "station_state_eci") if key not in dataset]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"dataset is missing geometry keys: {joined}. Regenerate with data/generate_dataset.py --save-geometry."
        )


def _build_raw_features(dataset: dict[str, np.ndarray], window: int, mode: str = "raw") -> FeatureBundle:
    _require_geometry(dataset)
    obs = dataset["obs"].astype(np.float32)
    state = dataset["state"].astype(np.float32)
    time_s = dataset["time_s"].astype(np.float32)[..., None]
    station_state = dataset["station_state_eci"].astype(np.float32)
    if obs.ndim != 3 or state.ndim != 3:
        raise ValueError("expected obs/state arrays shaped (episode, time, feature)")
    if station_state.shape[:2] != obs.shape[:2] or station_state.shape[-1] != 6:
        raise ValueError("station_state_eci must have shape (episode, time, 6)")
    if time_s.shape[:2] != obs.shape[:2]:
        raise ValueError("time_s must have shape (episode, time)")
    if window < 1 or window > obs.shape[1]:
        raise ValueError(f"window must be in [1, {obs.shape[1]}], got {window}")

    obs_features = _angle_features(obs)
    parts = [obs_features, time_s, station_state]
    if mode == "raw_eci":
        basis = dataset.get("topocentric_basis_eci")
        if basis is not None:
            basis = basis.astype(np.float32)
            if basis.shape[:2] != obs.shape[:2] or basis.shape[-2:] != (3, 3):
                raise ValueError("topocentric_basis_eci must have shape (episode, time, 3, 3)")
        parts.append(_eci_measurement_features(obs, station_state, basis))
    elif mode != "raw":
        raise ValueError(f"unknown feature mode: {mode}")
    step_features = np.concatenate(parts, axis=-1)
    index = [(e, s) for e in range(obs.shape[0]) for s in range(obs.shape[1] - window + 1)]
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ep_parts: list[int] = []
    time_parts: list[float] = []
    target_parts: list[int] = []
    for e, s in index:
        target = s + window - 1
        x_parts.append(step_features[e, s : s + window].reshape(-1))
        y_parts.append(state[e, target])
        ep_parts.append(e)
        time_parts.append(float(time_s[e, target, 0]) / 60.0)
        target_parts.append(target)

    return FeatureBundle(
        x=np.asarray(x_parts, dtype=np.float32),
        y=np.asarray(y_parts, dtype=np.float32),
        episode=np.asarray(ep_parts, dtype=np.int64),
        time_min=np.asarray(time_parts, dtype=np.float32),
        target_index=np.asarray(target_parts, dtype=np.int64),
        window=window,
        step_feature_dim=int(step_features.shape[-1]),
    )


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _predict(
    decoder: GeometryDecoder,
    x: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = (x[start : start + batch_size] - feature_mean) / feature_std
            yb = decoder(torch.tensor(xb, dtype=torch.float32)).cpu().numpy()
            outputs.append(yb * state_std + state_mean)
    return np.concatenate(outputs, axis=0)


def _rmse_by_block(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = pred - target
    pos_err = np.linalg.norm(residual[:, :3], axis=1)
    vel_err = np.linalg.norm(residual[:, 3:], axis=1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_median_m": float(np.median(pos_err)),
        "position_p95_m": float(np.percentile(pos_err, 95)),
        "position_max_m": float(np.max(pos_err)),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_median_m_s": float(np.median(vel_err)),
        "velocity_p95_m_s": float(np.percentile(vel_err, 95)),
        "velocity_max_m_s": float(np.max(vel_err)),
    }


def _covariance(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 2:
        return {"matrix": [], "trace": None, "determinant": None, "diag": []}
    cov = np.cov(values, rowvar=False)
    return {
        "matrix": cov.tolist(),
        "trace": float(np.trace(cov)),
        "determinant": float(np.linalg.det(cov)),
        "diag": np.diag(cov).tolist(),
        "position_trace_m2": float(np.trace(cov[:3, :3])),
        "velocity_trace_m2_s2": float(np.trace(cov[3:, 3:])),
    }


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--mode", choices=["raw", "raw_eci"], default="raw")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--metrics-out", default=str(DEFAULT_METRICS_OUT))
    parser.add_argument("--eval-episode", type=int, default=0)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "space-world-models"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "sps-tum"))
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-group", default="od-geometry-decoder")
    parser.add_argument("--wandb-tags", nargs="*", default=["od", "geometry-decoder"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_path = Path(args.dataset)

    load_start = _timer()
    dataset = _load_dataset(dataset_path)
    features = _build_raw_features(dataset, args.window, mode=args.mode)
    load_time_s = _elapsed(load_start)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_name = args.wandb_name or f"od-geometry-{args.mode}-w{args.window}"
        wandb_run = wandb.init(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            name=wandb_name,
            group=args.wandb_group or None,
            tags=args.wandb_tags,
            config={
                **vars(args),
                "dataset_keys": sorted(dataset.keys()),
                "dataset_obs_shape": list(dataset["obs"].shape),
                "feature_samples": int(features.x.shape[0]),
                "feature_input_dim": int(features.x.shape[1]),
                "step_feature_dim": features.step_feature_dim,
            },
        )

    train_mask = features.episode != args.eval_episode
    eval_mask = features.episode == args.eval_episode
    if not np.any(train_mask):
        raise ValueError(f"no training samples remain after holding out episode {args.eval_episode}")
    if not np.any(eval_mask):
        raise ValueError(f"eval episode {args.eval_episode} not present in dataset")

    feature_mean, feature_std = _fit_standardizer(features.x[train_mask])
    state_mean, state_std = _fit_standardizer(features.y[train_mask])
    train_x = (features.x[train_mask] - feature_mean) / feature_std
    train_y = (features.y[train_mask] - state_mean) / state_std

    perm = np.random.permutation(len(train_x))
    n_val = max(1, int(0.1 * len(perm)))
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    if len(fit_idx) == 0:
        fit_idx = val_idx
    train_ds = TensorDataset(
        torch.tensor(train_x[fit_idx], dtype=torch.float32),
        torch.tensor(train_y[fit_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(train_x[val_idx], dtype=torch.float32),
        torch.tensor(train_y[val_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    decoder = GeometryDecoder(train_x.shape[1], args.hidden_dim, args.depth)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_val = math.inf
    best_state = None
    history: list[dict[str, float]] = []
    train_start = _timer()

    for epoch in range(args.epochs):
        decoder.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)

        decoder.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                loss = loss_fn(decoder(xb), yb)
                val_loss += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
        train_loss /= max(1, train_n)
        val_loss /= max(1, val_n)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/best_loss": best_val,
                },
                step=epoch + 1,
            )
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )

    if best_state is not None:
        decoder.load_state_dict(best_state)
    train_time_s = _elapsed(train_start)

    eval_x = features.x[eval_mask]
    eval_y = features.y[eval_mask]
    predict_start = _timer()
    pred = _predict(
        decoder,
        eval_x,
        feature_mean,
        feature_std,
        state_mean,
        state_std,
        args.batch_size,
    )
    predict_time_s = _elapsed(predict_start)
    residual = pred - eval_y
    pos_err = np.linalg.norm(residual[:, :3], axis=1)
    vel_err = np.linalg.norm(residual[:, 3:], axis=1)

    created_at = datetime.now(timezone.utc).isoformat()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": decoder.state_dict(),
        "input_dim": train_x.shape[1],
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "output_dim": 6,
        "feature_mean": torch.tensor(feature_mean),
        "feature_std": torch.tensor(feature_std),
        "state_mean": torch.tensor(state_mean),
        "state_std": torch.tensor(state_std),
        "mode": args.mode,
        "window": args.window,
        "step_feature_dim": features.step_feature_dim,
        "dataset": _relative_path(dataset_path),
        "eval_episode": args.eval_episode,
        "created_at": created_at,
    }
    torch.save(artifact, out_path)

    rmse = _rmse_by_block(pred, eval_y)
    metrics = {
        "generated_at": created_at,
        "artifact": _relative_path(out_path),
        "dataset": _relative_path(dataset_path),
        "mode": args.mode,
        "eval_episode": args.eval_episode,
        "samples": int(eval_y.shape[0]),
        "train_samples": int(train_x.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "window": args.window,
        "step_feature_dim": features.step_feature_dim,
        "load_time_s": load_time_s,
        "train_time_s": train_time_s,
        "predict_time_s": predict_time_s,
        "time_per_sample_ms": predict_time_s / max(1, eval_y.shape[0]) * 1000.0,
        "best_val_loss": best_val,
        "history": history,
        "raw_geometry_decode": {
            **rmse,
            "residual_covariance": {
                "space": "ECI state residual [x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s]",
                **_covariance(residual),
            },
        },
        "series": {
            "time_min": features.time_min[eval_mask].tolist(),
            "target_index": features.target_index[eval_mask].tolist(),
            "position_error_m": pos_err.tolist(),
            "velocity_error_m_s": vel_err.tolist(),
        },
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    position_rmse = metrics["raw_geometry_decode"]["position_rmse_m"]
    if wandb_run is not None:
        wandb_run.log(
            {
                **{f"eval/{key}": value for key, value in rmse.items()},
                "timing/load_time_s": load_time_s,
                "timing/train_time_s": train_time_s,
                "timing/predict_time_s": predict_time_s,
                "timing/time_per_sample_ms": metrics["time_per_sample_ms"],
                "val/best_loss": best_val,
            },
            step=args.epochs,
        )
        import wandb

        artifact = wandb.Artifact(
            f"od-geometry-decoder-{wandb_run.id}",
            type="model",
            metadata={
                "mode": args.mode,
                "window": args.window,
                "dataset": _relative_path(dataset_path),
                "position_rmse_m": position_rmse,
            },
        )
        artifact.add_file(str(out_path))
        artifact.add_file(str(metrics_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(
        "wrote",
        _relative_path(out_path),
        "and",
        _relative_path(metrics_path),
        "position RMSE",
        f"{position_rmse:.3f} m",
    )


if __name__ == "__main__":
    main()
