"""Train a frozen OD LeWM latent decoder to ECI state.

This is a probe, not a replacement for the world-model objective. It freezes the
latest OD LeWM checkpoint, builds latent-window features for the generated OD
dataset, and trains a small supervised decoder:

    latent sequence -> [x, y, z, vx, vy, vz]

Usage:
    uv run python scripts/train_od_latent_decoder.py
"""
from __future__ import annotations

import argparse
import json
import math
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

DATASET = ROOT / "data/cache/od_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/od_latent_decoder.pt"
METRICS_OUT = ROOT / "data/figures/od_latent_decoder_metrics.json"


class StateDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, depth: int = 3, output_dim: int = 6):
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


def _timer() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_checkpoint_run() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for sidecar_path in RUN_ROOT.glob("*/*/*/sidecar.json"):
        sidecar = _read_json(sidecar_path)
        hparams = sidecar.get("hparams", {})
        checkpoint = sidecar.get("checkpoint_path") or str(sidecar_path.parent / "checkpoints/last.ckpt")
        if (
            hparams.get("embed_dim") is None
            or hparams.get("model._target_") != "models.od_jepa.ODJEPA"
            or hparams.get("data.path") != "data/cache/od_trajectories.npz"
            or not Path(checkpoint).exists()
        ):
            continue
        candidates.append(
            {
                "run_id": sidecar.get("run_id", sidecar_path.parent.name),
                "run_name": hparams.get("wandb.config.name") or sidecar.get("run_id", sidecar_path.parent.name),
                "run_dir": str(sidecar_path.parent),
                "checkpoint": checkpoint,
                "updated_at": float(sidecar.get("updated_at", sidecar_path.stat().st_mtime)),
            }
        )
    candidates.sort(key=lambda r: r["updated_at"], reverse=True)
    if not candidates:
        raise FileNotFoundError("no OD LeWM checkpoint found")
    return candidates[0]


def _load_model() -> tuple[Any, Any, dict[str, Any]]:
    import hydra
    from omegaconf import OmegaConf

    torch.backends.nnpack.enabled = False
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    run = _latest_checkpoint_run()
    cfg = OmegaConf.load(Path(run["run_dir"]) / "hparams.yaml")
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = {
        key[len("model.") :]: value
        for key, value in checkpoint.get("state_dict", checkpoint).items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, cfg, run


def _normalizers(obs: np.ndarray, action: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for key, arr in {"obs": obs, "action": action}.items():
        flat = arr.reshape(-1, arr.shape[-1])
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
        out[key] = (mean, std)
    return out


@dataclass
class FeatureBundle:
    pred_x: np.ndarray
    target_x: np.ndarray
    state_y: np.ndarray
    episode: np.ndarray
    time_min: np.ndarray
    history_size: int
    num_preds: int
    pred_len: int


def _build_features(model: Any, cfg: Any, dataset: dict[str, np.ndarray], batch_size: int) -> FeatureBundle:
    obs = dataset["obs"].astype(np.float32)
    action = dataset["action"].astype(np.float32)
    state = dataset["state"].astype(np.float32)
    history = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    window = int(cfg.data.window)
    norms = _normalizers(obs, action)
    obs_norm = (obs - norms["obs"][0]) / norms["obs"][1]
    action_norm = (action - norms["action"][0]) / norms["action"][1]

    index = [(e, s) for e in range(obs.shape[0]) for s in range(obs.shape[1] - window + 1)]
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    ep_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    pred_len = 0

    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            chunk = index[start : start + batch_size]
            obs_windows = np.stack([obs_norm[e, s : s + window] for e, s in chunk])
            act_windows = np.stack([action_norm[e, s : s + window] for e, s in chunk])
            batch = {
                "obs": torch.tensor(obs_windows, dtype=torch.float32),
                "action": torch.tensor(act_windows, dtype=torch.float32),
            }
            encoded = model.encode(batch)
            emb = encoded["emb"]
            act_emb = encoded["act_emb"]
            pred = model.predict(emb[:, :history], act_emb[:, :history])
            target = emb[:, num_preds:]
            m = min(pred.size(1), target.size(1))
            pred_len = int(m)
            pred_seq = pred[:, :m].reshape(len(chunk), -1).cpu().numpy().astype(np.float32)
            target_seq = target[:, :m].reshape(len(chunk), -1).cpu().numpy().astype(np.float32)
            final_offsets = np.asarray([s + num_preds + m - 1 for _, s in chunk], dtype=np.int64)
            episodes = np.asarray([e for e, _ in chunk], dtype=np.int64)

            pred_parts.append(pred_seq)
            target_parts.append(target_seq)
            state_parts.append(state[episodes, final_offsets])
            ep_parts.append(episodes)
            time_parts.append(final_offsets.astype(np.float32) * 30.0 / 60.0)

    return FeatureBundle(
        pred_x=np.concatenate(pred_parts, axis=0),
        target_x=np.concatenate(target_parts, axis=0),
        state_y=np.concatenate(state_parts, axis=0),
        episode=np.concatenate(ep_parts, axis=0),
        time_min=np.concatenate(time_parts, axis=0),
        history_size=history,
        num_preds=num_preds,
        pred_len=pred_len,
    )


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _rmse_by_block(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = pred - target
    pos_err = np.linalg.norm(residual[:, :3], axis=1)
    vel_err = np.linalg.norm(residual[:, 3:], axis=1)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_median_m": float(np.median(pos_err)),
        "position_max_m": float(np.max(pos_err)),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_median_m_s": float(np.median(vel_err)),
        "velocity_max_m_s": float(np.max(vel_err)),
    }


def _covariance(values: np.ndarray) -> dict[str, Any]:
    cov = np.cov(values, rowvar=False)
    return {
        "matrix": cov.tolist(),
        "trace": float(np.trace(cov)),
        "determinant": float(np.linalg.det(cov)),
        "diag": np.diag(cov).tolist(),
        "position_trace_m2": float(np.trace(cov[:3, :3])),
        "velocity_trace_m2_s2": float(np.trace(cov[3:, 3:])),
    }


def _predict(decoder: StateDecoder, x: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = (x[start : start + batch_size] - feature_mean) / feature_std
            yb = decoder(torch.tensor(xb, dtype=torch.float32)).cpu().numpy()
            outputs.append(yb * state_std + state_mean)
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--metrics-out", default=str(METRICS_OUT))
    parser.add_argument("--eval-episode", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3072)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_path = Path(args.dataset)
    dataset = {key: value for key, value in np.load(dataset_path).items()}

    load_start = _timer()
    model, cfg, run = _load_model()
    load_time_s = _elapsed(load_start)

    feature_start = _timer()
    features = _build_features(model, cfg, dataset, args.feature_batch_size)
    feature_time_s = _elapsed(feature_start)

    train_mask = features.episode != args.eval_episode
    eval_mask = features.episode == args.eval_episode
    if not np.any(eval_mask):
        raise ValueError(f"eval episode {args.eval_episode} not present in dataset")

    # Train on both model-predicted latents and encoder target latents so the
    # decoder is useful for rollout evaluation and for an encoder upper bound.
    train_features = np.concatenate([features.pred_x[train_mask], features.target_x[train_mask]], axis=0)
    train_state = np.concatenate([features.state_y[train_mask], features.state_y[train_mask]], axis=0)
    feature_mean, feature_std = _fit_standardizer(train_features)
    state_mean, state_std = _fit_standardizer(train_state)
    train_x = (train_features - feature_mean) / feature_std
    train_y = (train_state - state_mean) / state_std

    perm = np.random.permutation(len(train_x))
    n_val = max(1, int(0.1 * len(perm)))
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    train_ds = TensorDataset(torch.tensor(train_x[fit_idx], dtype=torch.float32), torch.tensor(train_y[fit_idx], dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(train_x[val_idx], dtype=torch.float32), torch.tensor(train_y[val_idx], dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    decoder = StateDecoder(train_x.shape[1], args.hidden_dim, args.depth)
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
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )

    if best_state is not None:
        decoder.load_state_dict(best_state)
    train_time_s = _elapsed(train_start)

    eval_state = features.state_y[eval_mask]
    pred_decoded = _predict(decoder, features.pred_x[eval_mask], feature_mean, feature_std, state_mean, state_std, args.batch_size)
    target_decoded = _predict(decoder, features.target_x[eval_mask], feature_mean, feature_std, state_mean, state_std, args.batch_size)
    pred_residual = pred_decoded - eval_state
    target_residual = target_decoded - eval_state
    pred_pos_err = np.linalg.norm(pred_residual[:, :3], axis=1)
    target_pos_err = np.linalg.norm(target_residual[:, :3], axis=1)

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
        "history_size": features.history_size,
        "num_preds": features.num_preds,
        "pred_len": features.pred_len,
        "run": run,
        "dataset": str(dataset_path.relative_to(ROOT) if dataset_path.is_relative_to(ROOT) else dataset_path),
        "eval_episode": args.eval_episode,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)

    metrics = {
        "generated_at": artifact["created_at"],
        "artifact": str(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path),
        "dataset": artifact["dataset"],
        "run": run,
        "eval_episode": args.eval_episode,
        "samples": int(eval_state.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "history_size": features.history_size,
        "num_preds": features.num_preds,
        "pred_len": features.pred_len,
        "load_time_s": load_time_s,
        "feature_time_s": feature_time_s,
        "train_time_s": train_time_s,
        "best_val_loss": best_val,
        "history": history,
        "predicted_latent_decode": {
            **_rmse_by_block(pred_decoded, eval_state),
            "residual_covariance": {
                "space": "ECI state residual [x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s]",
                **_covariance(pred_residual),
            },
        },
        "target_latent_decode": {
            **_rmse_by_block(target_decoded, eval_state),
            "residual_covariance": {
                "space": "ECI state residual [x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s]",
                **_covariance(target_residual),
            },
        },
        "series": {
            "time_min": features.time_min[eval_mask].tolist(),
            "predicted_position_error_m": pred_pos_err.tolist(),
            "target_position_error_m": target_pos_err.tolist(),
        },
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "wrote",
        out_path.relative_to(ROOT),
        "and",
        metrics_path.relative_to(ROOT),
        "predicted position RMSE",
        f"{metrics['predicted_latent_decode']['position_rmse_m']:.3f} m",
    )


if __name__ == "__main__":
    main()
