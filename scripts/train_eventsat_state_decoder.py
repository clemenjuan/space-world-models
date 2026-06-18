"""Train a frozen EventSat LeWM latent decoder to mission state and reward.

This is a probe on top of the existing ODJEPA world model. It does not change
the LeWM objective; it makes latent predictions interpretable as EventSat
operations quantities such as SoC, OBC data, downlink, reward, and forced mode.
"""
from __future__ import annotations

import argparse
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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from envs.eventsat_env import MODE_LIST, STATE_NAMES
from eventsat_world_model_utils import (
    DATASET,
    FIGURES,
    KEY_STATE_NAMES,
    STATE_INDEX,
    EventSatStateDecoder,
    fit_normalizers,
    load_eventsat_model,
    normalize_action,
    normalize_obs,
    relpath,
    write_json,
)


OUT = FIGURES / "eventsat_state_decoder.pt"
METRICS_OUT = FIGURES / "eventsat_state_decoder_metrics.json"
TARGET_NAMES = tuple(STATE_NAMES) + ("reward", "forced_mode")


@dataclass
class FeatureBundle:
    pred_x: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    persistence_y: np.ndarray
    episode: np.ndarray
    step: np.ndarray
    resolved_mode: np.ndarray
    history_size: int
    num_preds: int
    pred_len: int


def _timer() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _windows_index(n_episodes: int, n_steps: int, window: int) -> list[tuple[int, int]]:
    return [(e, s) for e in range(n_episodes) for s in range(n_steps - window + 1)]


def _build_features(
    model: Any,
    cfg: Any,
    dataset: dict[str, np.ndarray],
    dataset_path: Path,
    batch_size: int,
    max_samples: int | None,
    seed: int,
    device: torch.device,
) -> FeatureBundle:
    obs = dataset["obs"].astype(np.float32)
    action = dataset["action"].astype(np.float32)
    state = dataset["state"].astype(np.float32)
    reward = dataset["reward"].astype(np.float32)
    resolved = dataset.get("resolved_mode", np.argmax(action, axis=-1)).astype(np.int64)
    forced = dataset.get("forced_mode", (np.argmax(action, axis=-1) != resolved).astype(np.float32)).astype(np.float32)

    history = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    window = int(cfg.data.window)
    normalizers = fit_normalizers(dataset_path)
    obs_norm = normalize_obs(obs, normalizers)
    action_norm = normalize_action(action, normalizers)

    index = _windows_index(obs.shape[0], obs.shape[1], window)
    if max_samples is not None and max_samples > 0 and max_samples < len(index):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(index), size=int(max_samples), replace=False))
        index = [index[int(i)] for i in selected]

    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    target_y_parts: list[np.ndarray] = []
    persistence_y_parts: list[np.ndarray] = []
    ep_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    mode_parts: list[np.ndarray] = []
    pred_len = 0

    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            chunk = index[start : start + batch_size]
            obs_windows = np.stack([obs_norm[e, s : s + window] for e, s in chunk]).astype(np.float32)
            act_windows = np.stack([action_norm[e, s : s + window] for e, s in chunk]).astype(np.float32)
            batch = {
                "obs": torch.from_numpy(obs_windows).to(device),
                "action": torch.from_numpy(act_windows).to(device),
            }
            encoded = model.encode(batch)
            emb = encoded["emb"]
            act_emb = encoded["act_emb"]
            pred = model.predict(emb[:, :history], act_emb[:, :history])
            target = emb[:, num_preds:]
            m = min(pred.size(1), target.size(1))
            pred_len = int(m)
            pred_last = pred[:, m - 1].cpu().numpy().astype(np.float32)
            target_last = target[:, m - 1].cpu().numpy().astype(np.float32)

            episodes = np.asarray([e for e, _ in chunk], dtype=np.int64)
            final_offsets = np.asarray([s + num_preds + m - 1 for _, s in chunk], dtype=np.int64)
            persistence_offsets = np.asarray([s + history - 1 for _, s in chunk], dtype=np.int64)
            target_y = np.concatenate(
                [
                    state[episodes, final_offsets],
                    reward[episodes, final_offsets, None],
                    forced[episodes, final_offsets, None],
                ],
                axis=1,
            ).astype(np.float32)
            persistence_y = np.concatenate(
                [
                    state[episodes, persistence_offsets],
                    reward[episodes, persistence_offsets, None],
                    forced[episodes, persistence_offsets, None],
                ],
                axis=1,
            ).astype(np.float32)

            pred_parts.append(pred_last)
            target_parts.append(target_last)
            target_y_parts.append(target_y)
            persistence_y_parts.append(persistence_y)
            ep_parts.append(episodes)
            step_parts.append(final_offsets)
            mode_parts.append(resolved[episodes, final_offsets])

    return FeatureBundle(
        pred_x=np.concatenate(pred_parts, axis=0),
        target_x=np.concatenate(target_parts, axis=0),
        target_y=np.concatenate(target_y_parts, axis=0),
        persistence_y=np.concatenate(persistence_y_parts, axis=0),
        episode=np.concatenate(ep_parts, axis=0),
        step=np.concatenate(step_parts, axis=0),
        resolved_mode=np.concatenate(mode_parts, axis=0),
        history_size=history,
        num_preds=num_preds,
        pred_len=pred_len,
    )


def _predict(
    decoder: EventSatStateDecoder,
    x: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    decoder.to(device)
    decoder.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = (x[start:end] - feature_mean) / feature_std
            yb = decoder(torch.from_numpy(xb).to(device)).cpu().numpy()
            outputs.append(yb * target_std + target_mean)
    return np.concatenate(outputs, axis=0)


def _regression_metrics(pred: np.ndarray, target: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    residual = pred - target
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    mae = np.mean(np.abs(residual), axis=0)
    return {
        "rmse": {name: float(rmse[i]) for i, name in enumerate(names)},
        "mae": {name: float(mae[i]) for i, name in enumerate(names)},
    }


def _binary_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_b = pred >= threshold
    target_b = target >= threshold
    tp = int(np.sum(pred_b & target_b))
    fp = int(np.sum(pred_b & ~target_b))
    fn = int(np.sum(~pred_b & target_b))
    tn = int(np.sum(~pred_b & ~target_b))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / max(1, tp + fp + fn + tn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _mode_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_idx = np.clip(np.rint(pred), 0, len(MODE_LIST) - 1).astype(np.int64)
    target_idx = np.clip(np.rint(target), 0, len(MODE_LIST) - 1).astype(np.int64)
    return float(np.mean(pred_idx == target_idx))


def _per_mode_errors(pred: np.ndarray, target: np.ndarray, mode: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(MODE_LIST):
        mask = mode == idx
        if not np.any(mask):
            continue
        sub_pred = pred[mask]
        sub_target = target[mask]
        residual = sub_pred - sub_target
        row: dict[str, Any] = {
            "mode": name,
            "samples": int(np.sum(mask)),
            "reward_rmse": float(np.sqrt(np.mean(residual[:, len(STATE_NAMES)] ** 2))),
        }
        for key in KEY_STATE_NAMES:
            state_idx = STATE_INDEX[key]
            row[f"{key}_rmse"] = float(np.sqrt(np.mean(residual[:, state_idx] ** 2)))
        rows.append(row)
    return rows


def _baseline_metrics(eval_y: np.ndarray, persistence_y: np.ndarray, train_mean: np.ndarray) -> dict[str, Any]:
    mean_pred = np.repeat(train_mean[None], eval_y.shape[0], axis=0)
    return {
        "mean": _regression_metrics(mean_pred, eval_y, TARGET_NAMES),
        "persistence": _regression_metrics(persistence_y, eval_y, TARGET_NAMES),
    }


def _acceptance(decoder_metrics: dict[str, Any], baselines: dict[str, Any], metric_group: str) -> dict[str, Any]:
    rmse = decoder_metrics[metric_group]["rmse"]
    mean_rmse = baselines["mean"]["rmse"]
    persistence_rmse = baselines["persistence"]["rmse"]
    rows = {}
    for key in KEY_STATE_NAMES + ("reward",):
        rows[key] = {
            "decoder_rmse": float(rmse[key]),
            "mean_rmse": float(mean_rmse[key]),
            "persistence_rmse": float(persistence_rmse[key]),
            "beats_mean": bool(rmse[key] < mean_rmse[key]),
            "beats_persistence": bool(rmse[key] < persistence_rmse[key]),
            "beats_best_simple": bool(rmse[key] < min(mean_rmse[key], persistence_rmse[key])),
        }
    return {
        "by_target": rows,
        "beats_mean_on_all_key_targets": bool(all(row["beats_mean"] for row in rows.values())),
        "beats_persistence_on_all_key_targets": bool(all(row["beats_persistence"] for row in rows.values())),
        "beats_best_simple_on_all_key_targets": bool(all(row["beats_best_simple"] for row in rows.values())),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.backends.nnpack.enabled = False
    torch.set_num_threads(max(1, min(torch.get_num_threads(), args.torch_threads)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset = {key: value for key, value in np.load(dataset_path).items()}

    load_start = _timer()
    model, cfg, run = load_eventsat_model(device=device)
    load_time_s = _elapsed(load_start)

    feature_start = _timer()
    features = _build_features(
        model=model,
        cfg=cfg,
        dataset=dataset,
        dataset_path=dataset_path,
        batch_size=args.feature_batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        device=device,
    )
    feature_time_s = _elapsed(feature_start)

    episodes = np.unique(features.episode)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    n_eval_eps = max(1, int(math.ceil(len(episodes) * args.eval_fraction)))
    eval_episodes = np.sort(episodes[:n_eval_eps])
    eval_mask = np.isin(features.episode, eval_episodes)
    train_mask = ~eval_mask
    if not np.any(train_mask) or not np.any(eval_mask):
        raise ValueError("decoder split produced an empty train or eval set")

    train_features = np.concatenate([features.target_x[train_mask], features.pred_x[train_mask]], axis=0)
    train_targets = np.concatenate([features.target_y[train_mask], features.target_y[train_mask]], axis=0)
    feature_mean, feature_std = _fit_standardizer(train_features)
    target_mean, target_std = _fit_standardizer(train_targets)
    train_x = (train_features - feature_mean) / feature_std
    train_y = (train_targets - target_mean) / target_std

    perm = rng.permutation(train_x.shape[0])
    n_val = max(1, int(args.validation_fraction * len(perm)))
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    train_ds = TensorDataset(torch.from_numpy(train_x[fit_idx]), torch.from_numpy(train_y[fit_idx]))
    val_ds = TensorDataset(torch.from_numpy(train_x[val_idx]), torch.from_numpy(train_y[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    decoder = EventSatStateDecoder(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        output_dim=train_y.shape[1],
    )
    decoder.to(device)
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
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)
        train_loss /= max(1, train_n)

        decoder.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = loss_fn(decoder(xb), yb)
                val_loss += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)
        val_loss /= max(1, val_n)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best={best_val:.6f}",
                flush=True,
            )

    if best_state is not None:
        decoder.load_state_dict(best_state)
    train_time_s = _elapsed(train_start)

    eval_y = features.target_y[eval_mask]
    eval_persistence = features.persistence_y[eval_mask]
    pred_decoded = _predict(decoder, features.pred_x[eval_mask], feature_mean, feature_std, target_mean, target_std, args.batch_size, device)
    target_decoded = _predict(decoder, features.target_x[eval_mask], feature_mean, feature_std, target_mean, target_std, args.batch_size, device)
    baselines = _baseline_metrics(eval_y, eval_persistence, target_mean)

    pred_metrics = _regression_metrics(pred_decoded, eval_y, TARGET_NAMES)
    target_metrics = _regression_metrics(target_decoded, eval_y, TARGET_NAMES)
    pred_metrics["binary"] = {
        "in_sunlight": _binary_metrics(pred_decoded[:, STATE_INDEX["in_sunlight"]], eval_y[:, STATE_INDEX["in_sunlight"]]),
        "ground_pass_active": _binary_metrics(
            pred_decoded[:, STATE_INDEX["ground_pass_active"]],
            eval_y[:, STATE_INDEX["ground_pass_active"]],
        ),
        "forced_mode": _binary_metrics(pred_decoded[:, len(STATE_NAMES) + 1], eval_y[:, len(STATE_NAMES) + 1]),
    }
    target_metrics["binary"] = {
        "in_sunlight": _binary_metrics(target_decoded[:, STATE_INDEX["in_sunlight"]], eval_y[:, STATE_INDEX["in_sunlight"]]),
        "ground_pass_active": _binary_metrics(
            target_decoded[:, STATE_INDEX["ground_pass_active"]],
            eval_y[:, STATE_INDEX["ground_pass_active"]],
        ),
        "forced_mode": _binary_metrics(target_decoded[:, len(STATE_NAMES) + 1], eval_y[:, len(STATE_NAMES) + 1]),
    }
    pred_metrics["mode_accuracy"] = _mode_accuracy(pred_decoded[:, STATE_INDEX["current_mode_idx"]], eval_y[:, STATE_INDEX["current_mode_idx"]])
    target_metrics["mode_accuracy"] = _mode_accuracy(
        target_decoded[:, STATE_INDEX["current_mode_idx"]],
        eval_y[:, STATE_INDEX["current_mode_idx"]],
    )

    created_at = datetime.now(timezone.utc).isoformat()
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)
    artifact = {
        "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "output_dim": int(train_y.shape[1]),
        "feature_mean": torch.tensor(feature_mean),
        "feature_std": torch.tensor(feature_std),
        "target_mean": torch.tensor(target_mean),
        "target_std": torch.tensor(target_std),
        "target_names": list(TARGET_NAMES),
        "state_names": list(STATE_NAMES),
        "history_size": int(features.history_size),
        "num_preds": int(features.num_preds),
        "pred_len": int(features.pred_len),
        "dataset": relpath(dataset_path),
        "run": run,
        "created_at": created_at,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)

    metrics = {
        "ok": True,
        "generated_at": created_at,
        "artifact": relpath(out_path),
        "dataset": relpath(dataset_path),
        "run": run,
        "samples": {
            "total_windows": int(features.target_y.shape[0]),
            "train_windows": int(np.sum(train_mask)),
            "eval_windows": int(np.sum(eval_mask)),
            "eval_episodes": eval_episodes.astype(int).tolist(),
        },
        "input_dim": int(train_x.shape[1]),
        "output_dim": int(train_y.shape[1]),
        "target_names": list(TARGET_NAMES),
        "history_size": int(features.history_size),
        "num_preds": int(features.num_preds),
        "pred_len": int(features.pred_len),
        "best_val_loss": float(best_val),
        "load_time_s": float(load_time_s),
        "feature_time_s": float(feature_time_s),
        "train_time_s": float(train_time_s),
        "history": history,
        "target_latent_decode": target_metrics,
        "predicted_latent_decode": pred_metrics,
        "baselines": baselines,
        "acceptance": {
            "target_latent_decode": _acceptance({"target": target_metrics}, baselines, "target"),
            "predicted_latent_decode": _acceptance({"pred": pred_metrics}, baselines, "pred"),
        },
        "per_mode_error": {
            "target_latent_decode": _per_mode_errors(target_decoded, eval_y, features.resolved_mode[eval_mask]),
            "predicted_latent_decode": _per_mode_errors(pred_decoded, eval_y, features.resolved_mode[eval_mask]),
        },
    }
    write_json(metrics_path, metrics)
    print(
        f"wrote {relpath(out_path)} and {relpath(metrics_path)} "
        f"battery_rmse={pred_metrics['rmse']['battery_soc']:.4g} "
        f"reward_rmse={pred_metrics['rmse']['reward']:.4g}",
        flush=True,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--metrics-out", default=str(METRICS_OUT))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="optional window cap for smoke tests")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    args.max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else None
    train(args)


if __name__ == "__main__":
    main()
