"""Build a local HTML board for OD LeWM training results.

The board follows the same static-artifact pattern as the AUTOPS board: gather
local run outputs, sample the generated OD dataset, optionally probe the latest
checkpoint, then write a single HTML file under data/figures.

Usage:
    uv run python scripts/build_results_board.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = ROOT / "data/cache/od_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/results_board.html"
BENCHMARK = ROOT / "data/figures/od_method_benchmark.json"
GEOMETRY_DECODER_METRICS = [
    ROOT / "data/figures/od_geometry_decoder_raw_eci_w16_metrics.json",
    ROOT / "data/figures/od_geometry_decoder_raw_metrics.json",
]

SERIES_KEYS = [
    "fit/loss",
    "fit/pred_loss",
    "fit/sigreg_loss",
    "validate/loss_epoch",
    "validate/pred_loss_epoch",
    "validate/sigreg_loss_epoch",
    "hparams/lr_default_0",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt_metric(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1000 or (0 < abs(x) < 0.001):
        return f"{x:.2e}"
    return f"{x:.{digits}g}"


def _extract_metric(sidecar: dict[str, Any], summary: dict[str, Any], key: str) -> float | None:
    value = sidecar.get("summary", {}).get(key)
    if value is not None:
        return _maybe_float(value)
    metric = summary.get("metrics", {}).get(key, {})
    return _maybe_float(metric.get("last"))


def _series_from_csv(path: Path) -> dict[str, dict[str, list[float]]]:
    series = {key: {"x": [], "epoch": [], "y": []} for key in SERIES_KEYS}
    if not path.exists():
        return series
    with path.open(newline="", encoding="utf-8") as fh:
        for row_idx, row in enumerate(csv.DictReader(fh)):
            step = _maybe_float(row.get("step"))
            epoch = _maybe_float(row.get("epoch"))
            x = step if step is not None else float(row_idx)
            for key in SERIES_KEYS:
                y = _maybe_float(row.get(key))
                if y is None:
                    continue
                series[key]["x"].append(x)
                series[key]["epoch"].append(epoch if epoch is not None else float(row_idx))
                series[key]["y"].append(y)
    return series


def _find_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not RUN_ROOT.exists():
        return runs
    for summary_path in RUN_ROOT.glob("*/*/*/summary.json"):
        run_dir = summary_path.parent
        sidecar = _read_json(run_dir / "sidecar.json")
        summary = _read_json(summary_path)
        hparams = sidecar.get("hparams", {})
        run_id = sidecar.get("run_id") or summary.get("run_id") or run_dir.name
        checkpoint = sidecar.get("checkpoint_path")
        if not checkpoint:
            ckpt = run_dir / "checkpoints/last.ckpt"
            checkpoint = str(ckpt) if ckpt.exists() else ""
        updated = _maybe_float(sidecar.get("updated_at")) or summary_path.stat().st_mtime
        run = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "name": hparams.get("wandb.config.name") or run_id,
            "status": sidecar.get("status", "unknown"),
            "updated_at": updated,
            "updated_at_iso": datetime.fromtimestamp(updated, timezone.utc).isoformat(),
            "epoch": _extract_metric(sidecar, summary, "epoch"),
            "fit_loss": _extract_metric(sidecar, summary, "fit/loss"),
            "fit_pred_loss": _extract_metric(sidecar, summary, "fit/pred_loss"),
            "fit_sigreg_loss": _extract_metric(sidecar, summary, "fit/sigreg_loss"),
            "val_loss": _extract_metric(sidecar, summary, "validate/loss_epoch") or _extract_metric(sidecar, summary, "validate/loss"),
            "val_pred_loss": _extract_metric(sidecar, summary, "validate/pred_loss_epoch") or _extract_metric(sidecar, summary, "validate/pred_loss"),
            "val_sigreg_loss": _extract_metric(sidecar, summary, "validate/sigreg_loss_epoch") or _extract_metric(sidecar, summary, "validate/sigreg_loss"),
            "lr": _extract_metric(sidecar, summary, "hparams/lr_default_0"),
            "checkpoint": checkpoint,
            "hparams": {
                "embed_dim": hparams.get("embed_dim"),
                "history_size": hparams.get("history_size"),
                "num_preds": hparams.get("num_preds"),
                "max_epochs": hparams.get("trainer.max_epochs"),
                "dataset": hparams.get("data.path"),
                "model_target": hparams.get("model._target_"),
                "window": hparams.get("data.window"),
                "batch_size": hparams.get("data.batch_size"),
            },
            "series": _series_from_csv(run_dir / "metrics.csv"),
        }
        runs.append(run)
    runs.sort(key=lambda r: r["updated_at"], reverse=True)
    return runs


def _is_od_run(run: dict[str, Any]) -> bool:
    hparams = run.get("hparams", {})
    model_target = hparams.get("model_target")
    return (
        hparams.get("embed_dim") is not None
        and hparams.get("dataset") == "data/cache/od_trajectories.npz"
        and (model_target is None or model_target == "models.od_jepa.ODJEPA")
    )


def _earth_surface(radius_km: float = 6378.137) -> dict[str, list[list[float]]]:
    lat = np.linspace(-math.pi / 2, math.pi / 2, 25)
    lon = np.linspace(0, 2 * math.pi, 49)
    x = radius_km * np.outer(np.cos(lat), np.cos(lon))
    y = radius_km * np.outer(np.cos(lat), np.sin(lon))
    z = radius_km * np.outer(np.sin(lat), np.ones_like(lon))
    return {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}


def _station_marker(lat_deg: float = 48.15, lon_deg: float = 11.58, alt_m: float = 500.0) -> dict[str, float]:
    r = 6378.137 + alt_m / 1000.0
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return {
        "x": r * math.cos(lat) * math.cos(lon),
        "y": r * math.cos(lat) * math.sin(lon),
        "z": r * math.sin(lat),
        "label": f"{lat_deg:.2f}N, {lon_deg:.2f}E",
    }


def _dataset_payload() -> dict[str, Any]:
    if not DATASET.exists():
        return {"ok": False, "message": f"Dataset not found: {DATASET}"}

    blob = np.load(DATASET)
    obs = blob["obs"]
    actions = blob["action"]
    state = blob["state"]
    dt_s = 30.0
    ep_a = 0
    ep_b = 1 if state.shape[0] > 1 else 0
    pos_a = state[ep_a, :, :3] / 1000.0
    pos_b = state[ep_b, :, :3] / 1000.0
    vel_a = state[ep_a, :, 3:]
    radius_km = np.linalg.norm(pos_a, axis=1)
    speed = np.linalg.norm(vel_a, axis=1)
    elevation_deg = np.degrees(obs[ep_a, :, 2])
    t_min = np.arange(obs.shape[1]) * dt_s / 60.0

    return {
        "ok": True,
        "path": str(DATASET.relative_to(ROOT)),
        "episodes": int(obs.shape[0]),
        "steps": int(obs.shape[1]),
        "dt_s": dt_s,
        "duration_h": float((obs.shape[1] - 1) * dt_s / 3600.0),
        "obs_dim": int(obs.shape[2]),
        "action_dim": int(actions.shape[2]),
        "state_dim": int(state.shape[2]),
        "sample_episode": ep_a,
        "comparison_episode": ep_b,
        "time_min": t_min.tolist(),
        "orbit_a": {"x": pos_a[:, 0].tolist(), "y": pos_a[:, 1].tolist(), "z": pos_a[:, 2].tolist()},
        "orbit_b": {"x": pos_b[:, 0].tolist(), "y": pos_b[:, 1].tolist(), "z": pos_b[:, 2].tolist()},
        "earth": _earth_surface(),
        "station": _station_marker(),
        "obs": {
            "range_km": (obs[ep_a, :, 0] / 1000.0).tolist(),
            "az_deg": np.degrees(obs[ep_a, :, 1]).tolist(),
            "el_deg": elevation_deg.tolist(),
            "range_rate_km_s": (obs[ep_a, :, 3] / 1000.0).tolist(),
        },
        "stats": {
            "radius_mean_km": float(radius_km.mean()),
            "radius_span_km": float(radius_km.max() - radius_km.min()),
            "speed_mean_km_s": float(speed.mean() / 1000.0),
            "speed_span_m_s": float(speed.max() - speed.min()),
            "visible_fraction": float(np.mean(elevation_deg > 0.0)),
            "range_min_km": float(obs[ep_a, :, 0].min() / 1000.0),
            "range_max_km": float(obs[ep_a, :, 0].max() / 1000.0),
        },
    }


def _latest_checkpoint_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for run in runs:
        ckpt = run.get("checkpoint")
        if ckpt and Path(ckpt).exists() and _is_od_run(run):
            return run
    return None


def _strip_model_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
    return out or state_dict


def _pca_3d(target: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    both = np.concatenate([target, pred], axis=0)
    center = both.mean(axis=0, keepdims=True)
    centered = both - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:3].T
    denom = np.square(singular).sum()
    explained = (np.square(singular[:3]) / denom).tolist() if denom > 0 else [0.0, 0.0, 0.0]
    n = target.shape[0]
    return coords[:n], coords[n:], [float(x) for x in explained]


def _model_probe(runs: list[dict[str, Any]], dataset: dict[str, Any]) -> dict[str, Any]:
    run = _latest_checkpoint_run(runs)
    if run is None:
        return {"ok": False, "message": "No readable checkpoint found under stable-pretraining runs."}
    if not dataset.get("ok"):
        return {"ok": False, "message": "Dataset unavailable; checkpoint probe skipped."}

    try:
        import torch
        import hydra
        from omegaconf import OmegaConf
    except Exception as exc:
        return {"ok": False, "message": f"Probe dependencies unavailable: {exc}"}

    try:
        run_dir = Path(run["run_dir"])
        cfg = OmegaConf.load(run_dir / "hparams.yaml")
        model = hydra.utils.instantiate(cfg.model)
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        state_dict = _strip_model_prefix(checkpoint.get("state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        blob = np.load(DATASET)
        obs = blob["obs"]
        actions = blob["action"]
        ep = int(dataset["sample_episode"])
        history = int(cfg.history_size)
        num_preds = int(cfg.num_preds)
        window = int(cfg.data.window)

        obs_mean = obs.reshape(-1, obs.shape[-1]).mean(axis=0)
        obs_std = obs.reshape(-1, obs.shape[-1]).std(axis=0)
        obs_std[obs_std < 1e-8] = 1.0
        act_mean = actions.reshape(-1, actions.shape[-1]).mean(axis=0)
        act_std = actions.reshape(-1, actions.shape[-1]).std(axis=0)
        act_std[act_std < 1e-8] = 1.0

        obs_norm = (obs[ep] - obs_mean) / obs_std
        act_norm = (actions[ep] - act_mean) / act_std
        starts = np.arange(0, obs_norm.shape[0] - window + 1)
        obs_windows = np.stack([obs_norm[s : s + window] for s in starts])
        act_windows = np.stack([act_norm[s : s + window] for s in starts])

        with torch.no_grad():
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
            pred_last = pred[:, m - 1].detach().cpu().numpy()
            target_last = target[:, m - 1].detach().cpu().numpy()
            persistence = emb[:, history - 1].detach().cpu().numpy()

        mse = np.mean(np.square(pred_last - target_last), axis=1)
        persistence_mse = np.mean(np.square(persistence - target_last), axis=1)
        dot = np.sum(pred_last * target_last, axis=1)
        denom = np.linalg.norm(pred_last, axis=1) * np.linalg.norm(target_last, axis=1)
        cosine = dot / np.maximum(denom, 1e-8)
        target_pca, pred_pca, explained = _pca_3d(target_last, pred_last)
        t_min = (starts + num_preds + m - 1) * float(dataset["dt_s"]) / 60.0

        return {
            "ok": True,
            "run_id": run["run_id"],
            "run_name": run["name"],
            "checkpoint": run["checkpoint"],
            "sample_episode": ep,
            "history_size": history,
            "num_preds": num_preds,
            "window": window,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "time_min": t_min.tolist(),
            "mse": mse.tolist(),
            "persistence_mse": persistence_mse.tolist(),
            "cosine": cosine.tolist(),
            "latent_target": {
                "x": target_pca[:, 0].tolist(),
                "y": target_pca[:, 1].tolist(),
                "z": target_pca[:, 2].tolist(),
            },
            "latent_pred": {
                "x": pred_pca[:, 0].tolist(),
                "y": pred_pca[:, 1].tolist(),
                "z": pred_pca[:, 2].tolist(),
            },
            "stats": {
                "mse_mean": float(mse.mean()),
                "mse_median": float(np.median(mse)),
                "persistence_mse_mean": float(persistence_mse.mean()),
                "cosine_mean": float(cosine.mean()),
                "cosine_min": float(cosine.min()),
                "pca_explained": explained,
            },
        }
    except Exception as exc:
        return {"ok": False, "message": f"Checkpoint probe failed: {exc}"}


def _run_table(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        rows.append(
            {
                "name": run["name"],
                "run_id": run["run_id"],
                "status": run["status"],
                "updated_at_iso": run["updated_at_iso"],
                "epoch": run["epoch"],
                "fit_pred_loss": run["fit_pred_loss"],
                "val_pred_loss": run["val_pred_loss"],
                "fit_loss": run["fit_loss"],
                "val_loss": run["val_loss"],
                "checkpoint": bool(run.get("checkpoint") and Path(run["checkpoint"]).exists()),
                "embed_dim": run["hparams"].get("embed_dim"),
                "history_size": run["hparams"].get("history_size"),
                "max_epochs": run["hparams"].get("max_epochs"),
            }
        )
    return rows


def _compact_benchmark_method(method: dict[str, Any]) -> dict[str, Any]:
    compact = dict(method)
    cov = dict(method.get("residual_covariance") or {})
    cov.pop("matrix", None)
    diag = cov.get("diag")
    if isinstance(diag, list) and len(diag) > 12:
        cov["diag_preview"] = diag[:12]
        cov["diag_dim"] = len(diag)
        cov.pop("diag", None)
    compact["residual_covariance"] = cov
    return compact


def _benchmark_payload() -> dict[str, Any]:
    if not BENCHMARK.exists():
        return {
            "ok": False,
            "message": (
                f"Benchmark artifact not found: {BENCHMARK.relative_to(ROOT)}. "
                "Run uv run python scripts/benchmark_od_methods.py to generate it."
            ),
        }
    payload = _read_json(BENCHMARK)
    if not payload:
        return {"ok": False, "message": f"Could not parse {BENCHMARK.relative_to(ROOT)}."}
    payload = dict(payload)
    payload["ok"] = True
    payload["path"] = str(BENCHMARK.relative_to(ROOT))
    payload["methods"] = [_compact_benchmark_method(m) for m in payload.get("methods", [])]
    return payload



def _resolve_artifact_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _geometry_decoder_payload() -> dict[str, Any]:
    metrics_path = next((path for path in GEOMETRY_DECODER_METRICS if path.exists()), None)
    if metrics_path is None:
        return {
            "ok": False,
            "message": "No geometry decoder metrics found. Run scripts/train_od_geometry_decoder.py first.",
        }
    metrics = _read_json(metrics_path)
    if not metrics:
        return {"ok": False, "message": f"Could not parse {metrics_path.relative_to(ROOT)}."}

    stats = dict(metrics.get("raw_geometry_decode") or {})
    cov = stats.get("residual_covariance") or {}
    artifact_path = _resolve_artifact_path(metrics.get("artifact"))
    dataset_path = _resolve_artifact_path(metrics.get("dataset"))
    base = {
        "ok": False,
        "path": str(metrics_path.relative_to(ROOT)),
        "artifact": str(artifact_path.relative_to(ROOT)) if artifact_path and artifact_path.is_relative_to(ROOT) else str(artifact_path),
        "dataset": str(dataset_path.relative_to(ROOT)) if dataset_path and dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "mode": metrics.get("mode"),
        "eval_episode": metrics.get("eval_episode"),
        "window": metrics.get("window"),
        "samples": metrics.get("samples"),
        "stats": {
            "position_rmse_m": stats.get("position_rmse_m"),
            "position_median_m": stats.get("position_median_m"),
            "position_p95_m": stats.get("position_p95_m"),
            "position_max_m": stats.get("position_max_m"),
            "velocity_rmse_m_s": stats.get("velocity_rmse_m_s"),
            "velocity_median_m_s": stats.get("velocity_median_m_s"),
            "velocity_p95_m_s": stats.get("velocity_p95_m_s"),
            "velocity_max_m_s": stats.get("velocity_max_m_s"),
        },
        "uncertainty": {},
    }
    if artifact_path is None or dataset_path is None or not artifact_path.exists() or not dataset_path.exists():
        base["message"] = "Geometry decoder artifact or dataset is missing; scalar metrics only."
        return base

    try:
        import torch
        from scripts.train_od_geometry_decoder import GeometryDecoder, _build_raw_features, _predict

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        dataset_blob = np.load(dataset_path)
        dataset = {key: dataset_blob[key] for key in dataset_blob.files}
        mode = artifact.get("mode", metrics.get("mode", "raw"))
        window = int(artifact.get("window", metrics.get("window", 8)))
        eval_episode = int(artifact.get("eval_episode", metrics.get("eval_episode", 0)))
        features = _build_raw_features(dataset, window, mode=mode)
        eval_mask = features.episode == eval_episode
        if not np.any(eval_mask):
            raise ValueError(f"eval episode {eval_episode} not present in geometry features")

        decoder = GeometryDecoder(
            int(artifact["input_dim"]),
            int(artifact["hidden_dim"]),
            int(artifact["depth"]),
            int(artifact.get("output_dim", 6)),
        )
        decoder.load_state_dict(artifact["state_dict"])
        decoder.eval()
        to_np = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        pred = _predict(
            decoder,
            features.x[eval_mask],
            to_np(artifact["feature_mean"]),
            to_np(artifact["feature_std"]),
            to_np(artifact["state_mean"]),
            to_np(artifact["state_std"]),
            batch_size=4096,
        )
        truth = features.y[eval_mask]
        residual = pred - truth
        pos_err = np.linalg.norm(residual[:, :3], axis=1)
        vel_err = np.linalg.norm(residual[:, 3:], axis=1)
        residual_cov = np.cov(residual, rowvar=False) if residual.shape[0] > 1 else np.zeros((6, 6))
        pos_sigma_m = float(np.sqrt(max(0.0, np.trace(residual_cov[:3, :3]))))
        vel_sigma_m_s = float(np.sqrt(max(0.0, np.trace(residual_cov[3:, 3:]))))

        base.update(
            {
                "ok": True,
                "message": "Geometry decoder trajectory loaded from artifact.",
                "mode": mode,
                "eval_episode": eval_episode,
                "window": window,
                "samples": int(truth.shape[0]),
                "series": {
                    "time_min": features.time_min[eval_mask].tolist(),
                    "truth": {
                        "x": (truth[:, 0] / 1000.0).tolist(),
                        "y": (truth[:, 1] / 1000.0).tolist(),
                        "z": (truth[:, 2] / 1000.0).tolist(),
                    },
                    "decoded": {
                        "x": (pred[:, 0] / 1000.0).tolist(),
                        "y": (pred[:, 1] / 1000.0).tolist(),
                        "z": (pred[:, 2] / 1000.0).tolist(),
                    },
                    "position_error_m": pos_err.tolist(),
                    "velocity_error_m_s": vel_err.tolist(),
                },
                "uncertainty": {
                    "position_sigma_m": pos_sigma_m,
                    "position_2sigma_m": 2.0 * pos_sigma_m,
                    "position_p95_m": float(np.percentile(pos_err, 95)),
                    "velocity_sigma_m_s": vel_sigma_m_s,
                    "velocity_2sigma_m_s": 2.0 * vel_sigma_m_s,
                    "velocity_p95_m_s": float(np.percentile(vel_err, 95)),
                    "covariance_space": cov.get("space", "ECI state residual"),
                },
            }
        )
        return base
    except Exception as exc:
        base["message"] = f"Geometry decoder trajectory reconstruction failed: {exc}"
        return base

def build_payload() -> dict[str, Any]:
    runs = _find_runs()
    latest = next((r for r in runs if _is_od_run(r) and r["series"].get("fit/pred_loss", {}).get("y")), None)
    if latest is None:
        latest = next((r for r in runs if _is_od_run(r)), runs[0] if runs else None)
    dataset = _dataset_payload()
    probe = _model_probe(runs, dataset)
    benchmark = _benchmark_payload()
    geometry_decoder = _geometry_decoder_payload()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "runs": runs,
        "run_table": _run_table(runs),
        "latest_run_id": latest["run_id"] if latest else None,
        "dataset": dataset,
        "probe": probe,
        "benchmark": benchmark,
        "geometry_decoder": geometry_decoder,
        "formatted": {
            "latest_val_pred": _fmt_metric(latest["val_pred_loss"] if latest else None),
            "latest_fit_pred": _fmt_metric(latest["fit_pred_loss"] if latest else None),
            "latest_val_loss": _fmt_metric(latest["val_loss"] if latest else None),
            "probe_mse": _fmt_metric(probe.get("stats", {}).get("mse_mean") if probe.get("ok") else None),
        },
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT.write_text(TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)), encoding="utf-8")
    runs = len(payload["runs"])
    probe = "with checkpoint probe" if payload["probe"].get("ok") else "without checkpoint probe"
    print(f"wrote {OUT.relative_to(ROOT)} from {runs} run(s), {probe}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Space World Models - OD LeWM Board</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {
  --ink: #172026;
  --muted: #5d6872;
  --line: #d8e0e6;
  --panel: #f7f9fb;
  --blue: #1264a3;
  --green: #278a63;
  --gold: #a97000;
  --red: #b43d3d;
  --violet: #6b5aa8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: #ffffff;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
header {
  padding: 24px 44px 18px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #fbfcfd 0%, #ffffff 100%);
}
h1 { margin: 0; font-size: 25px; line-height: 1.15; letter-spacing: 0; color: var(--ink); }
.sub { margin-top: 8px; max-width: 1120px; color: var(--muted); font-size: 13.5px; }
main { padding: 0 44px 34px; }
section { max-width: 1380px; padding: 24px 0 4px; border-bottom: 1px solid #edf1f4; }
section:last-child { border-bottom: 0; }
h2 { margin: 0 0 5px; color: #18364e; font-size: 17px; letter-spacing: 0; }
.caption { margin: 0 0 14px; color: var(--muted); font-size: 12.5px; max-width: 1080px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 18px; }
.kpi {
  min-height: 82px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 14px;
  background: #fff;
}
.kpi b { display: block; font-size: 22px; line-height: 1.1; color: var(--blue); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.kpi span { display: block; margin-top: 7px; font-size: 12px; color: var(--muted); }
.grid2 { display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(360px, 0.82fr); gap: 24px; align-items: start; }
.gridEven { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; align-items: start; }
.plot { width: 100%; height: 430px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
.plot.tall { height: 520px; }
.plot.short { height: 340px; }
.tableWrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
table { border-collapse: collapse; width: 100%; min-width: 780px; font-size: 12.5px; }
th {
  text-align: left;
  padding: 9px 10px;
  color: #314354;
  background: #f3f6f8;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
td { padding: 8px 10px; border-bottom: 1px solid #edf1f4; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill {
  display: inline-block;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 11px;
  color: var(--muted);
  white-space: nowrap;
}
.pill.ok { color: var(--green); border-color: rgba(39, 138, 99, 0.45); background: rgba(39, 138, 99, 0.08); }
.pill.warn { color: var(--gold); border-color: rgba(169, 112, 0, 0.42); background: rgba(169, 112, 0, 0.08); }
.process {
  display: grid;
  grid-template-columns: repeat(6, minmax(130px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.step {
  position: relative;
  min-height: 128px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px;
  background: #fff;
}
.step:not(:last-child)::after {
  content: ">";
  position: absolute;
  top: 50%;
  right: -10px;
  width: 18px;
  transform: translateY(-50%);
  color: var(--blue);
  font-weight: 700;
  text-align: center;
  background: #fff;
}
.step b { display: block; color: #17344d; font-size: 13px; margin-bottom: 6px; }
.step span { display: block; color: var(--muted); font-size: 12px; }
.note {
  border-left: 3px solid var(--blue);
  background: var(--panel);
  padding: 10px 13px;
  font-size: 12.5px;
  color: #334;
}
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }
@media (max-width: 1100px) {
  header, main { padding-left: 22px; padding-right: 22px; }
  .grid2, .gridEven { grid-template-columns: 1fr; }
  .process { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .step:not(:last-child)::after { display: none; }
}
@media (max-width: 650px) {
  header, main { padding-left: 14px; padding-right: 14px; }
  h1 { font-size: 21px; }
  .plot, .plot.tall { height: 360px; }
  .process { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1>OD LeWM Training Board</h1>
  <div class="sub">A local instrument for orbit-determination world-model training: Orekit trajectory samples, ground-station observation streams, and latent next-step predictions from the newest readable checkpoint. Training curves live in WandB.</div>
  <div class="kpis" id="kpis"></div>
</header>
<main>
  <section>
    <h2>Orbit And Measurements</h2>
    <p class="caption">Blue is dataset episode 0, the same episode used for the observation plot and LeWM prediction probe. Gold is dataset episode 1, shown only as a comparison trajectory from the 192 generated episodes. The generated dataset stores hidden ECI state for diagnostics, while the model receives four ground-station measurements: range, azimuth, elevation, and range-rate.</p>
    <div class="gridEven">
      <div id="orbitPlot" class="plot tall"></div>
      <div id="obsPlot" class="plot tall"></div>
    </div>
  </section>

  <section>
    <h2>What The LeWM Predicts</h2>
    <p class="caption">This uses dataset episode 0. OD-JEPA itself still predicts in latent space, so this rollout is shown in latent PCA coordinates; the comparison section below uses a separate supervised probe decoder for ECI state. The target trace is the encoder's embedding of the next observation; the predicted trace is the autoregressive predictor output conditioned on the history window and action embedding.</p>
    <div class="gridEven">
      <div id="latentPlot" class="plot tall"></div>
      <div id="errorPlot" class="plot tall"></div>
    </div>
    <div style="height:14px"></div>
    <div class="note" id="probeNote"></div>
  </section>


  <section>
    <h2>Geometry Decoder Orbit</h2>
    <p class="caption">Tuned raw-ECI geometry decoder on held-out episode 0. The orbit plot overlays truth and decoded ECI position; decoded markers are colored by instantaneous position error, and red links show sampled truth-to-decode residual vectors. The error plot shows empirical residual uncertainty bands from the decoder artifact.</p>
    <div class="gridEven">
      <div id="geometryOrbitPlot" class="plot tall"></div>
      <div id="geometryErrorPlot" class="plot tall"></div>
    </div>
    <div style="height:14px"></div>
    <div class="note" id="geometryNote"></div>
  </section>

  <section>
    <h2>Algorithmic vs LeWM Comparison</h2>
    <p class="caption">Initial benchmark for episode 0. Orekit is a known-orbit physics replay using the same seed as the dataset, so it is an upper-bound propagation reference. The LeWM latent row measures embedding prediction; the decoded-state row uses the new frozen probe decoder to report ECI position and velocity error.</p>
    <div class="gridEven">
      <div id="benchTimePlot" class="plot short"></div>
      <div id="benchResourcePlot" class="plot short"></div>
    </div>
    <div style="height:14px"></div>
    <div class="gridEven">
      <div id="benchErrorPlot" class="plot short"></div>
      <div id="benchCovPlot" class="plot short"></div>
    </div>
    <div style="height:14px"></div>
    <div class="tableWrap"><table id="benchTable"></table></div>
    <div style="height:14px"></div>
    <div class="note" id="benchNote"></div>
  </section>

  <section>
    <h2>Training Process</h2>
    <p class="caption">A compact map of the current OD LeWM path from simulator data to the loss terms being optimized.</p>
    <div class="process">
      <div class="step"><b>Orekit propagation</b><span>Hidden ECI state [r, v] evolves with the Eckstein-Hechler J2-J6 analytical propagator.</span></div>
      <div class="step"><b>Ground station view</b><span>The learner sees range, azimuth, elevation, and range-rate with configured measurement noise.</span></div>
      <div class="step"><b>Windowed dataset</b><span>Short normalized windows pair observations with zero LVLH acceleration actions.</span></div>
      <div class="step"><b>Encoder</b><span>A vector MLP maps each 4D observation into a 192D LeWM embedding.</span></div>
      <div class="step"><b>AR predictor</b><span>A conditional transformer predicts future embeddings from context and action embeddings.</span></div>
      <div class="step"><b>Losses</b><span>Prediction MSE pulls next-step embeddings together; SIGReg keeps the latent space well spread.</span></div>
    </div>
  </section>
</main>

<script>
const DATA = __PAYLOAD__;

const colors = {
  blue: "#1264a3",
  green: "#278a63",
  gold: "#a97000",
  red: "#b43d3d",
  violet: "#6b5aa8",
  gray: "#6f7a83"
};

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "n/a";
  const v = Number(x);
  if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

function esc(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function layout(title, extra = {}) {
  return Object.assign({
    title: {text: title, font: {size: 14, color: "#18364e"}, x: 0.02},
    margin: {t: 48, r: 18, b: 48, l: 58},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: {family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 12, color: "#172026"},
    xaxis: {gridcolor: "#edf1f4", zerolinecolor: "#d8e0e6"},
    yaxis: {gridcolor: "#edf1f4", zerolinecolor: "#d8e0e6"},
    legend: {orientation: "h", x: 0, y: -0.18}
  }, extra);
}

function renderKpis() {
  const latest = DATA.runs.find(r => r.run_id === DATA.latest_run_id) || DATA.runs[0];
  const ds = DATA.dataset || {};
  const probe = DATA.probe || {};
  const bench = DATA.benchmark || {};
  const methods = bench.methods || [];
  const orekit = methods.find(m => m.id === "orekit_known_orbit") || {};
  const lewm = methods.find(m => m.id === "lewm_latent_predictor") || {};
  const cards = [
    [probe.ok ? probe.run_name : (latest ? latest.name : "No run"), "checkpoint used for LeWM probe"],
    [ds.ok ? `${ds.episodes} x ${ds.steps}` : "n/a", "episodes x steps in dataset"],
    [ds.ok ? `episode ${ds.sample_episode}` : "n/a", "observed and predicted episode"],
    [ds.ok ? `episode ${ds.comparison_episode}` : "n/a", "comparison orbit only"],
    [probe.ok ? fmt(probe.stats.mse_mean) : "n/a", "checkpoint latent MSE"],
    [probe.ok ? fmt(probe.stats.cosine_mean) : "n/a", "mean pred-target cosine"],
    [orekit.time_per_sample_ms !== undefined ? `${fmt(orekit.time_per_sample_ms)} ms` : "n/a", "Orekit propagation time/sample"],
    [lewm.time_per_sample_ms !== undefined ? `${fmt(lewm.time_per_sample_ms)} / ${fmt(lewm.online_time_per_sample_ms)} ms` : "n/a", "LeWM batch / online time/sample"]
  ];
  document.getElementById("kpis").innerHTML = cards.map(([value, label]) =>
    `<div class="kpi"><b>${value}</b><span>${label}</span></div>`
  ).join("");
}

function plotOrbit() {
  const ds = DATA.dataset || {};
  if (!ds.ok) return;
  const earth = {
    type: "surface",
    x: ds.earth.x,
    y: ds.earth.y,
    z: ds.earth.z,
    showscale: false,
    opacity: 0.35,
    colorscale: [[0, "#dcebf5"], [1, "#b8d3e5"]],
    name: "Earth",
    hoverinfo: "skip"
  };
  const orbitA = {
    type: "scatter3d",
    mode: "lines",
    x: ds.orbit_a.x,
    y: ds.orbit_a.y,
    z: ds.orbit_a.z,
    name: `episode ${ds.sample_episode} (used below)`,
    line: {color: colors.blue, width: 5}
  };
  const orbitB = {
    type: "scatter3d",
    mode: "lines",
    x: ds.orbit_b.x,
    y: ds.orbit_b.y,
    z: ds.orbit_b.z,
    name: `episode ${ds.comparison_episode} (comparison only)`,
    line: {color: colors.gold, width: 4, dash: "dot"}
  };
  const station = {
    type: "scatter3d",
    mode: "markers+text",
    x: [ds.station.x],
    y: [ds.station.y],
    z: [ds.station.z],
    text: ["station"],
    textposition: "top center",
    name: "ground station",
    marker: {size: 4, color: colors.red}
  };
  Plotly.newPlot("orbitPlot", [earth, orbitA, orbitB, station], layout("ECI orbit propagation samples", {
    scene: {
      xaxis: {title: "x km", showbackground: false, gridcolor: "#e8eef2"},
      yaxis: {title: "y km", showbackground: false, gridcolor: "#e8eef2"},
      zaxis: {title: "z km", showbackground: false, gridcolor: "#e8eef2"},
      aspectmode: "data",
      camera: {eye: {x: 1.55, y: 1.45, z: 0.95}}
    },
    margin: {t: 48, r: 0, b: 0, l: 0},
    legend: {orientation: "h", x: 0, y: 0}
  }), {responsive: true, displaylogo: false});
}

function plotObs() {
  const ds = DATA.dataset || {};
  if (!ds.ok) return;
  const t = ds.time_min;
  Plotly.newPlot("obsPlot", [
    {x: t, y: ds.obs.range_km, type: "scatter", mode: "lines", name: "range km", line: {color: colors.blue, width: 2}},
    {x: t, y: ds.obs.el_deg, type: "scatter", mode: "lines", name: "elevation deg", yaxis: "y2", line: {color: colors.green, width: 2}},
    {x: t, y: ds.obs.range_rate_km_s, type: "scatter", mode: "lines", name: "range-rate km/s", yaxis: "y3", line: {color: colors.red, width: 2}}
  ], layout(`Observation stream from episode ${ds.sample_episode}`, {
    xaxis: {title: "minutes", gridcolor: "#edf1f4"},
    yaxis: {title: "range km", gridcolor: "#edf1f4"},
    yaxis2: {title: "elevation deg", overlaying: "y", side: "right", showgrid: false},
    yaxis3: {title: "range-rate", overlaying: "y", side: "right", anchor: "free", position: 0.94, showgrid: false},
    margin: {t: 48, r: 88, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});
}

function plotProbe() {
  const probe = DATA.probe || {};
  if (!probe.ok) {
    document.getElementById("probeNote").innerHTML = probe.message || "No checkpoint probe available.";
    return;
  }
  Plotly.newPlot("latentPlot", [
    {type: "scatter3d", mode: "lines", x: probe.latent_target.x, y: probe.latent_target.y, z: probe.latent_target.z, name: "target latent", line: {color: colors.green, width: 5}},
    {type: "scatter3d", mode: "lines", x: probe.latent_pred.x, y: probe.latent_pred.y, z: probe.latent_pred.z, name: "predicted latent", line: {color: colors.violet, width: 4}}
  ], layout("Predicted vs target latent trajectory", {
    scene: {
      xaxis: {title: "PC1", showbackground: false, gridcolor: "#e8eef2"},
      yaxis: {title: "PC2", showbackground: false, gridcolor: "#e8eef2"},
      zaxis: {title: "PC3", showbackground: false, gridcolor: "#e8eef2"},
      aspectmode: "cube"
    },
    margin: {t: 48, r: 0, b: 0, l: 0}
  }), {responsive: true, displaylogo: false});

  Plotly.newPlot("errorPlot", [
    {x: probe.time_min, y: probe.mse, type: "scatter", mode: "lines", name: "LeWM latent MSE", line: {color: colors.blue, width: 2}},
    {x: probe.time_min, y: probe.persistence_mse, type: "scatter", mode: "lines", name: "persistence MSE", line: {color: colors.gold, width: 2, dash: "dot"}},
    {x: probe.time_min, y: probe.cosine, type: "scatter", mode: "lines", name: "cosine", yaxis: "y2", line: {color: colors.green, width: 2}}
  ], layout("Next-step prediction diagnostics", {
    xaxis: {title: "minutes", gridcolor: "#edf1f4"},
    yaxis: {title: "embedding MSE", type: "log", gridcolor: "#edf1f4"},
    yaxis2: {title: "cosine", overlaying: "y", side: "right", range: [-1, 1], showgrid: false},
    margin: {t: 48, r: 64, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});

  const ev = probe.stats.pca_explained.map(v => `${Math.round(v * 1000) / 10}%`).join(" / ");
  document.getElementById("probeNote").innerHTML =
    `Checkpoint <code>${probe.run_id}</code> loaded from <code>${probe.checkpoint}</code>. ` +
    `Probe dataset episode=${probe.sample_episode}; history=${probe.history_size}, prediction offset=${probe.num_preds}, window=${probe.window}. ` +
    `Mean latent MSE=${fmt(probe.stats.mse_mean)}, persistence baseline=${fmt(probe.stats.persistence_mse_mean)}, mean cosine=${fmt(probe.stats.cosine_mean)}. ` +
    `PCA variance shown by PC1/PC2/PC3: ${ev}. Missing keys=${probe.missing_keys}, unexpected keys=${probe.unexpected_keys}.`;
}

function accuracyText(method) {
  const obs = method.obs_rmse || {};
  if (method.position_rmse_m !== null && method.position_rmse_m !== undefined) {
    return [
      `position RMSE ${fmt(method.position_rmse_m, 3)} m`,
      `velocity RMSE ${fmt(method.velocity_rmse_m_s, 3)} m/s`,
      `range RMSE ${fmt(obs.range_m, 3)} m`
    ].join("<br>");
  }
  if (method.latent_mse_mean !== null && method.latent_mse_mean !== undefined) {
    return [
      `latent MSE mean ${fmt(method.latent_mse_mean, 4)}`,
      `latent MSE median ${fmt(method.latent_mse_median, 4)}`,
      `latent cosine ${fmt(method.latent_cosine_mean, 4)}`
    ].join("<br>");
  }
  return "n/a";
}


function renderGeometryDecoder() {
  const geom = DATA.geometry_decoder || {};
  const note = document.getElementById("geometryNote");
  if (!geom.ok || !geom.series) {
    if (note) note.innerHTML = esc(geom.message || "No geometry decoder trajectory available.");
    return;
  }
  const s = geom.series;
  const unc = geom.uncertainty || {};
  const stats = geom.stats || {};
  const t = s.time_min || [];
  const posErr = s.position_error_m || [];
  const posErrKm = posErr.map(v => Number(v) / 1000.0);
  const velErr = s.velocity_error_m_s || [];
  const traces = [];
  const ds = DATA.dataset || {};
  if (ds.ok && ds.earth) {
    traces.push({type: "surface", x: ds.earth.x, y: ds.earth.y, z: ds.earth.z, showscale: false, opacity: 0.22, colorscale: [[0, "#dcebf5"], [1, "#b8d3e5"]], name: "Earth", hoverinfo: "skip"});
  }
  traces.push({type: "scatter3d", mode: "lines", x: s.truth.x, y: s.truth.y, z: s.truth.z, name: "truth ECI", line: {color: colors.blue, width: 5}});
  traces.push({
    type: "scatter3d",
    mode: "lines+markers",
    x: s.decoded.x,
    y: s.decoded.y,
    z: s.decoded.z,
    name: "decoded ECI",
    line: {color: colors.green, width: 4},
    marker: {size: 3, color: posErrKm, colorscale: "YlOrRd", showscale: true, colorbar: {title: "error km"}}
  });
  const linkX = [], linkY = [], linkZ = [];
  const stride = Math.max(1, Math.floor((s.truth.x || []).length / 28));
  for (let i = 0; i < (s.truth.x || []).length; i += stride) {
    linkX.push(s.truth.x[i], s.decoded.x[i], null);
    linkY.push(s.truth.y[i], s.decoded.y[i], null);
    linkZ.push(s.truth.z[i], s.decoded.z[i], null);
  }
  traces.push({type: "scatter3d", mode: "lines", x: linkX, y: linkY, z: linkZ, name: "sampled residual vectors", line: {color: "rgba(180,61,61,0.42)", width: 2}, hoverinfo: "skip"});
  Plotly.newPlot("geometryOrbitPlot", traces, layout("Decoded orbit with sampled residuals", {
    scene: {
      xaxis: {title: "x km", showbackground: false, gridcolor: "#e8eef2"},
      yaxis: {title: "y km", showbackground: false, gridcolor: "#e8eef2"},
      zaxis: {title: "z km", showbackground: false, gridcolor: "#e8eef2"},
      aspectmode: "data",
      camera: {eye: {x: 1.55, y: 1.45, z: 0.95}}
    },
    margin: {t: 48, r: 0, b: 0, l: 0},
    legend: {orientation: "h", x: 0, y: 0}
  }), {responsive: true, displaylogo: false});

  const floor = values => values.map(v => Math.max(Number(v) || 0, 1e-6));
  const fillBand = (label, yMax, color) => ({
    x: t.concat([...t].reverse()),
    y: Array(t.length).fill(Math.max(Number(yMax) || 0, 1e-6)).concat(Array(t.length).fill(1e-6)),
    type: "scatter",
    mode: "lines",
    fill: "toself",
    fillcolor: color,
    line: {color: "rgba(0,0,0,0)"},
    name: label,
    hoverinfo: "skip"
  });
  const pSigma = Number(unc.position_sigma_m) || 0;
  const p2Sigma = Number(unc.position_2sigma_m) || 0;
  const p95 = Number(unc.position_p95_m) || Number(stats.position_p95_m) || 0;
  Plotly.newPlot("geometryErrorPlot", [
    fillBand("2 sigma position band", p2Sigma, "rgba(180,61,61,0.10)"),
    fillBand("1 sigma position band", pSigma, "rgba(39,138,99,0.12)"),
    {x: t, y: floor(posErr), type: "scatter", mode: "lines", name: "position error m", line: {color: colors.red, width: 2}},
    {x: t, y: Array(t.length).fill(Math.max(p95, 1e-6)), type: "scatter", mode: "lines", name: "position P95", line: {color: colors.gold, width: 2, dash: "dash"}},
    {x: t, y: floor(velErr), type: "scatter", mode: "lines", name: "velocity error m/s", yaxis: "y2", line: {color: colors.violet, width: 2}}
  ], layout("Decoded-state error and empirical uncertainty", {
    xaxis: {title: "minutes", gridcolor: "#edf1f4"},
    yaxis: {title: "position error m", type: "log", gridcolor: "#edf1f4"},
    yaxis2: {title: "velocity error m/s", type: "log", overlaying: "y", side: "right", showgrid: false},
    margin: {t: 48, r: 78, b: 48, l: 70}
  }), {responsive: true, displaylogo: false});

  note.innerHTML =
    `Geometry decoder <code>${esc(geom.mode)}</code> from <code>${esc(geom.artifact)}</code>, ` +
    `dataset <code>${esc(geom.dataset)}</code>, eval episode ${esc(geom.eval_episode)}, window ${esc(geom.window)}, samples ${esc(geom.samples)}.<br>` +
    `Position RMSE ${fmt(stats.position_rmse_m, 4)} m, median ${fmt(stats.position_median_m, 4)} m, P95 ${fmt(stats.position_p95_m, 4)} m. ` +
    `Velocity RMSE ${fmt(stats.velocity_rmse_m_s, 4)} m/s. ` +
    `Empirical residual sigma: position ${fmt(unc.position_sigma_m, 4)} m, velocity ${fmt(unc.velocity_sigma_m_s, 4)} m/s. ` +
    `Bands are empirical residual summaries, not calibrated estimator covariance.`;
}

function renderBenchmark() {
  const bench = DATA.benchmark || {};
  const table = document.getElementById("benchTable");
  const note = document.getElementById("benchNote");
  const methods = bench.methods || [];
  if (!bench.ok || methods.length === 0) {
    const msg = bench.message || "No benchmark data available.";
    table.innerHTML = `<tbody><tr><td>${esc(msg)}</td></tr></tbody>`;
    note.innerHTML = esc(msg);
    return;
  }

  const methodLabels = methods.map(m => m.label || m.id);
  const timeLabels = [];
  const timeValues = [];
  const timeColors = [];
  methods.forEach(method => {
    if (method.time_per_sample_ms !== null && method.time_per_sample_ms !== undefined) {
      timeLabels.push(method.id === "lewm_latent_predictor" ? "LeWM batch" : method.label);
      timeValues.push(method.time_per_sample_ms);
      timeColors.push(method.id === "lewm_latent_predictor" ? colors.violet : colors.blue);
    }
    if (method.online_time_per_sample_ms !== null && method.online_time_per_sample_ms !== undefined) {
      timeLabels.push("LeWM online");
      timeValues.push(method.online_time_per_sample_ms);
      timeColors.push(colors.gold);
    }
  });

  Plotly.newPlot("benchTimePlot", [{
    x: timeLabels,
    y: timeValues,
    type: "bar",
    text: timeValues.map(v => fmt(v, 3)),
    textposition: "auto",
    marker: {color: timeColors}
  }], layout("Compute time per sample", {
    xaxis: {tickangle: -18, gridcolor: "#edf1f4"},
    yaxis: {title: "ms / sample", gridcolor: "#edf1f4"},
    margin: {t: 48, r: 18, b: 72, l: 66},
    showlegend: false
  }), {responsive: true, displaylogo: false});

  Plotly.newPlot("benchResourcePlot", [
    {x: methodLabels, y: methods.map(m => m.cpu_percent_single_core), type: "bar", name: "CPU %", marker: {color: colors.blue}},
    {x: methodLabels, y: methods.map(m => m.rss_delta_mb), type: "bar", name: "RSS delta MB", yaxis: "y2", marker: {color: colors.green}}
  ], layout("CPU and memory footprint", {
    barmode: "group",
    xaxis: {tickangle: -15, gridcolor: "#edf1f4"},
    yaxis: {title: "CPU % of one core", gridcolor: "#edf1f4"},
    yaxis2: {title: "RSS delta MB", overlaying: "y", side: "right", showgrid: false},
    margin: {t: 48, r: 68, b: 82, l: 66}
  }), {responsive: true, displaylogo: false});

  const traces = [];
  const floor = values => values.map(v => Math.max(Number(v) || 0, 1e-12));
  const orekit = methods.find(m => m.id === "orekit_known_orbit");
  const lewm = methods.find(m => m.id === "lewm_latent_predictor");
  const decoded = methods.find(m => m.id === "lewm_decoded_state");
  if (orekit && orekit.series && orekit.series.position_error_m) {
    traces.push({
      x: orekit.series.time_min,
      y: floor(orekit.series.position_error_m),
      type: "scatter",
      mode: "lines",
      name: "Orekit position error m",
      line: {color: colors.blue, width: 2}
    });
  }
  if (decoded && decoded.series && decoded.series.position_error_m) {
    traces.push({
      x: decoded.series.time_min,
      y: floor(decoded.series.position_error_m),
      type: "scatter",
      mode: "lines",
      name: "LeWM decoded position error m",
      line: {color: colors.green, width: 2}
    });
  }
  if (lewm && lewm.series && lewm.series.latent_mse) {
    traces.push({
      x: lewm.series.time_min,
      y: floor(lewm.series.latent_mse),
      type: "scatter",
      mode: "lines",
      name: "LeWM latent MSE",
      yaxis: "y2",
      line: {color: colors.violet, width: 2}
    });
  }
  Plotly.newPlot("benchErrorPlot", traces, layout("Native-space error over the episode", {
    xaxis: {title: "minutes", gridcolor: "#edf1f4"},
    yaxis: {title: "position error m", type: "log", gridcolor: "#edf1f4"},
    yaxis2: {title: "latent MSE", type: "log", overlaying: "y", side: "right", showgrid: false},
    margin: {t: 48, r: 72, b: 48, l: 66}
  }), {responsive: true, displaylogo: false});

  const covTrace = methods.map(m => (m.residual_covariance || {}).trace);
  Plotly.newPlot("benchCovPlot", [{
    x: methodLabels,
    y: covTrace,
    type: "bar",
    text: covTrace.map(v => fmt(v, 3)),
    textposition: "auto",
    marker: {color: [colors.blue, colors.violet, colors.gold, colors.green]}
  }], layout("Residual covariance trace", {
    xaxis: {tickangle: -15, gridcolor: "#edf1f4"},
    yaxis: {title: "trace in native residual space", gridcolor: "#edf1f4"},
    margin: {t: 48, r: 18, b: 82, l: 72},
    showlegend: false
  }), {responsive: true, displaylogo: false});

  const rows = methods.map(method => {
    const cov = method.residual_covariance || {};
    const online = method.online_time_per_sample_ms !== null && method.online_time_per_sample_ms !== undefined
      ? `<br><span class="pill">online ${fmt(method.online_time_per_sample_ms, 3)} ms</span>`
      : "";
    return `<tr>
      <td><b>${esc(method.label || method.id)}</b><br><span class="pill">${esc(method.kind || "method")}</span></td>
      <td class="num">${fmt(method.samples, 4)}</td>
      <td class="num">${fmt(method.time_per_sample_ms, 3)} ms${online}</td>
      <td class="num">${fmt(method.throughput_samples_s, 4)}</td>
      <td class="num">${fmt(method.cpu_percent_single_core, 4)}</td>
      <td class="num">${fmt(method.rss_delta_mb, 4)} / ${fmt(method.process_max_rss_mb, 4)}</td>
      <td>${accuracyText(method)}</td>
      <td>${esc(cov.space || "n/a")}<br>trace ${fmt(cov.trace, 4)}<br>det ${fmt(cov.determinant, 4)}</td>
    </tr>`;
  }).join("");
  table.innerHTML = `<thead><tr>
    <th>Method</th><th class="num">Samples</th><th class="num">Time/sample</th><th class="num">Samples/s</th>
    <th class="num">CPU %</th><th class="num">RSS delta / max MB</th><th>Accuracy</th><th>Residual covariance</th>
  </tr></thead><tbody>${rows}</tbody>`;

  const caveats = (bench.caveats || []).map(c => esc(c)).join("<br>");
  note.innerHTML = `Benchmark <code>${esc(bench.path || "")}</code>, episode ${esc(bench.episode)}, generated ${esc(bench.generated_at || "unknown")}.<br>${caveats}`;
}

function renderDatasetNote() {
  const ds = DATA.dataset || {};
  if (!ds.ok) return;
  const note = document.createElement("div");
  note.className = "note";
  note.style.marginTop = "14px";
  note.innerHTML =
    `Dataset <code>${ds.path}</code>: ${ds.episodes} episodes, ${ds.steps} steps, ${ds.dt_s}s cadence, ` +
    `${fmt(ds.duration_h, 3)} h per sample trajectory. Sample orbit radius mean ${fmt(ds.stats.radius_mean_km)} km, ` +
    `range ${fmt(ds.stats.range_min_km)}-${fmt(ds.stats.range_max_km)} km, positive elevation fraction ${fmt(ds.stats.visible_fraction * 100, 3)}%.`;
  document.querySelectorAll("section")[0].appendChild(note);
}

renderKpis();
plotOrbit();
plotObs();
plotProbe();
renderGeometryDecoder();
renderBenchmark();
renderDatasetNote();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
