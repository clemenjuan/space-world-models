"""Benchmark algorithmic OD propagation against the current OD LeWM predictor.

This is intentionally an initial comparison, not a final OD bake-off. The
algorithmic method is a known-orbit Orekit propagation replay using the same
seed convention as the generated dataset. That makes it an upper-bound physics
baseline, not a measurement-only estimator.

Usage:
    uv run python scripts/benchmark_od_methods.py
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/cache/od_trajectories.npz"
RUN_ROOT = Path.home() / ".cache/stable-pretraining/runs"
OUT = ROOT / "data/figures/od_method_benchmark.json"
DECODER = ROOT / "data/figures/od_latent_decoder.pt"


def _rss_mb() -> float:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * resource.getpagesize() / 1024**2
    except Exception:
        return float("nan")


def _max_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. This workspace is Linux, but keep
    # the guard so the script is not surprising elsewhere.
    return value / 1024.0 if value > 10_000 else value / 1024**2


def _timer() -> dict[str, float]:
    return {"wall": time.perf_counter(), "cpu": time.process_time(), "rss": _rss_mb()}


def _elapsed(start: dict[str, float]) -> dict[str, float]:
    wall = time.perf_counter() - start["wall"]
    cpu = time.process_time() - start["cpu"]
    return {
        "wall_time_s": wall,
        "cpu_time_s": cpu,
        "cpu_percent_single_core": (cpu / wall * 100.0) if wall > 0 else None,
        "rss_delta_mb": _rss_mb() - start["rss"],
        "process_max_rss_mb": _max_rss_mb(),
    }


def _wrap_angle_rad(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _covariance(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 2:
        return {"matrix": [], "trace": None, "determinant": None, "diag": []}
    cov = np.cov(values, rowvar=False)
    return {
        "matrix": cov.tolist(),
        "trace": float(np.trace(cov)),
        "determinant": float(np.linalg.det(cov)),
        "diag": np.diag(cov).tolist(),
    }


def _load_dataset() -> dict[str, np.ndarray]:
    if not DATASET.exists():
        raise FileNotFoundError(f"dataset not found: {DATASET}")
    blob = np.load(DATASET)
    return {key: blob[key] for key in blob.files}


def _latest_checkpoint_run() -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for sidecar_path in RUN_ROOT.glob("*/*/*/sidecar.json"):
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except Exception:
            continue
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
    return candidates[0] if candidates else None


def _algorithmic_known_orbit(dataset: dict[str, np.ndarray], episode: int) -> dict[str, Any]:
    from envs.od_env import OdEnv

    target_obs = dataset["obs"][episode].astype(np.float64)
    target_state = dataset["state"][episode].astype(np.float64)
    steps = target_obs.shape[0]

    # Warm the JVM/Orekit classes before timing the rollout itself.
    _ = OdEnv(max_steps=1)

    start = _timer()
    env = OdEnv(max_steps=steps)
    obs, info = env.reset(seed=episode)
    obs_log = [obs]
    state_log = [info["state"]]
    for _ in range(steps - 1):
        obs, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        obs_log.append(obs)
        state_log.append(info["state"])
    timing = _elapsed(start)

    pred_obs = np.asarray(obs_log, dtype=np.float64)
    pred_state = np.asarray(state_log, dtype=np.float64)
    obs_residual = pred_obs - target_obs
    obs_residual[:, 1] = _wrap_angle_rad(obs_residual[:, 1])
    obs_residual[:, 2] = _wrap_angle_rad(obs_residual[:, 2])
    state_residual = pred_state - target_state
    pos_err = np.linalg.norm(state_residual[:, :3], axis=1)
    vel_err = np.linalg.norm(state_residual[:, 3:], axis=1)

    return {
        "id": "orekit_known_orbit",
        "label": "Orekit known-orbit propagation",
        "kind": "algorithmic_upper_bound",
        "notes": (
            "Replays the same seeded propagator used to generate the dataset. "
            "This is a physics upper bound, not a measurement-only OD estimator."
        ),
        "episode": episode,
        "samples": int(steps),
        **timing,
        "time_per_sample_ms": timing["wall_time_s"] / steps * 1000.0,
        "throughput_samples_s": steps / timing["wall_time_s"] if timing["wall_time_s"] > 0 else None,
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_max_m": float(np.max(pos_err)),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_max_m_s": float(np.max(vel_err)),
        "obs_rmse": {
            "range_m": float(np.sqrt(np.mean(obs_residual[:, 0] ** 2))),
            "az_rad": float(np.sqrt(np.mean(obs_residual[:, 1] ** 2))),
            "el_rad": float(np.sqrt(np.mean(obs_residual[:, 2] ** 2))),
            "range_rate_m_s": float(np.sqrt(np.mean(obs_residual[:, 3] ** 2))),
        },
        "residual_covariance": {
            "space": "observation [range_m, az_rad, el_rad, range_rate_m_s]",
            **_covariance(obs_residual),
        },
        "series": {
            "time_min": (np.arange(steps) * 30.0 / 60.0).tolist(),
            "position_error_m": pos_err.tolist(),
            "velocity_error_m_s": vel_err.tolist(),
            "obs_error_norm": np.linalg.norm(obs_residual, axis=1).tolist(),
        },
    }


def _load_lewm_model() -> tuple[Any, Any, dict[str, Any], dict[str, float]]:
    start = _timer()
    import hydra
    import torch
    torch.backends.nnpack.enabled = False
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    from omegaconf import OmegaConf

    run = _latest_checkpoint_run()
    if run is None:
        raise FileNotFoundError("no LeWM checkpoint with model hparams found")
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
    return model, cfg, run, _elapsed(start)


def _lewm_latent_predictor(dataset: dict[str, np.ndarray], episode: int) -> dict[str, Any]:
    import torch

    model, cfg, run, load_timing = _load_lewm_model()
    obs = dataset["obs"]
    actions = dataset["action"]
    history = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    window = int(cfg.data.window)

    obs_flat = obs.reshape(-1, obs.shape[-1])
    act_flat = actions.reshape(-1, actions.shape[-1])
    obs_mean = obs_flat.mean(axis=0)
    obs_std = obs_flat.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    act_mean = act_flat.mean(axis=0)
    act_std = act_flat.std(axis=0)
    act_std[act_std < 1e-8] = 1.0

    obs_norm = (obs[episode] - obs_mean) / obs_std
    act_norm = (actions[episode] - act_mean) / act_std
    starts = np.arange(0, obs_norm.shape[0] - window + 1)
    obs_windows = np.stack([obs_norm[s : s + window] for s in starts])
    act_windows = np.stack([act_norm[s : s + window] for s in starts])
    batch = {
        "obs": torch.tensor(obs_windows, dtype=torch.float32),
        "action": torch.tensor(act_windows, dtype=torch.float32),
    }

    # Batched throughput: how fast the learned model processes all windows.
    start = _timer()
    with torch.no_grad():
        encoded = model.encode(batch)
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
        pred = model.predict(emb[:, :history], act_emb[:, :history])
        target = emb[:, num_preds:]
        m = min(pred.size(1), target.size(1))
        pred_last = pred[:, m - 1].detach().cpu().numpy()
        target_last = target[:, m - 1].detach().cpu().numpy()
    batch_timing = _elapsed(start)

    # Online-ish latency: same calculation per window. This is closer to a
    # step-by-step onboard loop, though still Python/PyTorch eager execution.
    online_errors = []
    online_start = _timer()
    with torch.no_grad():
        for i in range(len(starts)):
            one = {"obs": batch["obs"][i : i + 1], "action": batch["action"][i : i + 1]}
            out = model.encode(one)
            one_pred = model.predict(out["emb"][:, :history], out["act_emb"][:, :history])
            one_target = out["emb"][:, num_preds:]
            one_m = min(one_pred.size(1), one_target.size(1))
            err = (one_pred[:, one_m - 1] - one_target[:, one_m - 1]).pow(2).mean().item()
            online_errors.append(err)
    online_timing = _elapsed(online_start)

    residual = pred_last - target_last
    mse = np.mean(residual**2, axis=1)
    cosine = np.sum(pred_last * target_last, axis=1) / np.maximum(
        np.linalg.norm(pred_last, axis=1) * np.linalg.norm(target_last, axis=1), 1e-8
    )

    return {
        "id": "lewm_latent_predictor",
        "label": "LeWM latent predictor",
        "kind": "learned_latent_model",
        "notes": (
            "Predicts 192D latent vectors, not ECI state. Accuracy and covariance "
            "are therefore reported in learned latent space until a decoder exists."
        ),
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "episode": episode,
        "samples": int(len(starts)),
        "load": load_timing,
        **batch_timing,
        "time_per_sample_ms": batch_timing["wall_time_s"] / len(starts) * 1000.0,
        "throughput_samples_s": len(starts) / batch_timing["wall_time_s"]
        if batch_timing["wall_time_s"] > 0
        else None,
        "online_wall_time_s": online_timing["wall_time_s"],
        "online_cpu_time_s": online_timing["cpu_time_s"],
        "online_time_per_sample_ms": online_timing["wall_time_s"] / len(starts) * 1000.0,
        "latent_mse_mean": float(np.mean(mse)),
        "latent_mse_median": float(np.median(mse)),
        "latent_mse_max": float(np.max(mse)),
        "latent_cosine_mean": float(np.mean(cosine)),
        "position_rmse_m": None,
        "velocity_rmse_m_s": None,
        "obs_rmse": None,
        "residual_covariance": {
            "space": "192D LeWM latent residual",
            **_covariance(residual),
        },
        "series": {
            "time_min": ((starts + num_preds + m - 1) * 30.0 / 60.0).tolist(),
            "latent_mse": mse.tolist(),
            "online_latent_mse": online_errors,
            "latent_cosine": cosine.tolist(),
        },
    }


def _load_state_decoder() -> tuple[Any, dict[str, Any]]:
    if not DECODER.exists():
        raise FileNotFoundError(f"decoder artifact not found: {DECODER}")
    import torch
    from scripts.train_od_latent_decoder import StateDecoder

    artifact = torch.load(DECODER, map_location="cpu", weights_only=False)
    decoder = StateDecoder(
        int(artifact["input_dim"]),
        int(artifact["hidden_dim"]),
        int(artifact["depth"]),
        int(artifact.get("output_dim", 6)),
    )
    decoder.load_state_dict(artifact["state_dict"])
    decoder.eval()
    return decoder, artifact


def _lewm_decoded_state(dataset: dict[str, np.ndarray], episode: int) -> dict[str, Any]:
    import torch

    model, cfg, run, load_timing = _load_lewm_model()
    decoder, artifact = _load_state_decoder()
    obs = dataset["obs"]
    actions = dataset["action"]
    state = dataset["state"][episode].astype(np.float64)
    history = int(cfg.history_size)
    num_preds = int(cfg.num_preds)
    window = int(cfg.data.window)

    obs_flat = obs.reshape(-1, obs.shape[-1])
    act_flat = actions.reshape(-1, actions.shape[-1])
    obs_mean = obs_flat.mean(axis=0)
    obs_std = obs_flat.std(axis=0)
    obs_std[obs_std < 1e-8] = 1.0
    act_mean = act_flat.mean(axis=0)
    act_std = act_flat.std(axis=0)
    act_std[act_std < 1e-8] = 1.0

    obs_norm = (obs[episode] - obs_mean) / obs_std
    act_norm = (actions[episode] - act_mean) / act_std
    starts = np.arange(0, obs_norm.shape[0] - window + 1)
    obs_windows = np.stack([obs_norm[s : s + window] for s in starts])
    act_windows = np.stack([act_norm[s : s + window] for s in starts])
    batch = {
        "obs": torch.tensor(obs_windows, dtype=torch.float32),
        "action": torch.tensor(act_windows, dtype=torch.float32),
    }

    feature_mean = artifact["feature_mean"].detach().cpu().numpy()
    feature_std = artifact["feature_std"].detach().cpu().numpy()
    state_mean = artifact["state_mean"].detach().cpu().numpy()
    state_std = artifact["state_std"].detach().cpu().numpy()

    start = _timer()
    with torch.no_grad():
        encoded = model.encode(batch)
        emb = encoded["emb"]
        act_emb = encoded["act_emb"]
        pred = model.predict(emb[:, :history], act_emb[:, :history])
        target = emb[:, num_preds:]
        m = min(pred.size(1), target.size(1))
        pred_features = pred[:, :m].reshape(len(starts), -1).detach().cpu().numpy()
        x = (pred_features - feature_mean) / feature_std
        decoded = decoder(torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
        decoded_state = decoded * state_std + state_mean
    timing = _elapsed(start)

    final_offsets = starts + num_preds + m - 1
    target_state = state[final_offsets]
    residual = decoded_state.astype(np.float64) - target_state
    pos_err = np.linalg.norm(residual[:, :3], axis=1)
    vel_err = np.linalg.norm(residual[:, 3:], axis=1)

    return {
        "id": "lewm_decoded_state",
        "label": "LeWM decoded ECI state",
        "kind": "learned_state_decoder_probe",
        "notes": (
            "Frozen LeWM encoder/predictor plus supervised MLP decoder trained "
            "from latent sequences to hidden ECI state. This is a probe decoder, "
            "not yet a jointly trained OD estimator."
        ),
        "run_id": run["run_id"],
        "run_name": run["run_name"],
        "decoder_artifact": str(DECODER.relative_to(ROOT)),
        "episode": episode,
        "samples": int(len(starts)),
        "load": load_timing,
        **timing,
        "time_per_sample_ms": timing["wall_time_s"] / len(starts) * 1000.0,
        "throughput_samples_s": len(starts) / timing["wall_time_s"] if timing["wall_time_s"] > 0 else None,
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_median_m": float(np.median(pos_err)),
        "position_max_m": float(np.max(pos_err)),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_median_m_s": float(np.median(vel_err)),
        "velocity_max_m_s": float(np.max(vel_err)),
        "obs_rmse": None,
        "residual_covariance": {
            "space": "ECI state residual [x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s]",
            **_covariance(residual),
        },
        "series": {
            "time_min": (final_offsets * 30.0 / 60.0).tolist(),
            "position_error_m": pos_err.tolist(),
            "velocity_error_m_s": vel_err.tolist(),
        },
    }


def _run_child(method: str, episode: int) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=f"_{method}.json", delete=False) as fh:
        path = Path(fh.name)
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--method",
                method,
                "--episode",
                str(episode),
                "--out",
                str(path),
            ],
            check=True,
        )
        return json.loads(path.read_text(encoding="utf-8"))
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_method(method: str, episode: int) -> dict[str, Any]:
    dataset = _load_dataset()
    if method == "orekit":
        return _algorithmic_known_orbit(dataset, episode)
    if method == "lewm":
        return _lewm_latent_predictor(dataset, episode)
    if method == "decoded":
        return _lewm_decoded_state(dataset, episode)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["all", "orekit", "lewm", "decoded"], default="all")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.method != "all":
        result = _run_method(args.method, args.episode)
        text = json.dumps(result, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    episode = args.episode
    methods = [_run_child("orekit", episode), _run_child("lewm", episode)]
    if DECODER.exists():
        methods.append(_run_child("decoded", episode))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "episode": episode,
        "caveats": [
            "Orekit known-orbit propagation uses the same seeded initial orbit as the dataset; it is an upper bound, not an OD estimator from measurements.",
            "The LeWM latent predictor reports latent-space accuracy; the decoded-state row uses a supervised probe decoder and is the first state-space learned comparison.",
            "RSS is measured at process level and includes loaded runtime/JVM/PyTorch state; use it as a practical footprint, not a perfectly isolated allocation.",
        ],
        "methods": methods,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(methods)} methods")


if __name__ == "__main__":
    main()
