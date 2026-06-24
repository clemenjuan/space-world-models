#!/usr/bin/env python3
"""Package a trained EventSat LeWM checkpoint for AUTOPS evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from swm_eventsat.planning import default_mode_weights
from swm_eventsat.schema import ACTION_NAMES, AUTOPS_STATE_NAMES, load_world_model_dataset


DEFAULT_DATASET = (
    "/home/clemente/autops-agentic-framework/data/world_model/"
    "eventsat_autops_v1/eventsat_world_model_v1.npz"
)
ROOT = Path(__file__).resolve().parents[2]


def _normalizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _load_probe(path: Path) -> dict[str, Any]:
    blob = np.load(path, allow_pickle=True)
    names = [str(v) for v in blob["attribute_names"].tolist()]
    sidecar = path.with_suffix(".json")
    validation = {}
    if sidecar.exists():
        try:
            validation = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            validation = {}
    return {
        "weights_path": str(path.resolve()),
        "W": blob["W"].astype(float).tolist(),
        "b": blob["b"].astype(float).tolist(),
        "attribute_names": names,
        "normalization": {
            "target_mean": blob["target_mean"].astype(float).tolist()
            if "target_mean" in blob
            else [],
            "target_std": blob["target_std"].astype(float).tolist()
            if "target_std" in blob
            else [],
        },
        "validation": validation,
    }


def _copy_checkpoint(src: Path, out_dir: Path) -> Path:
    dst = out_dir / "lewm.ckpt"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def _git_commit(root: str) -> str:
    if not root:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _mode_weight_presets(attribute_names: list[str]) -> dict[str, dict[str, float]]:
    return {
        mode: {
            name: float(weight)
            for name, weight in zip(attribute_names, default_mode_weights(mode, attribute_names))
        }
        for mode in ("science", "safe", "downlink")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", default="outputs/eventsat_autops_lewm/planner_artifact.json")
    parser.add_argument("--autops-root", default="/home/clemente/autops-agentic-framework")
    parser.add_argument("--source-autops-commit", default="")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--orin-planner-latency-ms", type=float, default=0.0)
    args = parser.parse_args()

    dataset = load_world_model_dataset(args.dataset)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    probe_path = Path(args.probe)
    if not probe_path.exists():
        raise FileNotFoundError(f"probe does not exist: {probe_path}")

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_mean, obs_std = _normalizer(dataset.obs)
    action_mean, action_std = _normalizer(dataset.action)
    normalizers_path = out_dir / "normalizers.npz"
    np.savez_compressed(
        normalizers_path,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        action_names=np.asarray(ACTION_NAMES),
    )

    checkpoint_copy = _copy_checkpoint(checkpoint, out_dir)
    probe = _load_probe(probe_path)
    attribute_names = list(probe["attribute_names"])
    probe_input_dim = int(np.asarray(probe["W"]).shape[1])
    if probe_input_dim != int(args.embed_dim):
        raise ValueError(
            "probe input dimension must match LeWM embed_dim for latent planning: "
            f"probe W has {probe_input_dim}, embed_dim is {args.embed_dim}. "
            "Train probes with --latents exported from the checkpoint."
        )
    mode_weight_presets = _mode_weight_presets(attribute_names)
    probe_rmse = probe.get("validation", {}).get("rmse", {})
    probe_validation_error = (
        float(np.mean([float(v) for v in probe_rmse.values()])) if probe_rmse else 0.0
    )
    model_size_mb = (
        checkpoint_copy.stat().st_size + normalizers_path.stat().st_size + probe_path.stat().st_size
    ) / (1024.0 * 1024.0)
    source_autops_commit = args.source_autops_commit or _git_commit(args.autops_root)

    payload: dict[str, Any] = {
        "schema": "eventsat_lewm_planner_artifact_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_root": str(ROOT),
        "worker": {
            "python": str(ROOT / ".venv" / "bin" / "python"),
            "module": "swm_eventsat.experiments.autops_planner_worker",
        },
        "dataset": {
            "path": str(dataset.path.resolve()),
            "schema": dataset.metadata.get("schema", "eventsat_world_model_v1"),
            "dataset_steps": dataset.dataset_steps,
            "obs_shape": list(dataset.obs.shape),
            "action_shape": list(dataset.action.shape),
            "state_shape": list(dataset.state.shape),
            "source_runs": dataset.metadata.get("source_runs", []),
            "state_names": list(AUTOPS_STATE_NAMES),
            "action_names": list(ACTION_NAMES),
        },
        "lewm": {
            "checkpoint": str(checkpoint_copy.resolve()),
            "source_checkpoint": str(checkpoint.resolve()),
            "normalizers": str(normalizers_path.resolve()),
            "obs_normalizer": "normalizers.npz:obs_mean,obs_std",
            "action_normalizer": "normalizers.npz:action_mean,action_std",
            "obs_dim": int(dataset.obs.shape[-1]),
            "action_dim": int(dataset.action.shape[-1]),
            "embed_dim": int(args.embed_dim),
            "history_size": int(args.history_size),
            "source_autops_commit": source_autops_commit,
        },
        "model_config": {
            "encoder_hidden_dim": 256,
            "predictor_depth": 4,
            "predictor_heads": 8,
            "predictor_mlp_dim": 512,
            "predictor_dim_head": 48,
            "dropout": 0.1,
            "emb_dropout": 0.0,
            "projector_hidden_dim": 512,
        },
        "probe": probe,
        "utility": {
            "attribute_names": attribute_names,
            "mode_weight_presets": mode_weight_presets,
        },
        "cem": {
            "horizon": int(args.horizon),
            "samples": int(args.samples),
            "elites": int(args.elites),
            "iterations": int(args.cem_iterations),
            "alpha": float(args.cem_alpha),
        },
        "mode_weight_presets": mode_weight_presets,
        "model_size_mb": float(model_size_mb),
        "peak_memory_mb": 0.0,
        "probe_validation_error": probe_validation_error,
        "train_dataset_steps": float(dataset.dataset_steps),
        "orin_planner_latency_ms": float(args.orin_planner_latency_ms),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
