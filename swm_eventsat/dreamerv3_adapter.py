"""DreamerV3 baseline manifest helpers for AUTOPS EventSat."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class DreamerV3TrainingSpec:
    env_id: str = "AUTOPS-EventSat-v1"
    observation_dim: int = 25
    action_space: str = "discrete_7_mode"
    simulator: str = "autops-agentic-framework EventSatGymnasium"
    training_steps: int = 1_000_000
    evaluation_episodes: int = 100
    cluster: str = "LRZ AI Systems"
    hardware: str = "4xH100 nodes, parallel sweeps when available"
    notes: Dict[str, Any] = field(default_factory=dict)


def write_training_spec(path_like: str | Path, spec: DreamerV3TrainingSpec) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")


def write_policy_manifest(
    path_like: str | Path,
    *,
    checkpoint: str,
    training_steps: int,
    source_repo_commit: str = "unknown",
    model_size_mb: float = 0.0,
    extra: Dict[str, Any] | None = None,
) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": "DreamerV3",
        "checkpoint": checkpoint,
        "training_steps": int(training_steps),
        "source_repo_commit": source_repo_commit,
        "model_size_mb": float(model_size_mb),
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
