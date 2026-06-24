"""Artifact manifests for EventSat LeWM planning experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class LeWMArtifact:
    checkpoint: str
    obs_normalizer: str
    action_normalizer: str
    obs_dim: int = 25
    action_dim: int = 7
    embed_dim: int = 128
    history_size: int = 4
    horizon_support: int = 12
    source_autops_commit: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeArtifact:
    W: List[List[float]]
    b: List[float]
    attribute_names: List[str]
    normalization: Dict[str, Any]
    validation: Dict[str, Any]


@dataclass(frozen=True)
class PlannerArtifact:
    lewm: LeWMArtifact
    probe: ProbeArtifact
    cem: Dict[str, Any]
    mode_weight_presets: Dict[str, Dict[str, float]]
    action_masks: Dict[str, Any]
    jetson_export_path: str = ""


def save_artifact(path_like: str | Path, artifact: Any) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(artifact) if hasattr(artifact, "__dataclass_fields__") else artifact
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_artifact(path_like: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))
