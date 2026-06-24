"""Write minimal AUTOPS-board-compatible result artifacts for LeWM-MPC runs."""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _clean_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _stats(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    cols = {key: [float(row.get(key, 0.0)) for row in rows] for key in keys}
    return {
        "mean": {key: statistics.mean(values) for key, values in cols.items()},
        "std": {key: statistics.pstdev(values) if len(values) > 1 else 0.0 for key, values in cols.items()},
        "min_val": {key: min(values) for key, values in cols.items()},
        "max_val": {key: max(values) for key, values in cols.items()},
    }


def write_minimal_results(
    output_dir: str | Path,
    experiment_id: str,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    num_episodes: int | None = 1,
    episode_metrics: list[dict[str, float]] | None = None,
) -> Path:
    """Write the compact ``results.json`` shape consumed by AUTOPS board refresh."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if episode_metrics is not None:
        rows = [_clean_metrics(row) for row in episode_metrics]
    else:
        clean = _clean_metrics(metrics or {})
        rows = [clean for _ in range(int(num_episodes or 1))]
    stats = _stats(rows) if rows else {"mean": {}, "std": {}, "min_val": {}, "max_val": {}}
    payload = {
        "experiment_id": experiment_id,
        "description": "LeWM-MPC EventSat run exported from space-world-models",
        "config": config or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_episodes": len(rows),
        "experiment_statistics": {
            "num_episodes": len(rows),
            **stats,
            "raw_episodes": [],
        },
        "episodes": [
            {
                "episode_id": idx,
                "episode_metrics": {"aggregated": row},
            }
            for idx, row in enumerate(rows)
        ],
    }
    path = output_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
