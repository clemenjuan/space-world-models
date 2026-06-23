"""Scheduling-facing wrapper around vector LeWM models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from swm_eventsat.models.checkpoint_io import latent_rollout


@dataclass
class LeWMRolloutModel:
    """Small adapter exposing a planner-friendly rollout method."""

    model: Any
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]]
    history_size: int
    device: str = "cpu"

    def rollout(
        self,
        obs_context: np.ndarray,
        action_context: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        return latent_rollout(
            model=self.model,
            obs_context=obs_context,
            action_context=action_context,
            candidate_actions=candidates,
            normalizers=self.normalizers,
            history_size=self.history_size,
            device=self.device,
        )

