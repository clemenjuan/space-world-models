"""Receding-horizon latent MPC built around categorical CEM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from swm_eventsat.planning.cem import CEMConfig, CEMResult, categorical_cem


@dataclass
class LatentMPC:
    """Minimal planner wrapper: sample actions, roll latents, score, execute first."""

    rollout_fn: Callable[[np.ndarray], np.ndarray]
    score_latents_fn: Callable[[np.ndarray], np.ndarray]
    cem_config: CEMConfig

    def plan(
        self,
        rng: np.random.Generator | None = None,
        action_mask: np.ndarray | None = None,
    ) -> CEMResult:
        def score_actions(candidates: np.ndarray) -> np.ndarray:
            return self.score_latents_fn(self.rollout_fn(candidates))

        return categorical_cem(
            score_fn=score_actions,
            config=self.cem_config,
            rng=rng,
            action_mask=action_mask,
        )

    def select_action(self, **kwargs) -> int:
        return int(self.plan(**kwargs).action_sequence[0])

