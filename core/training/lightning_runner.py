"""Small helpers shared by LeWM Lightning training entrypoints."""
from __future__ import annotations

from functools import partial
from typing import Any

from omegaconf import OmegaConf

from core.models.lewm_loss import lewm_forward


def make_lewm_forward(cfg: Any):
    """Return a stable-pretraining forward callback for vector LeWM models."""

    def _forward(self, batch, stage, cfg):
        flat = {
            "history_size": cfg.history_size,
            "num_preds": cfg.num_preds,
            "sigreg_weight": cfg.loss.sigreg.weight,
        }
        out = lewm_forward(self.model, self.sigreg, batch, flat)
        self.log_dict(
            {f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k},
            on_step=True,
            sync_dist=True,
            batch_size=batch["obs"].size(0),
        )
        return out

    return partial(_forward, cfg=cfg)


def config_to_container(cfg: Any) -> dict[str, Any]:
    """OmegaConf-aware conversion used by stable-pretraining hparams."""
    return OmegaConf.to_container(cfg, resolve=True)

