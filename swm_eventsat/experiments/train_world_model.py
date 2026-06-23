"""Full Lightning training loop for the EventSat LeWM model."""
from functools import partial

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from core.models.components import SIGReg
from core.data.window_dataset import WindowedTrajectoryDataset, fit_normalizers
from core.models.lewm_loss import lewm_forward
from core.training.spt_compat import (
    configure_utf8_stdio,
    patch_pyarrow_for_legacy_datasets,
    patch_stable_pretraining_windows_signals,
    stub_lance_if_no_avx,
)


OmegaConf.register_new_resolver("eval", eval, replace=True)


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


@hydra.main(version_base=None, config_path="../config", config_name="train")
def run(cfg):
    configure_utf8_stdio()
    patch_pyarrow_for_legacy_datasets()
    stub_lance_if_no_avx()
    if OmegaConf.has_resolver("eval"):
        OmegaConf.clear_resolver("eval")
    import stable_pretraining as spt
    from lightning.pytorch.loggers import WandbLogger
    patch_stable_pretraining_windows_signals()

    pl.seed_everything(cfg.seed, workers=True)
    norms = fit_normalizers(cfg.data.path)
    full = WindowedTrajectoryDataset(cfg.data.path, window=cfg.data.window, normalizers=norms)
    gen = torch.Generator().manual_seed(cfg.seed)
    n_train = int(len(full) * cfg.data.train_split)
    train_set, val_set = random_split(full, [n_train, len(full) - n_train], generator=gen)
    train = DataLoader(
        full if len(val_set) == 0 else train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val = DataLoader(
        val_set if len(val_set) > 0 else train_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
    )

    world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                **OmegaConf.to_container(cfg.optimizer, resolve=True),
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(_forward, cfg=cfg),
        optim=optimizers,
        hparams=OmegaConf.to_container(cfg, resolve=True),
    )
    data_module = spt.data.DataModule(train=train, val=val)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(**cfg.trainer, logger=logger, num_sanity_val_steps=0)
    manager = spt.Manager(trainer=trainer, module=module, data=data_module, seed=cfg.seed)
    manager()


if __name__ == "__main__":
    run()
