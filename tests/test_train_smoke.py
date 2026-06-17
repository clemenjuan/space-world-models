from functools import partial

import lightning as pl
from torch.utils.data import DataLoader

from data.generate_dataset import generate
from module import SIGReg
from models.od_forward import od_lejepa_forward
from od_datasets.od_dataset import OdWindowDataset, fit_normalizers
from spt_compat import (
    configure_utf8_stdio,
    patch_pyarrow_for_legacy_datasets,
    patch_stable_pretraining_windows_signals,
    stub_lance_if_no_avx,
)


def test_train_smoke(tmp_path):
    configure_utf8_stdio()
    patch_pyarrow_for_legacy_datasets()
    stub_lance_if_no_avx()
    import stable_pretraining as spt
    from tests.test_model import _make_odjepa
    patch_stable_pretraining_windows_signals()

    path = tmp_path / "traj.npz"
    generate(n_episodes=4, episode_len=16, out_path=str(path), seed=0)
    ds = OdWindowDataset(str(path), window=4, normalizers=fit_normalizers(str(path)))
    train_loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds, batch_size=8, shuffle=False)
    model = _make_odjepa()

    def _fwd(self, batch, stage, cfg):
        out = od_lejepa_forward(self.model, self.sigreg, batch, cfg)
        self.log_dict(
            {f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k},
            batch_size=batch["obs"].size(0),
        )
        return out

    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=64),
        forward=partial(
            _fwd,
            cfg={"history_size": 3, "num_preds": 1, "sigreg_weight": 0.09},
        ),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": {"type": "AdamW", "lr": 5e-5},
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            }
        },
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    spt.Manager(trainer=trainer, module=module, data=data_module, seed=0)()
