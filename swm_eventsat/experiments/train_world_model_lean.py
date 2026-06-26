"""Step-budgeted LeWM trainer (CPU/GPU), no lightning/stable_pretraining.

Functionally mirrors ``train_world_model`` — same VectorJEPA from the same
``model/eventsat_autops`` config, same ``lewm_forward`` loss, same canonical
optimizer/hparams — but:

  * trains to a **step budget** (``--max-steps``) rather than epochs, so the
    target is dataset-size invariant (the LeJEPA/LeWM recipe is specified in
    steps, ~150k);
  * logs LeJEPA collapse diagnostics (embedding std, effective rank,
    off-diagonal correlation) so the SIGReg weight can be judged from the
    embedding geometry rather than trusted blindly;
  * avoids the lightning/torchmetrics/torchvision import chain, which is broken
    under torch 2.12.1+cpu in this environment.

SIGReg weight defaults to 0.09 (LeJEPA reports the objective peaks near 0.09 and
stays within ~80% of peak across 0.01-0.2, so it is robust, not finicky).

Example:
    python -m swm_eventsat.experiments.train_world_model_lean \
        --max-steps 150000 --val-every 1000 \
        --ckpt-out outputs/eventsat_autops_lewm/lewm.ckpt
"""
from __future__ import annotations

import argparse
import math
import os
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

from core.models.components import SIGReg
from core.models.lewm_loss import lewm_forward
from core.data.window_dataset import WindowedTrajectoryDataset, fit_normalizers

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    "/home/clemente/autops-agentic-framework/data/world_model/"
    "eventsat_autops_v1/eventsat_world_model_v1.npz"
)


@torch.no_grad()
def collapse_diagnostics(model, sigreg, loader, cfg, device, max_batches=8):
    """LeJEPA embedding-health metrics on a few val batches."""
    embs = []
    val = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    n = 0
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = lewm_forward(model, sigreg, batch, cfg)
        bs = batch["obs"].size(0)
        for k in val:
            val[k] += float(out[k]) * bs
        n += bs
        embs.append(out["emb"].reshape(-1, out["emb"].shape[-1]).cpu())
        if i + 1 >= max_batches:
            break
    val = {k: v / max(n, 1) for k, v in val.items()}
    E = torch.cat(embs, 0).float()
    E = E - E.mean(0, keepdim=True)
    std = E.std(0)
    cov = (E.T @ E) / max(E.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    eff_rank = float((eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12))  # participation ratio
    d = torch.sqrt(torch.diag(cov)).clamp_min(1e-8)
    corr = cov / (d[:, None] * d[None, :])
    offdiag = float(corr.abs().sum() - corr.diag().abs().sum()) / (corr.numel() - corr.shape[0])
    return val, {
        "emb_std_mean": float(std.mean()),
        "emb_std_min": float(std.min()),
        "eff_rank": eff_rank,
        "eff_rank_frac": eff_rank / E.shape[1],
        "offdiag_corr": offdiag,
    }


def build_model(model_cfg_path, embed_dim, history, action_dim):
    model_cfg = OmegaConf.load(model_cfg_path)
    model_cfg.action_encoder.input_dim = action_dim
    model_cfg.action_encoder.smoothed_dim = action_dim
    merged = OmegaConf.merge(
        OmegaConf.create({"embed_dim": embed_dim, "history_size": history}),
        {"model": model_cfg},
    )
    OmegaConf.resolve(merged)
    return hydra.utils.instantiate(merged.model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--max-steps", type=int, default=150000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--ckpt-out", default=str(ROOT / "outputs/eventsat_autops_lewm/lewm.ckpt"))
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=os.environ.get("WANDB_RUN_NAME", "eventsat-autops-action7-lewm"))
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    window = args.history + args.num_preds

    norms = fit_normalizers(args.data)
    full = WindowedTrajectoryDataset(args.data, window=window, normalizers=norms)
    action_dim = int(full.action.shape[-1])
    n_train = int(len(full) * args.train_split)
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(full, [n_train, len(full) - n_train], generator=gen)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=False)

    model = build_model(ROOT / "swm_eventsat/config/model/eventsat_autops.yaml",
                        args.embed_dim, args.history, action_dim).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024)
    n_params = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_loader)
    print(f"params={n_params:,} obs_dim={full.obs.shape[-1]} action_dim={action_dim} "
          f"windows={len(full)} train={len(train_set)} val={len(val_set)} "
          f"steps/epoch={steps_per_epoch} max_steps={args.max_steps} "
          f"(~{args.max_steps/max(steps_per_epoch,1):.1f} epochs) device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    cfg = {"history_size": args.history, "num_preds": args.num_preds, "sigreg_weight": args.sigreg_weight}

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "sps-tum"),
            project=os.environ.get("WANDB_PROJECT", "space-world-models"),
            name=args.run_name,
            tags=["eventsat", "autops", "obs25", f"action{action_dim}", "step-budget", "lean"],
            config={**vars(args), "params": n_params, "windows": len(full), "steps_per_epoch": steps_per_epoch},
        )
        print(f"W&B: {run.url}")

    ckpt_out = Path(args.ckpt_out)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    model.train()
    running = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    seen = 0

    for step, batch in enumerate(cycle(train_loader)):
        if step >= args.max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = lewm_forward(model, sigreg, batch, cfg)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k in running:
            running[k] += float(out[k].detach())
        seen += 1

        if (step + 1) % args.val_every == 0 or step + 1 == args.max_steps:
            tr = {k: v / seen for k, v in running.items()}
            running = {k: 0.0 for k in running}
            seen = 0
            model.eval()
            va, diag = collapse_diagnostics(model, sigreg, val_loader, cfg, device)
            model.train()
            log = {
                "step": step + 1, "lr": sched.get_last_lr()[0],
                **{f"train/{k}": v for k, v in tr.items()},
                **{f"val/{k}": v for k, v in va.items()},
                **{f"diag/{k}": v for k, v in diag.items()},
            }
            if run:
                run.log(log)
            print(f"step {step+1:6d}  train_pred={tr['pred_loss']:.4f}  val_pred={va['pred_loss']:.4f}  "
                  f"val_loss={va['loss']:.4f}  eff_rank={diag['eff_rank']:.1f}/{args.embed_dim}  "
                  f"emb_std={diag['emb_std_mean']:.3f}  offdiag={diag['offdiag_corr']:.3f}", flush=True)
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save({"state_dict": model.state_dict(),
                            "config": {"embed_dim": args.embed_dim, "history_size": args.history,
                                       "obs_dim": int(full.obs.shape[-1]), "action_dim": action_dim},
                            "step": step + 1, "val_loss": best_val}, ckpt_out)

    if run:
        run.summary["best_val_loss"] = best_val
        run.finish()
    print(f"DONE best_val_loss={best_val:.4f} ckpt={ckpt_out}")


if __name__ == "__main__":
    main()
