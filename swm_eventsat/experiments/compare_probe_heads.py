#!/usr/bin/env python3
"""Compare linear vs non-linear probe heads on frozen LeWM latents.

Diagnostic: is each mission attribute *linearly* decodable from the LeWM latent,
or only non-linearly? Fits a ridge (affine) probe and a small MLP on the SAME
frozen latents with an **episode-level** train/val holdout (whole episodes held
out, so temporally-correlated timesteps don't leak across the split), and reports
per-attribute R2 and rmse/std for both.

Context: the AO CEM planner scores candidates with the affine W.z+b probe, so a
large MLP>linear gap on a utility-relevant attribute (e.g. communication_opportunity)
means the planner is leaving signal on the table and a non-linear scoring head
(artifact + backend change) may be worth it. This script only measures; it does
not change the artifact contract.

Example:
    python -m swm_eventsat.experiments.compare_probe_heads \
        --latents outputs/eventsat_autops_latents.npz --val-episodes 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swm_eventsat.models.probes import build_attribute_targets, DEFAULT_ATTRIBUTE_NAMES
from swm_eventsat.schema import load_world_model_dataset

DEFAULT_DATASET = (
    "/home/clemente/autops-agentic-framework/data/world_model/"
    "eventsat_autops_v1/eventsat_world_model_v1.npz"
)


def _load_latents(path: Path) -> np.ndarray:
    blob = np.load(path)
    if hasattr(blob, "files"):
        arr = blob["latents"] if "latents" in blob.files else blob[blob.files[0]]
    else:
        arr = blob
    return np.asarray(arr, dtype=np.float32)


def _r2_and_rmse_over_std(pred_n: np.ndarray, y_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both inputs are standardized by TRAIN target stats -> rmse_n == raw rmse/std."""
    err = pred_n - y_n
    rmse_over_std = np.sqrt((err ** 2).mean(0))
    denom = (y_n ** 2).sum(0)
    r2 = np.where(denom > 1e-12, 1.0 - (err ** 2).sum(0) / denom, np.nan)
    return r2, rmse_over_std


def _fit_linear(Xtr, Ytr, Xv, ridge):
    A = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1), np.float32)], 1)
    reg = ridge * np.eye(A.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ Ytr)
    Av = np.concatenate([Xv, np.ones((Xv.shape[0], 1), np.float32)], 1)
    return Av @ coef


def _fit_mlp(Xtr, Ytr, Xv, hidden, epochs, lr, weight_decay, seed):
    import torch

    torch.manual_seed(seed)
    layers, prev = [], Xtr.shape[1]
    for h in hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        prev = h
    layers.append(torch.nn.Linear(prev, Ytr.shape[1]))
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = torch.nn.MSELoss()
    xt, yt, xvt = torch.tensor(Xtr), torch.tensor(Ytr), torch.tensor(Xv)
    idx = np.arange(xt.shape[0])
    rng = np.random.default_rng(seed)
    bs = 2048
    for _ in range(epochs):
        rng.shuffle(idx)
        net.train()
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xvt).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--latents", default=str(ROOT / "outputs/eventsat_autops_latents.npz"))
    p.add_argument("--val-episodes", type=int, default=3, help="whole episodes held out for validation")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=100)
    p.add_argument("--hidden", default="256,128")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    ds = load_world_model_dataset(args.dataset)
    names = list(DEFAULT_ATTRIBUTE_NAMES)
    Y = build_attribute_targets(ds).astype(np.float32)          # (E,T,K)
    X = _load_latents(Path(args.latents))                        # (E,T,D)
    if X.shape[:2] != Y.shape[:2]:
        raise ValueError(f"latents {X.shape} and targets {Y.shape} episode/time axes differ")
    E = X.shape[0]
    n_val = max(1, min(args.val_episodes, E - 1))
    val_ep = set(range(E - n_val, E))                            # hold out the LAST n episodes
    tr_ep = [e for e in range(E) if e not in val_ep]

    Xtr = X[tr_ep].reshape(-1, X.shape[-1]); Ytr = Y[tr_ep].reshape(-1, Y.shape[-1])
    Xv = X[sorted(val_ep)].reshape(-1, X.shape[-1]); Yv = Y[sorted(val_ep)].reshape(-1, Y.shape[-1])

    xm, xs = Xtr.mean(0), Xtr.std(0); xs[xs < 1e-8] = 1.0
    ym, ysd = Ytr.mean(0), Ytr.std(0); ysd[ysd < 1e-8] = 1.0
    Xtr_n, Xv_n = ((Xtr - xm) / xs).astype(np.float32), ((Xv - xm) / xs).astype(np.float32)
    Ytr_n, Yv_n = ((Ytr - ym) / ysd).astype(np.float32), ((Yv - ym) / ysd).astype(np.float32)
    degenerate = [names[i] for i in range(len(names)) if Ytr[:, i].std() < 1e-8]

    hidden = [int(h) for h in str(args.hidden).split(",") if h]
    pred_lin = _fit_linear(Xtr_n, Ytr_n, Xv_n, args.ridge)
    pred_mlp = _fit_mlp(Xtr_n, Ytr_n, Xv_n, hidden, args.mlp_epochs, args.lr, args.weight_decay, args.seed)
    lin_r2, lin_rmse = _r2_and_rmse_over_std(pred_lin, Yv_n)
    mlp_r2, mlp_rmse = _r2_and_rmse_over_std(pred_mlp, Yv_n)

    print(f"episode-level holdout: train={len(tr_ep)} ep, val={sorted(val_ep)} ({Xv.shape[0]} steps)")
    print(f"{'attribute':26s} {'LIN R2':>8s} {'MLP R2':>8s} {'LIN rmse/std':>12s} {'MLP rmse/std':>12s}  verdict")
    rows = {}
    for i, n in enumerate(names):
        if n in degenerate:
            print(f"{n:26s} {'degen':>8s} {'degen':>8s} {'--':>12s} {'--':>12s}  skip")
            rows[n] = {"degenerate": True}
            continue
        d = float(mlp_r2[i] - lin_r2[i])
        v = f"MLP +{d:.3f}" if d > 0.01 else ("~same" if abs(d) <= 0.01 else f"LIN +{-d:.3f}")
        print(f"{n:26s} {lin_r2[i]:8.3f} {mlp_r2[i]:8.3f} {lin_rmse[i]:12.3f} {mlp_rmse[i]:12.3f}  {v}")
        rows[n] = {"lin_r2": float(lin_r2[i]), "mlp_r2": float(mlp_r2[i]),
                   "lin_rmse_over_std": float(lin_rmse[i]), "mlp_rmse_over_std": float(mlp_rmse[i]),
                   "mlp_minus_lin_r2": d}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"val_episodes": sorted(val_ep), "hidden": hidden, "mlp_epochs": args.mlp_epochs,
             "degenerate": degenerate, "attributes": rows}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
