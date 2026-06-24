#!/usr/bin/env python3
"""Train linear EventSat mission-attribute probes from AUTOPS trace datasets.

If --latents is omitted, the script uses the 25D observation vector as a smoke
feature source. For paper runs, pass frozen LeWM latents with shape (E,T,D).
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

from swm_eventsat.models.probes import build_attribute_targets, fit_ridge_probe
from swm_eventsat.schema import load_world_model_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--latents", default="", help="Optional .npz/.npy latents with shape (E,T,D)")
    parser.add_argument("--out", default="outputs/eventsat_autops_probe.npz")
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = load_world_model_dataset(args.dataset)
    targets = build_attribute_targets(dataset)
    if args.latents:
        path = Path(args.latents)
        if path.suffix == ".npz":
            blob = np.load(path)
            latents = blob["latents"] if "latents" in blob else blob[blob.files[0]]
        else:
            latents = np.load(path)
    else:
        latents = dataset.obs
    fit = fit_ridge_probe(latents, targets, ridge=args.ridge)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        W=fit.W,
        b=fit.b,
        attribute_names=np.asarray(fit.attribute_names),
        target_mean=fit.target_mean,
        target_std=fit.target_std,
    )
    manifest = {
        "probe": str(out),
        "dataset": str(dataset.path),
        "feature_source": str(args.latents) if args.latents else "obs25_smoke_features",
        "attribute_names": fit.attribute_names,
        "rmse": fit.rmse,
        "dataset_steps": dataset.dataset_steps,
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} attributes={len(fit.attribute_names)} dataset_steps={dataset.dataset_steps}")


if __name__ == "__main__":
    main()
