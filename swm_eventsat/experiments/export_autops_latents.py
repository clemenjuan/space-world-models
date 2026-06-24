#!/usr/bin/env python3
"""Export frozen LeWM latents for an AUTOPS EventSat world-model dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from core.models.components import ARPredictor, Embedder, MLP
from core.models.vector_encoder import VectorEncoder
from core.models.vector_jepa import VectorJEPA
from swm_eventsat.schema import load_world_model_dataset


DEFAULT_DATASET = (
    "/home/clemente/autops-agentic-framework/data/world_model/"
    "eventsat_autops_v1/eventsat_world_model_v1.npz"
)


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
        elif not key.startswith("sigreg."):
            out[key] = value
    return out


def _manual_model(obs_dim: int, action_dim: int, embed_dim: int, history_size: int) -> VectorJEPA:
    encoder = VectorEncoder(in_dim=obs_dim, hidden_dim=256, out_dim=embed_dim)
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=4,
        heads=8,
        mlp_dim=512,
        dim_head=48,
        dropout=0.1,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(embed_dim, 512, embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, 512, embed_dim, norm_fn=None)
    return VectorJEPA(encoder, predictor, action_encoder, projector, pred_proj)


def _load_model(
    checkpoint_path: Path,
    obs_dim: int,
    action_dim: int,
    embed_dim: int,
    history_size: int,
    device: torch.device,
) -> tuple[VectorJEPA, dict[str, Any]]:
    hparams_path = checkpoint_path.parent.parent / "hparams.yaml"
    if hparams_path.exists():
        cfg = OmegaConf.load(hparams_path)
        model = hydra.utils.instantiate(cfg.model)
        meta = OmegaConf.to_container(cfg, resolve=True)
    else:
        model = _manual_model(obs_dim, action_dim, embed_dim, history_size)
        meta = {
            "embed_dim": embed_dim,
            "history_size": history_size,
            "model": "manual_eventsat_autops",
        }
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _strip_model_prefix(checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, dict(meta)


def _normalizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/eventsat_autops_latents.npz")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--history-size", type=int, default=3)
    args = parser.parse_args()

    dataset = load_world_model_dataset(args.dataset)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    device = torch.device(args.device)
    model, model_meta = _load_model(
        checkpoint,
        obs_dim=dataset.obs.shape[-1],
        action_dim=dataset.action.shape[-1],
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        device=device,
    )
    obs_mean, obs_std = _normalizer(dataset.obs)
    flat = ((dataset.obs.reshape(-1, dataset.obs.shape[-1]) - obs_mean) / obs_std).astype(np.float32)

    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(args.batch_size)):
            end = min(start + int(args.batch_size), flat.shape[0])
            batch = torch.from_numpy(flat[start:end, None, :]).to(device)
            emb = model.encode({"obs": batch})["emb"][:, 0]
            rows.append(emb.detach().cpu().numpy().astype(np.float32))
    latents = np.concatenate(rows, axis=0).reshape(dataset.obs.shape[0], dataset.obs.shape[1], -1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        latents=latents,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.path),
        "checkpoint": str(checkpoint.resolve()),
        "latents": str(out.resolve()),
        "latent_shape": list(latents.shape),
        "model": model_meta,
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} latents={latents.shape}")


if __name__ == "__main__":
    main()
