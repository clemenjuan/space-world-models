"""Adapter between EventSat planner code and existing VectorJEPA/LeWM models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class LeWMRolloutAdapter:
    """Frozen latent rollout API expected by the CEM planner."""

    model: Any
    normalizers: Dict[str, tuple[np.ndarray, np.ndarray]]
    history_size: int = 4
    device: str = "cpu"

    def encode_history(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        import torch

        obs_n = _normalize(obs[-self.history_size :], self.normalizers["obs"])
        act_n = _normalize(action[-self.history_size :], self.normalizers["action"])
        batch = {
            "obs": torch.from_numpy(obs_n[None].astype(np.float32)).to(self.device),
            "action": torch.from_numpy(act_n[None].astype(np.float32)).to(self.device),
        }
        with torch.no_grad():
            encoded = self.model.encode(batch)
        return encoded["emb"].detach().cpu().numpy()[0].astype(np.float32)

    def rollout(self, history: Dict[str, np.ndarray], action11: np.ndarray) -> np.ndarray:
        import torch

        obs = np.asarray(history["obs"], dtype=np.float32)[-self.history_size :]
        act = np.asarray(history["action"], dtype=np.float32)[-self.history_size :]
        candidates = np.asarray(action11, dtype=np.float32)
        n, horizon, _ = candidates.shape
        obs_n = _normalize(obs, self.normalizers["obs"])
        act_n = _normalize(act, self.normalizers["action"])
        with torch.no_grad():
            batch = {
                "obs": torch.from_numpy(obs_n[None]).to(self.device),
                "action": torch.from_numpy(act_n[None]).to(self.device),
            }
            encoded = self.model.encode(batch)
            emb_hist = encoded["emb"].repeat(n, 1, 1)
            act_hist = torch.from_numpy(np.repeat(act_n[None], n, axis=0)).to(self.device)
            first = _normalize(candidates[:, 0], self.normalizers["action"])
            act_hist[:, -1, :] = torch.from_numpy(first).to(self.device)
            pred_rows = []
            for t in range(horizon):
                act_emb = self.model.action_encoder(act_hist[:, -self.history_size :])
                pred = self.model.predict(emb_hist[:, -self.history_size :], act_emb)[:, -1:]
                pred_rows.append(pred[:, 0])
                emb_hist = torch.cat([emb_hist, pred], dim=1)
                if t + 1 < horizon:
                    nxt = _normalize(candidates[:, t + 1], self.normalizers["action"])
                    act_hist = torch.cat([act_hist, torch.from_numpy(nxt[:, None]).to(self.device)], dim=1)
            return torch.stack(pred_rows, dim=1).detach().cpu().numpy().astype(np.float32)


def _normalize(x: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = normalizer
    std = np.where(std < 1e-8, 1.0, std)
    return ((x.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)


def fit_normalizers_from_dataset(dataset) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for key in ("obs", "action"):
        arr = getattr(dataset, key).reshape(-1, getattr(dataset, key).shape[-1]).astype(np.float32)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-8] = 1.0
        out[key] = (mean.astype(np.float32), std.astype(np.float32))
    return out
