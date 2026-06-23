"""Windowed torch Dataset over generated trajectory archives.

The archive schema stores ``obs`` and ``action`` arrays are shaped ``(episode, time, dim)``. Additional truth arrays
may be present, but LeWM training consumes only observation/action windows.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def fit_normalizers(npz_path):
    """Return ``{"obs": (mean, std), "action": (mean, std)}`` over all timesteps."""
    blob = np.load(npz_path)
    out = {}
    for key in ("obs", "action"):
        flat = blob[key].reshape(-1, blob[key].shape[-1])
        mean = flat.mean(0)
        std = flat.std(0)
        std[std < 1e-8] = 1.0
        out[key] = (
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32),
        )
    return out


class WindowedTrajectoryDataset(Dataset):
    """Dimension-agnostic trajectory window dataset."""

    def __init__(self, npz_path, window=4, normalizers=None):
        blob = np.load(npz_path)
        self.obs = torch.tensor(blob["obs"], dtype=torch.float32)
        self.action = torch.tensor(blob["action"], dtype=torch.float32)
        self.window = int(window)
        self.normalizers = normalizers
        episodes, length, _ = self.obs.shape
        self.index = [(e, s) for e in range(episodes) for s in range(length - self.window + 1)]

    def __len__(self):
        return len(self.index)

    def _norm(self, x, key):
        if self.normalizers is None:
            return x
        mean, std = self.normalizers[key]
        return (x - mean) / std

    def __getitem__(self, i):
        episode, start = self.index[i]
        obs = self._norm(self.obs[episode, start : start + self.window], "obs")
        action = self._norm(self.action[episode, start : start + self.window], "action")
        return {"obs": obs, "action": action}


__all__ = ["WindowedTrajectoryDataset", "fit_normalizers"]
