"""Windowed torch Dataset over generated OD trajectories, with z-score normalization."""
import numpy as np
import torch
from torch.utils.data import Dataset


def fit_normalizers(npz_path):
    """Return {'obs': (mean,std), 'action': (mean,std)} over all timesteps."""
    blob = np.load(npz_path)
    out = {}
    for key in ("obs", "action"):
        flat = blob[key].reshape(-1, blob[key].shape[-1])
        mean = flat.mean(0)
        std = flat.std(0)
        std[std < 1e-8] = 1.0
        out[key] = (torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
    return out


class OdWindowDataset(Dataset):
    def __init__(self, npz_path, window=4, normalizers=None):
        blob = np.load(npz_path)
        self.obs = torch.tensor(blob["obs"], dtype=torch.float32)        # (E, L, 4)
        self.action = torch.tensor(blob["action"], dtype=torch.float32)  # (E, L, 3)
        self.window = window
        self.normalizers = normalizers
        E, L, _ = self.obs.shape
        self.index = [
            (e, s) for e in range(E) for s in range(L - window + 1)
        ]

    def __len__(self):
        return len(self.index)

    def _norm(self, x, key):
        if self.normalizers is None:
            return x
        mean, std = self.normalizers[key]
        return (x - mean) / std

    def __getitem__(self, i):
        e, s = self.index[i]
        obs = self._norm(self.obs[e, s : s + self.window], "obs")
        action = self._norm(self.action[e, s : s + self.window], "action")
        return {"obs": obs, "action": action}
