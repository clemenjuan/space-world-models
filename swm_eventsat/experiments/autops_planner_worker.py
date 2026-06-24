#!/usr/bin/env python3
"""JSON-lines LeWM-CEM planner worker for AUTOPS.

The worker runs in the space-world-models Python environment so AUTOPS does not
need to import Torch directly. It keeps rolling history and CEM state across
requests from one AUTOPS episode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.models.components import ARPredictor, Embedder, MLP
from core.models.vector_encoder import VectorEncoder
from core.models.vector_jepa import VectorJEPA
from swm_eventsat.schema import ACTION_NAMES, MODE_LIST

# CEM rollouts run in a subprocess (spawned by AUTOPS) that can share CPUs with a
# concurrent trainer. Under that contention torch intra-op thread oversubscription
# can hard-crash the worker (SIGSEGV -> "planner worker closed stdout" on the
# AUTOPS side). Pin to a small thread count for stability; override with
# LEWM_WORKER_THREADS (or OMP_NUM_THREADS) for faster standalone runs.
_WORKER_THREADS = max(1, int(os.environ.get("LEWM_WORKER_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))))
torch.set_num_threads(_WORKER_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _strip_checkpoint_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            out[key[len("model.") :]] = value
        elif not key.startswith("sigreg."):
            out[key] = value
    return out


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _build_model(artifact: dict[str, Any]) -> VectorJEPA:
    lewm = artifact["lewm"]
    cfg = artifact.get("model_config", {})
    embed_dim = int(lewm.get("embed_dim", 192))
    history_size = int(lewm.get("history_size", 3))
    obs_dim = int(lewm.get("obs_dim", 25))
    action_dim = int(lewm.get("action_dim", len(ACTION_NAMES)))
    encoder = VectorEncoder(
        in_dim=obs_dim,
        hidden_dim=int(cfg.get("encoder_hidden_dim", 256)),
        out_dim=embed_dim,
    )
    predictor = ARPredictor(
        num_frames=history_size,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=int(cfg.get("predictor_depth", 4)),
        heads=int(cfg.get("predictor_heads", 8)),
        mlp_dim=int(cfg.get("predictor_mlp_dim", 512)),
        dim_head=int(cfg.get("predictor_dim_head", 48)),
        dropout=float(cfg.get("dropout", 0.1)),
        emb_dropout=float(cfg.get("emb_dropout", 0.0)),
    )
    action_encoder = Embedder(input_dim=action_dim, smoothed_dim=action_dim, emb_dim=embed_dim)
    projector = MLP(embed_dim, int(cfg.get("projector_hidden_dim", 512)), embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, int(cfg.get("projector_hidden_dim", 512)), embed_dim, norm_fn=None)
    return VectorJEPA(encoder, predictor, action_encoder, projector, pred_proj)


class WorkerPlanner:
    def __init__(self, artifact_path: Path, device: str = "cpu") -> None:
        self.artifact_path = artifact_path.resolve()
        self.artifact_dir = self.artifact_path.parent
        self.artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        self.device = torch.device(device)
        torch.backends.nnpack.enabled = False
        self.model = _build_model(self.artifact)
        lewm = self.artifact["lewm"]
        checkpoint = torch.load(_resolve(self.artifact_dir, lewm["checkpoint"]), map_location="cpu", weights_only=False)
        self.model.load_state_dict(_strip_checkpoint_state(checkpoint.get("state_dict", checkpoint)), strict=False)
        self.model.to(self.device)
        self.model.eval()

        normalizers = np.load(_resolve(self.artifact_dir, lewm["normalizers"]))
        self.obs_mean = normalizers["obs_mean"].astype(np.float32)
        self.obs_std = normalizers["obs_std"].astype(np.float32)
        self.action_mean = normalizers["action_mean"].astype(np.float32)
        self.action_std = normalizers["action_std"].astype(np.float32)
        self.obs_std[self.obs_std < 1e-8] = 1.0
        self.action_std[self.action_std < 1e-8] = 1.0

        probe = self.artifact["probe"]
        self.W = np.asarray(probe["W"], dtype=np.float32)
        self.b = np.asarray(probe["b"], dtype=np.float32)
        self.attribute_names = [str(v) for v in probe["attribute_names"]]
        self.history_size = int(lewm.get("history_size", 3))
        self.action_dim = int(lewm.get("action_dim", len(ACTION_NAMES)))
        if self.action_dim != len(ACTION_NAMES):
            raise ValueError(f"mode-only planner artifacts must use action_dim={len(ACTION_NAMES)}, got {self.action_dim}")
        self.rng = np.random.default_rng(0)
        self.previous_solution: np.ndarray | None = None
        self.obs_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.last_action = self._action(0)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        self.previous_solution = None
        self.obs_history = []
        self.action_history = []
        self.last_action = self._action(0)

    def select(self, request: dict[str, Any]) -> dict[str, Any]:
        obs = np.asarray(request["obs25"], dtype=np.float32).reshape(-1)
        if obs.shape[0] != 25:
            raise ValueError(f"obs25 must have length 25, got {obs.shape}")
        self.obs_history.append(obs)
        self.action_history.append(self.last_action.copy())
        keep = max(int(request.get("horizon", 12)) + self.history_size, self.history_size + 1)
        self.obs_history = self.obs_history[-keep:]
        self.action_history = self.action_history[-keep:]

        horizon = max(1, int(request.get("horizon", 12)))
        samples = max(1, int(request.get("samples", 256)))
        elites = max(1, min(int(request.get("elites", 32)), samples))
        iterations = max(1, int(request.get("iterations", 4)))
        alpha = float(request.get("alpha", 0.7))
        first_mask = np.asarray(request.get("first_mask", [True] * len(MODE_LIST)), dtype=bool)
        weights_by_name = {str(k): float(v) for k, v in request.get("mode_weights", {}).items()}
        weights = np.asarray([weights_by_name.get(name, 0.0) for name in self.attribute_names], dtype=np.float32)

        probs = self._initial_probs(horizon)
        best_seq = None
        best_score = -np.inf
        for _ in range(iterations):
            seq = self._sample(probs, samples)
            allowed = np.flatnonzero(first_mask)
            if allowed.size == 0:
                allowed = np.asarray([0], dtype=np.int64)
            bad = ~first_mask[seq[:, 0]]
            if np.any(bad):
                seq[bad, 0] = self.rng.choice(allowed, size=int(np.sum(bad)))
            scores = self._score(seq, weights)
            idx = int(np.argmax(scores))
            if float(scores[idx]) > best_score:
                best_score = float(scores[idx])
                best_seq = seq[idx].copy()
            elite_idx = np.argpartition(scores, -elites)[-elites:]
            empirical = np.full_like(probs, 1e-4)
            for t in range(horizon):
                counts = np.bincount(seq[elite_idx, t], minlength=len(MODE_LIST)).astype(np.float64)
                empirical[t] += counts / max(1.0, counts.sum())
            empirical /= empirical.sum(axis=1, keepdims=True)
            probs = alpha * empirical + (1.0 - alpha) * probs
            probs /= probs.sum(axis=1, keepdims=True)
        if best_seq is None:
            best_seq = np.zeros(horizon, dtype=np.int64)
        self.previous_solution = best_seq
        self.last_action = self._action(int(best_seq[0]))
        self.action_history[-1] = self.last_action.copy()
        return {
            "mode_index": int(best_seq[0]),
            "mode": MODE_LIST[int(best_seq[0])],
            "best_sequence": best_seq.astype(int).tolist(),
            "best_score": float(best_score),
            "backend": "external_artifact_latent",
        }

    def _initial_probs(self, horizon: int) -> np.ndarray:
        if self.previous_solution is None:
            probs = np.full((horizon, len(MODE_LIST)), 1.0 / len(MODE_LIST), dtype=np.float64)
            probs[:, 0] += 0.08
            probs[:, 6] *= 0.20
            return probs / probs.sum(axis=1, keepdims=True)
        shifted = np.concatenate([self.previous_solution[1:], self.previous_solution[-1:]])[:horizon]
        probs = np.full((horizon, len(MODE_LIST)), 0.04 / (len(MODE_LIST) - 1), dtype=np.float64)
        for t, idx in enumerate(shifted):
            probs[t, int(idx)] = 0.96
        return probs / probs.sum(axis=1, keepdims=True)

    def _sample(self, probs: np.ndarray, samples: int) -> np.ndarray:
        seq = np.zeros((samples, probs.shape[0]), dtype=np.int64)
        actions = np.arange(len(MODE_LIST), dtype=np.int64)
        for t in range(probs.shape[0]):
            seq[:, t] = self.rng.choice(actions, size=samples, p=probs[t])
        return seq

    def _score(self, seq: np.ndarray, weights: np.ndarray) -> np.ndarray:
        z = self._rollout(seq)
        attrs = z[:, -1, :] @ self.W.T + self.b
        return (attrs @ weights).astype(np.float64)

    def _rollout(self, seq: np.ndarray) -> np.ndarray:
        obs = self._pad(np.asarray(self.obs_history, dtype=np.float32), 25)
        act = self._pad(np.asarray(self.action_history, dtype=np.float32), self.action_dim)
        action = self._encode_sequences(seq)
        obs_n = ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
        act_n = ((act - self.action_mean) / self.action_std).astype(np.float32)
        n, horizon, _ = action.shape
        with torch.no_grad():
            encoded = self.model.encode({"obs": torch.from_numpy(obs_n[None]).to(self.device)})
            emb_hist = encoded["emb"].repeat(n, 1, 1)
            act_hist = torch.from_numpy(np.repeat(act_n[None], n, axis=0)).to(self.device)
            first = ((action[:, 0] - self.action_mean) / self.action_std).astype(np.float32)
            act_hist[:, -1, :] = torch.from_numpy(first).to(self.device)
            pred_rows = []
            for t in range(horizon):
                act_emb = self.model.action_encoder(act_hist[:, -self.history_size :])
                pred = self.model.predict(emb_hist[:, -self.history_size :], act_emb)[:, -1:]
                pred_rows.append(pred[:, 0])
                emb_hist = torch.cat([emb_hist, pred], dim=1)
                if t + 1 < horizon:
                    nxt = ((action[:, t + 1] - self.action_mean) / self.action_std).astype(np.float32)
                    act_hist = torch.cat([act_hist, torch.from_numpy(nxt[:, None]).to(self.device)], dim=1)
            return torch.stack(pred_rows, dim=1).detach().cpu().numpy().astype(np.float32)

    def _pad(self, arr: np.ndarray, dim: int) -> np.ndarray:
        arr = arr.reshape(-1, dim)
        if arr.shape[0] >= self.history_size:
            return arr[-self.history_size :]
        first = arr[0] if arr.shape[0] else np.zeros(dim, dtype=np.float32)
        return np.concatenate([np.repeat(first[None], self.history_size - arr.shape[0], axis=0), arr], axis=0)

    def _action(self, mode_index: int) -> np.ndarray:
        out = np.zeros(self.action_dim, dtype=np.float32)
        out[int(mode_index)] = 1.0
        return out

    def _encode_sequences(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.int64)
        out = np.zeros((*seq.shape, self.action_dim), dtype=np.float32)
        rows = np.indices(seq.shape)
        out[rows[0], rows[1], seq] = 1.0
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    planner = WorkerPlanner(Path(args.artifact), device=args.device)
    print(json.dumps({"ok": True, "backend": "external_artifact_latent"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "seed":
                planner.seed(int(request.get("seed", 0)))
                response = {"ok": True}
            elif request.get("type") == "select":
                response = {"ok": True, **planner.select(request)}
            else:
                response = {"ok": False, "error": f"unknown request type {request.get('type')}"}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
