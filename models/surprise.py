"""Surprise metric for FDIR: per-step prediction error ||z_hat_t - enc(o_t)||^2.

Pure inference over a trained LeWM model (encoder + predictor). For each absolute
timestep t, predict the next-step embedding from the preceding history window and
score the squared L2 distance to the actual encoded observation. High surprise flags
anomalous telemetry. No gradients, no new loss terms; matches the predict windowing in
jepa.JEPA.rollout / models.od_forward.od_lejepa_forward.
"""
import torch


def surprise_score(model, obs_seq, action_seq, history_size=3):
    """Per-step surprise ||z_hat_t - enc(o_t)||^2 over a single episode.

    obs_seq: (1, T, obs_dim); action_seq: (1, T, action_dim) one-hot float.
    Returns: (T - history_size,) tensor, aligned to absolute steps history_size .. T-1.
    Dim-agnostic (any obs_dim / action_dim). Pure inference (torch.no_grad).
    """
    with torch.no_grad():
        out = model.encode({"obs": obs_seq, "action": action_seq})
        emb = out["emb"]          # (1, T, D)
        act_emb = out["act_emb"]  # (1, T, A_emb)
        hs = history_size
        t_total = emb.size(1)

        scores = []
        for t in range(hs, t_total):
            # predict z_hat_t from the preceding history window; last step is the
            # prediction (see jepa.JEPA.rollout: pred = predict(emb[:, -HS:], ...)[:, -1]).
            z_hat_t = model.predict(emb[:, t - hs:t], act_emb[:, t - hs:t])[:, -1]  # (1, D)
            # squared L2 over the embedding dimension, scalar per step.
            score_t = ((z_hat_t - emb[:, t]) ** 2).sum()
            scores.append(score_t)

        return torch.stack(scores)  # (T - history_size,)
