"""Surprise metric for FDIR: per-step prediction error ||z_hat_t - enc(o_t)||^2.

Pure inference over a trained LeWM model (encoder + predictor). For each absolute
timestep t, predict the next-step embedding from the preceding history window and
score the squared L2 distance to the actual encoded observation. High surprise flags
anomalous telemetry. No gradients, no new loss terms; matches the predict windowing in
jepa.JEPA.rollout / models.od_forward.od_lejepa_forward.

This module also provides the detection primitives used to turn the raw surprise
signal into calibrated, quantitative detector statistics (threshold, detection
delay, false-alarm count) so multi-seed / multi-fault-mode comparisons are
grounded rather than read off a single plot.
"""
import numpy as np
import torch


def surprise_scores(model, obs_seq, action_seq, history_size=3):
    """Batched per-step surprise ||z_hat_t - enc(o_t)||^2 over many episodes.

    obs_seq: (N, T, obs_dim); action_seq: (N, T, action_dim). The squared L2 is
    reduced over the embedding dimension only, so each episode keeps its own
    per-step score.
    Returns: (N, T - history_size) tensor, aligned to absolute steps
    history_size .. T-1. Dim-agnostic. Pure inference (torch.no_grad).
    """
    with torch.no_grad():
        out = model.encode({"obs": obs_seq, "action": action_seq})
        emb = out["emb"]          # (N, T, D)
        act_emb = out["act_emb"]  # (N, T, A_emb)
        hs = history_size
        t_total = emb.size(1)

        scores = []
        for t in range(hs, t_total):
            # predict z_hat_t from the preceding history window; last step is the
            # prediction (see jepa.JEPA.rollout: pred = predict(emb[:, -HS:], ...)[:, -1]).
            z_hat_t = model.predict(emb[:, t - hs:t], act_emb[:, t - hs:t])[:, -1]  # (N, D)
            # squared L2 over the embedding dim only -> one score per episode.
            score_t = ((z_hat_t - emb[:, t]) ** 2).sum(dim=-1)  # (N,)
            scores.append(score_t)

        return torch.stack(scores, dim=1)  # (N, T - history_size)


def surprise_score(model, obs_seq, action_seq, history_size=3):
    """Per-step surprise ||z_hat_t - enc(o_t)||^2 over a single episode.

    obs_seq: (1, T, obs_dim); action_seq: (1, T, action_dim) one-hot float.
    Returns: (T - history_size,) tensor, aligned to absolute steps history_size .. T-1.
    Thin wrapper over surprise_scores for the single-episode case.
    """
    return surprise_scores(model, obs_seq, action_seq, history_size=history_size)[0]


def calibrate_threshold(nominal_scores, k=3.0):
    """Detection threshold from pooled nominal surprise: mean + k * std.

    nominal_scores: array-like of nominal per-step surprise values (any shape; it
    is flattened). Returns a float threshold. This is the standard k-sigma novelty
    threshold; calibrating on held-out nominal data fixes the nominal false-alarm
    rate instead of eyeballing a single trace.
    """
    flat = np.asarray(nominal_scores, dtype=float).ravel()
    if flat.size == 0:
        return float("nan")
    return float(flat.mean() + k * flat.std())


def _alarm_edges(scores, threshold, min_consecutive=1):
    """Indices where a fresh alarm fires (rising edge above threshold).

    An alarm fires at the step that *completes* a run of >= min_consecutive
    consecutive above-threshold samples; each such run is counted once. Returns a
    list of (absolute) score indices at which alarms fire.
    """
    s = np.asarray(scores, dtype=float).ravel()
    above = s > threshold
    edges = []
    run = 0
    fired = False
    for i, hot in enumerate(above):
        if hot:
            run += 1
            if run >= min_consecutive and not fired:
                edges.append(i)
                fired = True
        else:
            run = 0
            fired = False
    return edges


def detection_delay(scores, threshold, onset_idx, min_consecutive=1):
    """Steps from fault onset to first alarm at/after onset, or None if never.

    scores: per-step surprise for one faulted episode. onset_idx is the index into
    `scores` at which the fault becomes active (fault_step - history_size). Returns
    the integer step delay (>= 0) of the first qualifying alarm, or None if the
    fault is never detected.
    """
    for i in _alarm_edges(scores, threshold, min_consecutive=min_consecutive):
        if i >= onset_idx:
            return int(i - onset_idx)
    return None


def false_alarm_count(scores, threshold, min_consecutive=1):
    """Number of distinct alarm events on a nominal (fault-free) episode."""
    return len(_alarm_edges(scores, threshold, min_consecutive=min_consecutive))
