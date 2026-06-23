"""LeWM / LeJEPA loss for vector-observation trajectories."""

import torch


def lewm_forward(model, sigreg, batch, cfg):
    ctx_len = cfg["history_size"]
    n_preds = cfg["num_preds"]
    lambd = cfg["sigreg_weight"]

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = model.encode(batch)
    emb = out["emb"]
    act_emb = out["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    m = min(pred_emb.size(1), tgt_emb.size(1))
    out["pred_loss"] = (pred_emb[:, :m] - tgt_emb[:, :m]).pow(2).mean()
    out["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    out["loss"] = out["pred_loss"] + lambd * out["sigreg_loss"]
    return out


__all__ = ["lewm_forward"]
