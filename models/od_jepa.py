"""ODJEPA: LeWM's JEPA with encode() adapted for 4-dim vector observations.

Only encode() changes (pixels -> obs vector + MLP encoder). predict(), the predictor,
action encoder, projector, pred_proj, and SIGReg are inherited / used as-is.
"""
from einops import rearrange

from jepa import JEPA


class ODJEPA(JEPA):
    def encode(self, info):
        obs = info["obs"].float()  # (B, T, 4)
        b = obs.size(0)
        flat = rearrange(obs, "b t d -> (b t) d")
        emb = self.encoder(flat.unsqueeze(1)).squeeze(1)  # OdEncoder accepts (N,1,4)->(N,1,D)
        emb = self.projector(emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info
