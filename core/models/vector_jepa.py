"""Vector-observation JEPA used as the LeWM transition model."""

from einops import rearrange

from core.models.jepa import JEPA


class VectorJEPA(JEPA):
    """JEPA variant that reads vector observations from ``info["obs"]``."""

    def encode(self, info):
        obs = info["obs"].float()
        batch_size = obs.size(0)
        flat = rearrange(obs, "b t d -> (b t) d")
        emb = self.encoder(flat.unsqueeze(1)).squeeze(1)
        emb = self.projector(emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch_size)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info


__all__ = ["VectorJEPA"]

