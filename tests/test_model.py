import torch


def test_od_encoder_shape():
    from models.od_encoder import OdEncoder
    enc = OdEncoder(in_dim=4, hidden_dim=256, out_dim=192)
    x = torch.randn(2, 5, 4)  # (B, T, 4)
    z = enc(x)
    assert z.shape == (2, 5, 192)
    assert torch.isfinite(z).all()
