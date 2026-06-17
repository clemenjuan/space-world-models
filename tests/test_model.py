import torch


def test_od_encoder_shape():
    from models.od_encoder import OdEncoder
    enc = OdEncoder(in_dim=4, hidden_dim=256, out_dim=192)
    x = torch.randn(2, 5, 4)  # (B, T, 4)
    z = enc(x)
    assert z.shape == (2, 5, 192)
    assert torch.isfinite(z).all()


def _make_odjepa(embed_dim=192, history=3):
    from models.od_jepa import ODJEPA
    from models.od_encoder import OdEncoder
    from module import ARPredictor, Embedder, MLP
    encoder = OdEncoder(4, 256, embed_dim)
    predictor = ARPredictor(
        num_frames=history, input_dim=embed_dim, hidden_dim=embed_dim,
        output_dim=embed_dim, depth=2, heads=4, mlp_dim=256, dim_head=48, dropout=0.0,
    )
    action_encoder = Embedder(input_dim=3, smoothed_dim=3, emb_dim=embed_dim)
    projector = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    return ODJEPA(encoder, predictor, action_encoder, projector, pred_proj)


def test_odjepa_encode_predict_shapes():
    model = _make_odjepa()
    batch = {"obs": torch.randn(2, 3, 4), "action": torch.randn(2, 3, 3)}
    out = model.encode(batch)
    assert out["emb"].shape == (2, 3, 192)
    assert out["act_emb"].shape == (2, 3, 192)
    preds = model.predict(out["emb"], out["act_emb"])
    assert preds.shape == (2, 3, 192)
