import numpy as np
import gymnasium as gym
import torch


def _make_eventsat_odjepa(embed_dim=64, history=3):
    from models.od_jepa import ODJEPA
    from models.od_encoder import OdEncoder
    from module import ARPredictor, Embedder, MLP

    encoder = OdEncoder(25, 128, embed_dim)
    predictor = ARPredictor(
        num_frames=history,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=2,
        heads=4,
        mlp_dim=128,
        dim_head=32,
        dropout=0.0,
    )
    action_encoder = Embedder(input_dim=7, smoothed_dim=7, emb_dim=embed_dim)
    projector = MLP(embed_dim, 128, embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, 128, embed_dim, norm_fn=None)
    return ODJEPA(encoder, predictor, action_encoder, projector, pred_proj)


def test_eventsat_env_contract_and_windows():
    from envs.eventsat_env import EventSatEnv, MODE_TO_INDEX

    env = EventSatEnv(max_steps=220)
    obs, info = env.reset(seed=0)
    assert obs.shape == (25,)
    assert np.isfinite(obs).all()
    assert info["state"].shape == (16,)
    assert env.observation_space.shape == (25,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 7

    saw_pass = bool(info["ground_pass_active"])
    saw_eclipse = not bool(info["in_sunlight"])
    truncated = False
    for _ in range(220):
        obs, reward, terminated, truncated, info = env.step(MODE_TO_INDEX["charging"])
        assert obs.shape == (25,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert not terminated
        assert 0.0 <= info["state"][0] <= 1.0
        assert (info["state"][1:5] >= 0.0).all()
        saw_pass = saw_pass or bool(info["ground_pass_active"])
        saw_eclipse = saw_eclipse or not bool(info["in_sunlight"])

    assert saw_pass
    assert saw_eclipse
    assert truncated


def test_eventsat_payload_pipeline_scripted():
    from envs.eventsat_env import EventSatEnv, MODE_TO_INDEX

    env = EventSatEnv(max_steps=64, randomize_phase=False)
    env.reset(seed=1)

    env.step(MODE_TO_INDEX["payload_observe"])
    assert env.uncompressed_observations == 1
    assert env.jetson_raw_mb > 9.0

    env.step(MODE_TO_INDEX["payload_compress"])
    assert env.compression_progress == 1
    env.step(MODE_TO_INDEX["payload_compress"])
    assert env.uncompressed_observations == 0
    assert env.undetected_observations == 1
    assert env.jetson_compressed_mb > 1.0

    for _ in range(env.detection_steps):
        env.step(MODE_TO_INDEX["payload_detect"])
    assert env.undetected_observations == 0
    assert env.total_detections == 1
    assert env.obc_data_mb >= env.detection_metadata_mb

    before = env.obc_data_mb
    env.step(MODE_TO_INDEX["payload_send"])
    assert env.jetson_compressed_mb == 0.0
    assert env.obc_data_mb > before


def test_eventsat_generator_dataset_and_model_forward(tmp_path):
    from data.generate_eventsat import generate
    from od_datasets.od_dataset import OdWindowDataset, fit_normalizers

    path = tmp_path / "eventsat.npz"
    generate(n_episodes=2, episode_len=48, out_path=str(path), seed=0, exploration=0.0)
    blob = np.load(path)
    obs, action, state = blob["obs"], blob["action"], blob["state"]
    assert obs.shape == (2, 48, 25)
    assert action.shape == (2, 48, 7)
    assert state.shape == (2, 48, 16)
    assert np.allclose(action.sum(axis=-1), 1.0)
    assert np.isfinite(obs).all()
    assert np.isfinite(state).all()

    ds = OdWindowDataset(str(path), window=4, normalizers=fit_normalizers(str(path)))
    item = ds[0]
    assert item["obs"].shape == (4, 25)
    assert item["action"].shape == (4, 7)

    torch.manual_seed(0)
    model = _make_eventsat_odjepa()
    batch = {
        "obs": item["obs"].unsqueeze(0).float(),
        "action": item["action"].unsqueeze(0).float(),
    }
    out = model.encode(batch)
    pred = model.predict(out["emb"][:, :3], out["act_emb"][:, :3])
    assert out["emb"].shape == (1, 4, 64)
    assert out["act_emb"].shape == (1, 4, 64)
    assert pred.shape == (1, 3, 64)
    assert torch.isfinite(pred).all()
