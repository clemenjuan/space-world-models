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



def test_eventsat_decoder_forward_shape():
    from scripts.eventsat_world_model_utils import EventSatStateDecoder

    decoder = EventSatStateDecoder(input_dim=64, hidden_dim=32, depth=1, output_dim=18)
    out = decoder(torch.randn(5, 64))
    assert out.shape == (5, 18)
    assert torch.isfinite(out).all()


def test_eventsat_action_trace_loader_json(tmp_path):
    import json

    from scripts.evaluate_eventsat_action_trace import load_actions

    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"actions": ["charging", "payload_observe", "communication"]}))
    actions, source = load_actions(path)
    assert source.endswith("actions.json")
    assert actions.tolist() == [0, 2, 1]


def test_eventsat_mpc_candidates_respect_first_action_safety():
    from envs.eventsat_env import EventSatEnv, MODE_TO_INDEX
    from scripts.run_eventsat_lewm_mpc import generate_candidate_sequences, safe_first_action_mask

    env = EventSatEnv(max_steps=32, randomize_phase=False)
    env.reset(seed=0)
    mask = safe_first_action_mask(env)
    assert mask[MODE_TO_INDEX["charging"]]
    assert not mask[MODE_TO_INDEX["communication"]]

    candidates = generate_candidate_sequences(
        env,
        horizon=4,
        n_random=16,
        rng=np.random.default_rng(0),
    )
    assert candidates.shape[1] == 4
    allowed = set(np.flatnonzero(mask).tolist())
    assert set(candidates[:, 0].tolist()).issubset(allowed)


def test_eventsat_board_html_accepts_missing_artifacts():
    from scripts.build_eventsat_results_board import _html

    html = _html(
        dataset={"ok": False, "message": "missing"},
        runs=[],
        week={"ok": False, "message": "missing"},
        decoder={"ok": False, "message": "missing"},
        trace={"ok": False, "message": "missing"},
        mpc={"ok": False, "message": "missing"},
    )
    assert "Decoder Quality" in html
    assert "Fixed Trace Evaluator" in html
    assert "Controller Comparison" in html
