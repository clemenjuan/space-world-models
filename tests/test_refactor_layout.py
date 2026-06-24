from __future__ import annotations

import json

import numpy as np


def test_core_and_swm_eventsat_canonical_imports():
    from core.data.window_dataset import WindowedTrajectoryDataset
    from core.models.lewm_loss import lewm_forward
    from core.models.vector_encoder import VectorEncoder
    from core.models.vector_jepa import VectorJEPA
    from swm_eventsat.data.eventsat_lite_env import EventSatLiteEnv
    from swm_eventsat.data.toy_eventsat_env import EventSatEnv

    assert WindowedTrajectoryDataset is not None
    assert lewm_forward is not None
    assert VectorEncoder is not None
    assert VectorJEPA is not None
    assert EventSatEnv is not None
    assert EventSatLiteEnv is not None


def test_schema_roundtrip(tmp_path):
    from swm_eventsat.schema import TrajectoryBatch, load_trajectory_npz, save_trajectory_npz

    batch = TrajectoryBatch(
        obs=np.ones((2, 3, 25), dtype=np.float32),
        action=np.zeros((2, 3, 7), dtype=np.float32),
        state=np.zeros((2, 3, 16), dtype=np.float32),
        reward=np.zeros((2, 3), dtype=np.float32),
        mode=np.zeros((2, 3), dtype=np.int64),
        metadata={"source": "unit-test"},
    )
    batch.action[..., 0] = 1.0
    path = save_trajectory_npz(tmp_path / "traj.npz", batch)
    loaded = load_trajectory_npz(path)

    assert loaded.obs.shape == (2, 3, 25)
    assert loaded.action.shape == (2, 3, 7)
    assert loaded.metadata["source"] == "unit-test"


def test_linear_probe_recovers_synthetic_mapping():
    from swm_eventsat.models.probes import fit_linear_probe

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    weight = np.asarray([[2.0, -1.0, 0.5, 0.0], [0.0, 1.0, 0.0, -3.0]], dtype=np.float32)
    bias = np.asarray([0.25, -0.75], dtype=np.float32)
    y = x @ weight.T + bias
    probe = fit_linear_probe(x, y, ridge=1e-8, target_names=("a", "b"))

    assert probe.target_names == ("a", "b")
    assert np.allclose(probe.predict(x), y, atol=1e-4)


def test_cem_respects_mask_and_improves_score():
    from swm_eventsat.planning.cem import CEMConfig, categorical_cem

    def score_fn(sequences: np.ndarray) -> np.ndarray:
        return (sequences == 2).sum(axis=1).astype(np.float32)

    mask = np.asarray([True, False, True, False])
    result = categorical_cem(
        score_fn,
        CEMConfig(horizon=5, action_dim=4, population=64, elite_frac=0.25, iterations=4, smoothing=0.8),
        rng=np.random.default_rng(0),
        action_mask=mask,
    )

    assert set(result.action_sequence.tolist()).issubset({0, 2})
    assert result.score >= 4.0
    assert np.allclose(result.probabilities[:, ~mask], 0.0)


def test_autops_results_writer_shape(tmp_path):
    from swm_eventsat.experiments.write_autops_results import write_minimal_results

    path = write_minimal_results(
        tmp_path / "eventsat_lewm_mpc",
        experiment_id="eventsat_lewm_mpc",
        episode_metrics=[
            {"utility": 1.5, "policy_loaded": 1.0},
            {"utility": 2.5, "policy_loaded": 1.0},
        ],
        config={"planner": "lewm_cem"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "eventsat_lewm_mpc"
    assert payload["experiment_statistics"]["mean"]["utility"] == 2.0
    assert len(payload["episodes"]) == 2
