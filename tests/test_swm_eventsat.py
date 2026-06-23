from __future__ import annotations

import numpy as np


def _write_dataset(path):
    from swm_eventsat.schema import AUTOPS_STATE_NAMES

    E, T = 2, 5
    obs = np.random.default_rng(0).random((E, T, 25), dtype=np.float32)
    action = np.zeros((E, T, 11), dtype=np.float32)
    action[..., 0] = 1.0
    action[..., 7] = 1.0
    action[..., 9] = 1.0
    state = np.zeros((E, T, len(AUTOPS_STATE_NAMES)), dtype=np.float32)
    ix = {name: i for i, name in enumerate(AUTOPS_STATE_NAMES)}
    state[..., ix["battery_soc"]] = 0.7
    state[..., ix["storage_capacity_mb"]] = 4096.0
    state[..., ix["jetson_capacity_mb"]] = 249036.8
    state[..., ix["daily_downlink_budget_mb"]] = 27.0
    state[..., ix["health_nominal"]] = 1.0
    state[..., ix["data_downlinked_mb"]] = np.linspace(0, 1, T)
    state[..., ix["total_observation_s"]] = np.linspace(0, 240, T)
    reward = np.zeros((E, T), dtype=np.float32)
    mode = np.zeros((E, T), dtype=np.int64)
    np.savez_compressed(
        path,
        obs=obs,
        action=action,
        state=state,
        reward=reward,
        mode=mode,
        resolved_mode=mode,
        forced_mode=np.zeros((E, T), dtype=np.float32),
        episode_seed=np.asarray([1, 2], dtype=np.int64),
    )


def test_world_model_dataset_and_probe_targets(tmp_path):
    from swm_eventsat.linear_probes import build_attribute_targets, fit_ridge_probe
    from swm_eventsat.schema import load_world_model_dataset

    path = tmp_path / "eventsat_world_model_v1.npz"
    _write_dataset(path)
    ds = load_world_model_dataset(path)
    assert ds.obs.shape == (2, 5, 25)
    assert ds.action.shape == (2, 5, 11)
    assert ds.dataset_steps == 10

    targets = build_attribute_targets(ds)
    assert targets.shape[:2] == ds.obs.shape[:2]
    assert targets.shape[-1] == 8
    fit = fit_ridge_probe(ds.obs, targets, ridge=1e-2)
    assert fit.W.shape == (8, 25)
    assert fit.b.shape == (8,)
    assert "battery_margin" in fit.rmse


class FakeLatentModel:
    def rollout(self, history, action11):
        mode_signal = action11[..., :7] @ np.arange(7, dtype=np.float32)
        z0 = mode_signal / 6.0
        z1 = np.cumsum(action11[..., 0], axis=1)
        z2 = np.cumsum(action11[..., 2], axis=1)
        return np.stack([z0, z1, z2], axis=-1).astype(np.float32)


def test_cem_respects_first_mask():
    from swm_eventsat.planners import CEMPlanner, default_mode_weights

    W = np.zeros((8, 3), dtype=np.float32)
    W[0, 1] = 1.0
    W[3, 2] = 1.0
    b = np.zeros(8, dtype=np.float32)
    weights = default_mode_weights("science")
    mask = np.ones(7, dtype=bool)
    mask[1] = False
    history = {"obs": np.zeros((4, 25), dtype=np.float32), "action": np.zeros((4, 11), dtype=np.float32)}

    cem = CEMPlanner(FakeLatentModel(), W, b, weights, horizon=4, samples=32, elites=4, iterations=2)
    result = cem.select_action(history, first_mask=mask)
    assert result.mode_index != 1
    assert result.best_sequence.shape == (4,)
    assert result.diagnostics["candidate_count"] == 32.0
