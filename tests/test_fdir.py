"""FDIR environment + (later) surprise-detection tests.

This file is accreted across the Step-2 tasks; Task 1 adds the env-contract test.
"""
import numpy as np
import gymnasium as gym
import torch


def test_fdir_env_contract():
    from envs.fdir_env import FdirEnv, CHANNEL_INDEX

    # --- Rollout under nominal dynamics: shapes, finiteness, spaces. ---
    env = FdirEnv(fault_mode=None)
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    assert np.isfinite(obs).all()
    assert info["state"].shape == (8,)
    for _ in range(160):
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (8,)
        assert np.isfinite(obs).all()
        assert info["state"].shape == (8,)

    assert env.observation_space.shape == (8,)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

    # --- Stability: spectral radius of the dynamics matrix A must be < 1. ---
    A = env.A
    eigvals = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals)))
    assert spectral_radius < 1.0, f"spectral radius {spectral_radius} >= 1"

    # --- Determinism + fault divergence: same seed, nominal vs stuck_at fault. ---
    n_steps = 200
    fault_step = 100
    faulted_ch = CHANNEL_INDEX["solar_array_voltage"]
    coupled_ch = CHANNEL_INDEX["panel_temp"]  # solar_array_voltage -> panel_temp

    nom = FdirEnv(fault_mode=None)
    flt = FdirEnv(
        fault_mode="stuck_at",
        fault_channel="solar_array_voltage",
        fault_step=fault_step,
    )
    _, nom_info = nom.reset(seed=42)
    _, flt_info = flt.reset(seed=42)
    nom_states = [nom_info["state"].copy()]
    flt_states = [flt_info["state"].copy()]
    for _ in range(n_steps):
        _, _, _, _, nom_info = nom.step(0)
        _, _, _, _, flt_info = flt.step(0)
        nom_states.append(nom_info["state"].copy())
        flt_states.append(flt_info["state"].copy())

    nom_arr = np.asarray(nom_states)  # (n_steps+1, 8)
    flt_arr = np.asarray(flt_states)

    diff = np.abs(flt_arr - nom_arr)
    pre = slice(0, fault_step)            # indices 0..fault_step-1 (pre-onset)
    post = slice(fault_step + 1, n_steps + 1)  # strictly after onset

    # Same seed => the two state trajectories match exactly before the fault: the
    # pre-fault max difference is the "noise floor" the post-fault divergence must
    # clearly exceed (both runs consume identical randomness until the fault fires).
    floor = float(diff[pre].max())
    assert floor < 1e-4, "pre-fault trajectories should match under the same seed"

    # The faulted channel must diverge materially after onset (stuck value frozen
    # while nominal keeps fluctuating) -- well above the pre-fault floor.
    assert diff[post, faulted_ch].max() > floor + 0.1, \
        "faulted channel did not diverge from nominal after the fault"
    # ...and the divergence must propagate through coupling to panel_temp.
    assert diff[post, coupled_ch].max() > floor + 0.02, \
        "coupled channel did not diverge from nominal after the fault"


def _make_fdir_odjepa(embed_dim=192, history=3):
    """Tiny ODJEPA for FDIR: in_dim=8, action input_dim=4 (mirrors
    tests/test_model._make_odjepa)."""
    from models.od_jepa import ODJEPA
    from models.od_encoder import OdEncoder
    from module import ARPredictor, Embedder, MLP
    encoder = OdEncoder(8, 256, embed_dim)
    predictor = ARPredictor(
        num_frames=history, input_dim=embed_dim, hidden_dim=embed_dim,
        output_dim=embed_dim, depth=2, heads=4, mlp_dim=256, dim_head=48, dropout=0.0,
    )
    action_encoder = Embedder(input_dim=4, smoothed_dim=4, emb_dim=embed_dim)
    projector = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    return ODJEPA(encoder, predictor, action_encoder, projector, pred_proj)


def test_surprise_shapes():
    from models.surprise import surprise_score

    torch.manual_seed(0)
    model = _make_fdir_odjepa()

    T, history_size = 10, 3
    obs_seq = torch.randn(1, T, 8)
    # valid one-hot action sequence: nominal action 0 -> [1, 0, 0, 0] each step.
    action_seq = torch.zeros(1, T, 4)
    action_seq[..., 0] = 1.0

    scores = surprise_score(model, obs_seq, action_seq, history_size=history_size)
    assert scores.shape == (T - history_size,)
    assert torch.isfinite(scores).all()
