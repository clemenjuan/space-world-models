import math


def test_orekit_bootstrap_idempotent():
    from envs.orekit_setup import ensure_orekit
    ensure_orekit()
    ensure_orekit()  # second call must be a no-op, not re-init the JVM
    from org.orekit.utils import Constants
    assert abs(Constants.EIGEN5C_EARTH_MU - 3.986004415e14) < 1e6


import numpy as np


def test_state_shape_and_propagation():
    from envs.od_env import OdEnv
    env = OdEnv()
    obs, info = env.reset(seed=0)
    assert info["state"].shape == (6,)
    r = np.linalg.norm(info["state"][:3])
    v = np.linalg.norm(info["state"][3:])
    assert 6.7e6 < r < 6.9e6
    assert 7.4e3 < v < 7.8e3
    s0 = info["state"].copy()
    _, _, _, _, info2 = env.step(np.zeros(3, dtype=np.float32))
    assert not np.allclose(s0, info2["state"])


def test_measurement_ranges_and_noise():
    from envs.od_env import OdEnv

    env = OdEnv(noise_std=(0.0, 0.0, 0.0, 0.0))
    obs, _ = env.reset(seed=1)
    rng, az, el, rr = obs
    assert 3e5 < rng < 5e7
    assert -math.pi <= az <= math.pi
    assert -math.pi / 2 <= el <= math.pi / 2
    assert abs(rr) < 1e4

    env_noisy = OdEnv(noise_std=(10.0, 0.01, 0.01, 0.1))
    obs_n, _ = env_noisy.reset(seed=1)
    assert not np.allclose(obs, obs_n)


def test_rollout_200_steps_shapes():
    from envs.od_env import OdEnv
    env = OdEnv()
    obs, info = env.reset(seed=2)
    obs_log, state_log = [obs], [info["state"]]
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        obs_log.append(obs)
        state_log.append(info["state"])
    obs_arr = np.asarray(obs_log)
    state_arr = np.asarray(state_log)
    assert obs_arr.shape == (201, 4)
    assert state_arr.shape == (201, 6)
    assert np.isfinite(obs_arr).all()
    assert np.isfinite(state_arr).all()
