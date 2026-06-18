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
    assert info["geometry"]["time_s"].shape == (1,)
    assert info["geometry"]["station_state_eci"].shape == (6,)
    assert info["geometry"]["topocentric_basis_eci"].shape == (3, 3)
    assert float(info["geometry"]["time_s"][0]) == 0.0
    station_r = np.linalg.norm(info["geometry"]["station_state_eci"][:3])
    assert 6.3e6 < station_r < 6.5e6
    r = np.linalg.norm(info["state"][:3])
    v = np.linalg.norm(info["state"][3:])
    assert 6.7e6 < r < 6.9e6
    assert 7.4e3 < v < 7.8e3
    s0 = info["state"].copy()
    _, _, _, _, info2 = env.step(np.zeros(3, dtype=np.float32))
    assert not np.allclose(s0, info2["state"])
    assert float(info2["geometry"]["time_s"][0]) == env.dt


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


def _energy_hz(state, mu, Re, J2):
    r = np.linalg.norm(state[:3])
    v = np.linalg.norm(state[3:])
    sin_phi = state[2] / r
    P2 = (3 * sin_phi ** 2 - 1) / 2
    U = mu / r * (1 - J2 * (Re / r) ** 2 * P2)
    energy = 0.5 * v ** 2 - U
    hz = state[0] * state[4] - state[1] * state[3]
    return energy, hz


def test_invariants_bounded():
    from envs.orekit_setup import ensure_orekit
    ensure_orekit()
    from envs.od_env import OdEnv
    from org.orekit.utils import Constants
    mu = Constants.EIGEN5C_EARTH_MU
    Re = Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS
    J2 = -Constants.EIGEN5C_EARTH_C20

    env = OdEnv(noise_std=(0.0, 0.0, 0.0, 0.0))
    _, info = env.reset(seed=3)
    e0, h0 = _energy_hz(info["state"], mu, Re, J2)
    de, dh = [], []
    for _ in range(200):
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        e, h = _energy_hz(info["state"], mu, Re, J2)
        de.append(abs(e - e0) / abs(e0))
        dh.append(abs(h - h0) / abs(h0))
    assert max(de) < 1e-3   # measured ~4.5e-5
    assert max(dh) < 1e-3   # measured ~2.2e-5


def test_sso_raan_precession():
    """RAAN should precess ~0.9856 deg/day for a sun-synchronous orbit."""
    from envs.od_env import OdEnv
    env = OdEnv(noise_std=(0.0, 0.0, 0.0, 0.0))
    env.reset(seed=4)
    from org.orekit.orbits import KeplerianOrbit

    def raan_deg(step):
        st, _ = env._state_at(step)
        return math.degrees(KeplerianOrbit(st.getOrbit()).getRightAscensionOfAscendingNode())

    steps_per_day = int(round(86400.0 / env.dt))
    rate = raan_deg(steps_per_day) - raan_deg(0)
    assert 0.95 < rate < 1.02  # measured ~0.981 deg/day
