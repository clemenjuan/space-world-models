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
