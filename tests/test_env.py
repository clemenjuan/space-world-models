import math


def test_orekit_bootstrap_idempotent():
    from envs.orekit_setup import ensure_orekit
    ensure_orekit()
    ensure_orekit()  # second call must be a no-op, not re-init the JVM
    from org.orekit.utils import Constants
    assert abs(Constants.EIGEN5C_EARTH_MU - 3.986004415e14) < 1e6
