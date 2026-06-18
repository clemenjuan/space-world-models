"""Gymnasium orbit-determination environment.

Hidden state: 6-dim ECI [r, v]. Observation: [range, az, el, range_rate] from one
ground station. Action: 3-dim LVLH acceleration (zero in Step 1; plumbed, ignored).
Ground-truth dynamics: Orekit EcksteinHechlerPropagator (J2-J6 zonal) -> sun-synchronous.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.orekit_setup import ensure_orekit


class OdEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=30.0,
        station_lat_deg=48.15,
        station_lon_deg=11.58,
        station_alt_m=500.0,
        sma_m=6778137.0,
        ecc=1e-3,
        inc_deg=97.0,
        noise_std=(10.0, 0.01, 0.01, 0.1),  # range[m], az[rad], el[rad], range_rate[m/s]
        max_steps=1000,
    ):
        super().__init__()
        ensure_orekit()
        self.dt = float(dt)
        self.sma_m = float(sma_m)
        self.ecc = float(ecc)
        self.inc = math.radians(inc_deg)
        self.noise_std = np.asarray(noise_std, dtype=np.float64)
        self.max_steps = int(max_steps)

        from org.orekit.frames import FramesFactory, TopocentricFrame
        from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
        from org.orekit.utils import Constants, IERSConventions

        self._eci = FramesFactory.getEME2000()
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self._mu = Constants.EIGEN5C_EARTH_MU
        self._Re = Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS
        self._C = (
            Constants.EIGEN5C_EARTH_C20,
            Constants.EIGEN5C_EARTH_C30,
            Constants.EIGEN5C_EARTH_C40,
            Constants.EIGEN5C_EARTH_C50,
            Constants.EIGEN5C_EARTH_C60,
        )
        earth = OneAxisEllipsoid(self._Re, Constants.WGS84_EARTH_FLATTENING, itrf)
        self._topo = TopocentricFrame(
            earth,
            GeodeticPoint(
                math.radians(station_lat_deg), math.radians(station_lon_deg), station_alt_m
            ),
            "station",
        )

        high = np.array([1e8, np.pi, np.pi / 2, 1e5], dtype=np.float32)
        low = np.array([0.0, -np.pi, -np.pi / 2, -1e5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._prop = None
        self._t0 = None
        self._step = 0

    def _build_propagator(self, raan, argp, nu):
        from org.orekit.orbits import KeplerianOrbit, PositionAngleType
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.propagation.analytical import EcksteinHechlerPropagator

        self._t0 = AbsoluteDate(2026, 6, 17, 0, 0, 0.0, TimeScalesFactory.getUTC())
        orbit = KeplerianOrbit(
            self.sma_m, self.ecc, self.inc, float(argp), float(raan), float(nu),
            PositionAngleType.TRUE, self._eci, self._t0, self._mu,
        )
        self._prop = EcksteinHechlerPropagator(orbit, self._Re, self._mu, *self._C)

    def _state_at(self, step):
        st = self._prop.propagate(self._t0.shiftedBy(self.dt * step))
        pv = st.getPVCoordinates(self._eci)
        p, v = pv.getPosition(), pv.getVelocity()
        return st, np.array(
            [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()], dtype=np.float64
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raan = self.np_random.uniform(0, 2 * math.pi)
        argp = self.np_random.uniform(0, 2 * math.pi)
        nu = self.np_random.uniform(0, 2 * math.pi)
        self._build_propagator(raan, argp, nu)
        self._step = 0
        st, state = self._state_at(0)
        obs = self._measure(st)
        return obs, {"state": state, "geometry": self._geometry(st)}

    def step(self, action):
        self._step += 1
        st, state = self._state_at(self._step)
        obs = self._measure(st)
        terminated = False
        truncated = self._step >= self.max_steps
        return obs, 0.0, terminated, truncated, {"state": state, "geometry": self._geometry(st)}

    def _station_pv(self, st):
        from org.orekit.utils import PVCoordinates

        return self._topo.getTransformTo(self._eci, st.getDate()).transformPVCoordinates(
            PVCoordinates.ZERO
        )

    def _geometry(self, st):
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        sta = self._station_pv(st)
        p = sta.getPosition()
        v = sta.getVelocity()
        transform = self._topo.getTransformTo(self._eci, st.getDate())
        axes = []
        for local_axis in (Vector3D.PLUS_I, Vector3D.PLUS_J, Vector3D.PLUS_K):
            axis = transform.transformVector(local_axis)
            axes.append([axis.getX(), axis.getY(), axis.getZ()])
        return {
            "time_s": np.asarray([self.dt * self._step], dtype=np.float64),
            "station_state_eci": np.asarray(
                [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()],
                dtype=np.float64,
            ),
            "topocentric_basis_eci": np.asarray(axes, dtype=np.float64),
        }

    def _measure(self, st):
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        date = st.getDate()
        sat = st.getPVCoordinates(self._eci)
        pos = sat.getPosition()
        sta = self._station_pv(st)
        rel_p = sat.getPosition().subtract(sta.getPosition())
        rel_v = sat.getVelocity().subtract(sta.getVelocity())
        rng = self._topo.getRange(pos, self._eci, date)
        az = self._topo.getAzimuth(pos, self._eci, date)
        el = self._topo.getElevation(pos, self._eci, date)
        range_rate = Vector3D.dotProduct(rel_p, rel_v) / rel_p.getNorm()
        # wrap azimuth from [0, 2pi) to [-pi, pi] to match observation_space
        if az > math.pi:
            az -= 2 * math.pi
        clean = np.array([rng, az, el, range_rate], dtype=np.float64)
        noisy = clean + self.np_random.normal(0.0, self.noise_std)
        return noisy.astype(np.float32)
