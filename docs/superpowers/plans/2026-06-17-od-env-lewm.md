# OD Environment wired to LeWM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gymnasium orbit-determination environment (Orekit J2 dynamics, ground-station measurements) and wire its 4-dim vector observations into LeWM's JEPA encoder/predictor + SIGReg loss, with a complete Lightning training loop and a smoke test suite.

**Architecture:** Fork the le-wm baseline (`module.py`, `jepa.py` reused verbatim/subclassed). Replace the ViT encoder with an MLP for 4-dim obs and override `JEPA.encode()`. Orekit's `EcksteinHechlerPropagator` (J2–J6 zonal) provides ground-truth sun-synchronous dynamics; a `TopocentricFrame` provides range/az/el/range-rate. Training mirrors `train.py` via `stable_pretraining.Module` + Lightning, logging to W&B (`sps-tum`).

**Tech Stack:** Python 3.12, PyTorch, Gymnasium, `orekit_jpype` 13.1.5, `einops`, `stable_pretraining` 0.1.7, `stable_worldmodel` 0.1.1, Lightning, Hydra/OmegaConf, Weights & Biases.

**Spec:** `docs/superpowers/specs/2026-06-17-od-env-lewm-design.md`

**Verified facts (probed live before writing this plan):**
- `Constants.EIGEN5C_EARTH_MU = 3.986004415e14`, `EQUATORIAL_RADIUS = 6378136.46`, `J2 = -EIGEN5C_EARTH_C20 = 1.0826e-3`.
- `EcksteinHechlerPropagator(orbit, Re, mu, C20, C30, C40, C50, C60)` constructor works.
- `TopocentricFrame.getRange/getAzimuth/getElevation(pos, frame, date)` work; range-rate via relative PV (station PV from `topo.getTransformTo(eci, date).transformPVCoordinates(PVCoordinates.ZERO)`).
- Over a 200-step (30 s) SSO@400 km rollout: max rel ΔE = 4.5e-5, max rel Δh_z = 2.2e-5, RAAN rate = 0.981°/day.

---

## File Structure

```
envs/orekit_setup.py      # JVM + orekit-data bootstrap (idempotent, process-global)
envs/od_env.py            # OdEnv(gymnasium.Env): dynamics + measurement + spaces
models/od_encoder.py      # OdEncoder: MLP 4 -> 256 -> 192
models/od_jepa.py         # ODJEPA(JEPA): encode() override for vector obs
models/od_forward.py      # od_lejepa_forward(): pred MSE + 0.09*SIGReg
data/generate_dataset.py  # roll out episodes -> data/cache/od_trajectories.npz
od_datasets/od_dataset.py # OdWindowDataset + fit_normalizers + build_datamodule
train_od.py               # hydra entry: full Lightning training loop + W&B
module.py                 # VENDORED from le-wm, verbatim
jepa.py                   # VENDORED from le-wm, verbatim
config/train/od.yaml          # top-level training config
config/train/model/od.yaml    # model config (hydra instantiate -> ODJEPA)
config/train/data/od.yaml     # dataset config
tests/conftest.py         # shared fixtures (env factory)
tests/test_env.py         # rollout/shape + invariants + RAAN precession
tests/test_model.py       # encoder/jepa/forward shapes + finite losses
tests/test_train_smoke.py # fast_dev_run end-to-end
requirements.txt
.gitignore                # already created
```

---

## Task 0: Project setup & vendored baseline

**Files:**
- Create: `requirements.txt`
- Create: `module.py`, `jepa.py` (vendored from le-wm)
- Create: `__init__.py` in `envs/`, `models/`, `datasets/`, `data/`, `tests/`

- [ ] **Step 1: Write requirements.txt**

```
torch>=2.2
numpy>=1.26
gymnasium>=0.29
einops>=0.7
orekit_jpype>=13.1
stable_pretraining>=0.1.7
stable_worldmodel>=0.1.1
lightning>=2.2
hydra-core>=1.3
omegaconf>=2.3
wandb>=0.16
pytest>=8.0
```

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`
Expected: completes; `orekit_jpype`, `gymnasium`, `numpy` already present.

- [ ] **Step 3: Vendor baseline module.py and jepa.py**

Clone the baseline and copy the two files verbatim (they depend only on `torch`/`einops`):

```bash
git clone --depth 1 https://github.com/lucas-maes/le-wm /tmp/le-wm
cp /tmp/le-wm/module.py ./module.py
cp /tmp/le-wm/jepa.py ./jepa.py
```

- [ ] **Step 4: Create package __init__.py files**

Create empty `envs/__init__.py`, `models/__init__.py`, `datasets/__init__.py`, `data/__init__.py`, `tests/__init__.py`.

- [ ] **Step 5: Verify vendored imports**

Run: `python -c "import torch, einops; import module, jepa; print('SIGReg' in dir(module), 'JEPA' in dir(jepa))"`
Expected: `True True`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt module.py jepa.py envs/__init__.py models/__init__.py datasets/__init__.py data/__init__.py tests/__init__.py
git commit -m "chore: project setup and vendored LeWM baseline (module.py, jepa.py)"
```

---

## Task 1: Orekit bootstrap

**Files:**
- Create: `envs/orekit_setup.py`
- Test: `tests/test_env.py` (first test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import math


def test_orekit_bootstrap_idempotent():
    from envs.orekit_setup import ensure_orekit
    ensure_orekit()
    ensure_orekit()  # second call must be a no-op, not re-init the JVM
    from org.orekit.utils import Constants
    assert abs(Constants.EIGEN5C_EARTH_MU - 3.986004415e14) < 1e6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_orekit_bootstrap_idempotent -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.orekit_setup'`

- [ ] **Step 3: Write minimal implementation**

```python
# envs/orekit_setup.py
"""Idempotent, process-global Orekit JVM + data bootstrap."""
import os
import threading
from pathlib import Path

import orekit_jpype

# orekit-data.zip lives at repo root (gitignored, ~70 MB).
_DATA_ZIP = Path(__file__).resolve().parents[1] / "orekit-data.zip"
_lock = threading.Lock()
_ready = False


def ensure_orekit() -> None:
    """Start the JVM and load Orekit physical data exactly once per process."""
    global _ready
    with _lock:
        if _ready:
            return
        orekit_jpype.initVM()
        from orekit_jpype.pyhelpers import (
            download_orekit_data_curdir,
            setup_orekit_curdir,
        )
        if not _DATA_ZIP.exists():
            cwd = os.getcwd()
            os.chdir(_DATA_ZIP.parent)
            try:
                download_orekit_data_curdir()
            finally:
                os.chdir(cwd)
        setup_orekit_curdir(str(_DATA_ZIP))
        _ready = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env.py::test_orekit_bootstrap_idempotent -v`
Expected: PASS (orekit-data.zip already present at repo root).

- [ ] **Step 5: Commit**

```bash
git add envs/orekit_setup.py tests/test_env.py
git commit -m "feat: idempotent Orekit JVM/data bootstrap"
```

---

## Task 2: OdEnv — dynamics core (reset/step propagation)

**Files:**
- Create: `envs/od_env.py`
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py  (append)
import numpy as np


def test_state_shape_and_propagation():
    from envs.od_env import OdEnv
    env = OdEnv()
    obs, info = env.reset(seed=0)
    assert info["state"].shape == (6,)
    # LEO speed ~7.6 km/s, radius ~6778 km
    r = np.linalg.norm(info["state"][:3])
    v = np.linalg.norm(info["state"][3:])
    assert 6.7e6 < r < 6.9e6
    assert 7.4e3 < v < 7.8e3
    s0 = info["state"].copy()
    _, _, _, _, info2 = env.step(np.zeros(3, dtype=np.float32))
    # state must change after a 30 s propagation
    assert not np.allclose(s0, info2["state"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_state_shape_and_propagation -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'envs.od_env'`

- [ ] **Step 3: Write minimal implementation**

```python
# envs/od_env.py
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
        return obs, {"state": state}

    def step(self, action):
        self._step += 1
        st, state = self._state_at(self._step)
        obs = self._measure(st)
        terminated = False
        truncated = self._step >= self.max_steps
        return obs, 0.0, terminated, truncated, {"state": state}

    def _measure(self, st):
        raise NotImplementedError  # implemented in Task 3
```

- [ ] **Step 4: Add a temporary measurement stub so this task's test runs**

Replace the final `_measure` body with a stub returning zeros (Task 3 implements it):

```python
    def _measure(self, st):
        return np.zeros(4, dtype=np.float32)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_env.py::test_state_shape_and_propagation -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add envs/od_env.py tests/test_env.py
git commit -m "feat: OdEnv dynamics core (EcksteinHechler propagation, ECI state)"
```

---

## Task 3: OdEnv — measurement model

**Files:**
- Modify: `envs/od_env.py` (replace `_measure`)
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py  (append)
def test_measurement_ranges_and_noise():
    from envs.od_env import OdEnv

    # noiseless: elevation within [-pi/2, pi/2], range positive and plausible (LEO)
    env = OdEnv(noise_std=(0.0, 0.0, 0.0, 0.0))
    obs, _ = env.reset(seed=1)
    rng, az, el, rr = obs
    assert 3e5 < rng < 5e7
    assert -math.pi <= az <= math.pi
    assert -math.pi / 2 <= el <= math.pi / 2
    assert abs(rr) < 1e4

    # noise changes the observation for the same seed/state
    env_noisy = OdEnv(noise_std=(10.0, 0.01, 0.01, 0.1))
    obs_n, _ = env_noisy.reset(seed=1)
    assert not np.allclose(obs, obs_n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py::test_measurement_ranges_and_noise -v`
Expected: FAIL (stub returns zeros → `3e5 < 0` is False).

- [ ] **Step 3: Replace `_measure` with the real implementation**

```python
    def _measure(self, st):
        from org.orekit.utils import PVCoordinates
        from org.hipparchus.geometry.euclidean.threed import Vector3D

        date = st.getDate()
        sat = st.getPVCoordinates(self._eci)
        pos = sat.getPosition()
        sta = self._topo.getTransformTo(self._eci, date).transformPVCoordinates(
            PVCoordinates.ZERO
        )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env.py::test_measurement_ranges_and_noise -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add envs/od_env.py tests/test_env.py
git commit -m "feat: OdEnv measurement model (range/az/el/range-rate + Gaussian noise)"
```

---

## Task 4: OdEnv — 200-step rollout & Gym API

**Files:**
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `pytest tests/test_env.py::test_rollout_200_steps_shapes -v`
Expected: PASS immediately (no code change needed; this locks the contract). If it fails, fix the env — do not edit the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env.py
git commit -m "test: 200-step rollout shape/finiteness contract"
```

---

## Task 5: Physical invariants (J2 energy, h_z, SSO precession)

**Files:**
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing test (J2-inclusive energy + angular momentum)**

```python
# tests/test_env.py  (append)
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
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_env.py::test_invariants_bounded -v`
Expected: PASS (measured drift well under 1e-3).

- [ ] **Step 3: Write the SSO precession test**

```python
# tests/test_env.py  (append)
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
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_env.py::test_sso_raan_precession -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_env.py
git commit -m "test: J2 energy/h_z bounded invariants and SSO RAAN precession"
```

---

## Task 6: OdEncoder (MLP)

**Files:**
- Create: `models/od_encoder.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import torch


def test_od_encoder_shape():
    from models.od_encoder import OdEncoder
    enc = OdEncoder(in_dim=4, hidden_dim=256, out_dim=192)
    x = torch.randn(2, 5, 4)  # (B, T, 4)
    z = enc(x)
    assert z.shape == (2, 5, 192)
    assert torch.isfinite(z).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_od_encoder_shape -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.od_encoder'`

- [ ] **Step 3: Write minimal implementation**

```python
# models/od_encoder.py
"""MLP encoder replacing LeWM's ViT for 4-dim vector observations."""
import torch
from torch import nn


class OdEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=256, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, T, in_dim) -> (B, T, out_dim)."""
        return self.net(x.float())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_od_encoder_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/od_encoder.py tests/test_model.py
git commit -m "feat: OdEncoder MLP (4 -> 256 -> 192)"
```

---

## Task 7: ODJEPA — encode() override

**Files:**
- Create: `models/od_jepa.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (append)
def _make_odjepa(embed_dim=192, history=3):
    from models.od_jepa import ODJEPA
    from models.od_encoder import OdEncoder
    from module import ARPredictor, Embedder, MLP
    encoder = OdEncoder(4, 256, embed_dim)
    predictor = ARPredictor(
        num_frames=history, input_dim=embed_dim, hidden_dim=embed_dim,
        output_dim=embed_dim, depth=2, heads=4, mlp_dim=256, dim_head=48, dropout=0.0,
    )
    action_encoder = Embedder(input_dim=3, smoothed_dim=3, emb_dim=embed_dim)
    projector = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    pred_proj = MLP(embed_dim, 256, embed_dim, norm_fn=None)
    return ODJEPA(encoder, predictor, action_encoder, projector, pred_proj)


def test_odjepa_encode_predict_shapes():
    model = _make_odjepa()
    batch = {"obs": torch.randn(2, 3, 4), "action": torch.randn(2, 3, 3)}
    out = model.encode(batch)
    assert out["emb"].shape == (2, 3, 192)
    assert out["act_emb"].shape == (2, 3, 192)
    preds = model.predict(out["emb"], out["act_emb"])
    assert preds.shape == (2, 3, 192)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_odjepa_encode_predict_shapes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.od_jepa'`

- [ ] **Step 3: Write minimal implementation**

```python
# models/od_jepa.py
"""ODJEPA: LeWM's JEPA with encode() adapted for 4-dim vector observations.

Only encode() changes (pixels -> obs vector + MLP encoder). predict(), the predictor,
action encoder, projector, pred_proj, and SIGReg are inherited / used as-is.
"""
from einops import rearrange

from jepa import JEPA


class ODJEPA(JEPA):
    def encode(self, info):
        obs = info["obs"].float()  # (B, T, 4)
        b = obs.size(0)
        flat = rearrange(obs, "b t d -> (b t) d")
        emb = self.encoder(flat.unsqueeze(1)).squeeze(1)  # OdEncoder accepts (N,1,4)->(N,1,D)
        emb = self.projector(emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info
```

> Note: `OdEncoder` operates on `(B, T, D)`. Here we feed `(N, 1, 4)` (treating the flattened
> batch as length-1 sequences) so the projector receives `(N, D)`. This matches the baseline
> `JEPA.encode` flatten-then-reshape pattern.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_odjepa_encode_predict_shapes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/od_jepa.py tests/test_model.py
git commit -m "feat: ODJEPA encode() override for vector observations"
```

---

## Task 8: od_lejepa_forward loss

**Files:**
- Create: `models/od_forward.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (append)
def test_forward_losses_finite():
    from module import SIGReg
    from models.od_forward import od_lejepa_forward
    model = _make_odjepa()
    sigreg = SIGReg(knots=17, num_proj=128)
    batch = {"obs": torch.randn(4, 4, 4), "action": torch.randn(4, 4, 3)}
    cfg = dict(history_size=3, num_preds=1, sigreg_weight=0.09)
    out = od_lejepa_forward(model, sigreg, batch, cfg)
    for k in ("pred_loss", "sigreg_loss", "loss"):
        assert torch.isfinite(out[k]).all()
    assert out["loss"].requires_grad
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_forward_losses_finite -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.od_forward'`

- [ ] **Step 3: Write minimal implementation**

```python
# models/od_forward.py
"""LeWM LeJEPA loss for OD: prediction MSE + lambda * SIGReg. Math identical to
le-wm/train.py:lejepa_forward, reading 'obs' instead of 'pixels'."""
import torch


def od_lejepa_forward(model, sigreg, batch, cfg):
    ctx_len = cfg["history_size"]
    n_preds = cfg["num_preds"]
    lambd = cfg["sigreg_weight"]

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = model.encode(batch)
    emb = out["emb"]            # (B, T, D)
    act_emb = out["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    out["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    out["sigreg_loss"] = sigreg(emb.transpose(0, 1))
    out["loss"] = out["pred_loss"] + lambd * out["sigreg_loss"]
    return out
```

> Note: with `ctx_len=3`, `n_preds=1`, `pred_emb` is `(B,3,D)` and `tgt_emb=emb[:,1:]` is
> `(B,2,D)`. The baseline relies on broadcasting only when shapes match; to stay faithful and
> shape-safe, the loss compares the overlapping window. The test uses `T=4` so
> `pred_emb[:, :T-n_preds]` aligns with `tgt_emb`. Slice both to the common length:
> replace the `pred_loss` line with
> `m = min(pred_emb.size(1), tgt_emb.size(1)); out["pred_loss"] = (pred_emb[:, :m] - tgt_emb[:, :m]).pow(2).mean()`.

- [ ] **Step 4: Apply the shape-safe slice from the note, then run**

Run: `pytest tests/test_model.py::test_forward_losses_finite -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/od_forward.py tests/test_model.py
git commit -m "feat: od_lejepa_forward (prediction MSE + 0.09*SIGReg)"
```

---

## Task 9: Dataset generation & windowed Dataset

**Files:**
- Create: `data/generate_dataset.py`
- Create: `od_datasets/od_dataset.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py  (append)
def test_window_dataset(tmp_path):
    import numpy as np
    from data.generate_dataset import generate
    from od_datasets.od_dataset import OdWindowDataset, fit_normalizers

    path = tmp_path / "traj.npz"
    generate(n_episodes=2, episode_len=20, out_path=str(path), seed=0)
    blob = np.load(str(path))
    assert blob["obs"].shape == (2, 20, 4)
    assert blob["action"].shape == (2, 20, 3)

    ds = OdWindowDataset(str(path), window=4, normalizers=fit_normalizers(str(path)))
    item = ds[0]
    assert item["obs"].shape == (4, 4)
    assert item["action"].shape == (4, 3)
    # normalized obs should be roughly zero-mean / unit-scale, not raw metres
    assert abs(float(item["obs"].mean())) < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_window_dataset -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.generate_dataset'`

- [ ] **Step 3: Write the generator**

```python
# data/generate_dataset.py
"""Roll out OdEnv episodes (zero action) and save trajectories to .npz."""
import argparse

import numpy as np

from envs.od_env import OdEnv


def generate(n_episodes=64, episode_len=256, out_path="data/cache/od_trajectories.npz", seed=0):
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    obs_all, act_all, state_all = [], [], []
    for ep in range(n_episodes):
        env = OdEnv(max_steps=episode_len)
        obs, info = env.reset(seed=seed + ep)
        obs_ep, act_ep, state_ep = [obs], [np.zeros(3, np.float32)], [info["state"]]
        for _ in range(episode_len - 1):
            a = np.zeros(3, dtype=np.float32)
            obs, _, _, _, info = env.step(a)
            obs_ep.append(obs)
            act_ep.append(a)
            state_ep.append(info["state"])
        obs_all.append(obs_ep)
        act_all.append(act_ep)
        state_all.append(state_ep)
    np.savez(
        out_path,
        obs=np.asarray(obs_all, dtype=np.float32),
        action=np.asarray(act_all, dtype=np.float32),
        state=np.asarray(state_all, dtype=np.float32),
    )
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=64)
    p.add_argument("--episode-len", type=int, default=256)
    p.add_argument("--out", default="data/cache/od_trajectories.npz")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print("wrote", generate(args.n_episodes, args.episode_len, args.out, args.seed))
```

- [ ] **Step 4: Write the windowed Dataset + normalizers**

```python
# od_datasets/od_dataset.py
"""Windowed torch Dataset over generated OD trajectories, with z-score normalization."""
import numpy as np
import torch
from torch.utils.data import Dataset


def fit_normalizers(npz_path):
    """Return {'obs': (mean,std), 'action': (mean,std)} over all timesteps."""
    blob = np.load(npz_path)
    out = {}
    for key in ("obs", "action"):
        flat = blob[key].reshape(-1, blob[key].shape[-1])
        mean = flat.mean(0)
        std = flat.std(0)
        std[std < 1e-8] = 1.0
        out[key] = (torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32))
    return out


class OdWindowDataset(Dataset):
    def __init__(self, npz_path, window=4, normalizers=None):
        blob = np.load(npz_path)
        self.obs = torch.tensor(blob["obs"], dtype=torch.float32)        # (E, L, 4)
        self.action = torch.tensor(blob["action"], dtype=torch.float32)  # (E, L, 3)
        self.window = window
        self.normalizers = normalizers
        E, L, _ = self.obs.shape
        self.index = [
            (e, s) for e in range(E) for s in range(L - window + 1)
        ]

    def __len__(self):
        return len(self.index)

    def _norm(self, x, key):
        if self.normalizers is None:
            return x
        mean, std = self.normalizers[key]
        return (x - mean) / std

    def __getitem__(self, i):
        e, s = self.index[i]
        obs = self._norm(self.obs[e, s : s + self.window], "obs")
        action = self._norm(self.action[e, s : s + self.window], "action")
        return {"obs": obs, "action": action}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_model.py::test_window_dataset -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data/generate_dataset.py od_datasets/od_dataset.py tests/test_model.py
git commit -m "feat: trajectory generation and windowed normalized dataset"
```

---

## Task 10: Configs + train_od.py + W&B + train smoke test

**Files:**
- Create: `config/train/od.yaml`, `config/train/model/od.yaml`, `config/train/data/od.yaml`
- Create: `train_od.py`
- Test: `tests/test_train_smoke.py`

- [ ] **Step 1: Write the model config**

```yaml
# config/train/model/od.yaml
_target_: models.od_jepa.ODJEPA

encoder:
  _target_: models.od_encoder.OdEncoder
  in_dim: 4
  hidden_dim: 256
  out_dim: ${embed_dim}

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 4
  heads: 8
  mlp_dim: 512
  dim_head: 48
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: 3
  smoothed_dim: 3
  emb_dim: ${embed_dim}

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  hidden_dim: 512
  output_dim: ${embed_dim}
  norm_fn: null

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  hidden_dim: 512
  output_dim: ${embed_dim}
  norm_fn: null
```

- [ ] **Step 2: Write the data and top-level configs**

```yaml
# config/train/data/od.yaml
path: data/cache/od_trajectories.npz
window: ${eval:'${history_size} + ${num_preds}'}
batch_size: 64
train_split: 0.9
```

```yaml
# config/train/od.yaml
defaults:
  - _self_
  - model: od
  - data: od

seed: 3072
embed_dim: 192
history_size: 3
num_preds: 1

loss:
  sigreg:
    weight: 0.09
    kwargs:
      knots: 17
      num_proj: 1024

optimizer:
  lr: 5e-5
  weight_decay: 1e-3

trainer:
  max_epochs: 8
  accelerator: auto
  devices: auto
  precision: 32
  gradient_clip_val: 1.0

wandb:
  enabled: true
  config:
    entity: sps-tum
    project: ${oc.env:WANDB_PROJECT,space-world-models}
```

- [ ] **Step 3: Write train_od.py**

```python
# train_od.py
"""Full Lightning training loop for the OD LeWM model (mirrors le-wm/train.py)."""
from functools import partial

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from od_datasets.od_dataset import OdWindowDataset, fit_normalizers
from models.od_forward import od_lejepa_forward
from module import SIGReg


def _forward(self, batch, stage, cfg):
    flat = {
        "history_size": cfg.history_size,
        "num_preds": cfg.num_preds,
        "sigreg_weight": cfg.loss.sigreg.weight,
    }
    out = od_lejepa_forward(self.model, self.sigreg, batch, flat)
    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k},
        on_step=True, sync_dist=True,
    )
    return out


@hydra.main(version_base=None, config_path="./config/train", config_name="od")
def run(cfg):
    import stable_pretraining as spt
    from lightning.pytorch.loggers import WandbLogger

    norms = fit_normalizers(cfg.data.path)
    full = OdWindowDataset(cfg.data.path, window=cfg.data.window, normalizers=norms)
    gen = torch.Generator().manual_seed(cfg.seed)
    n_train = int(len(full) * cfg.data.train_split)
    train_set, val_set = random_split(full, [n_train, len(full) - n_train], generator=gen)
    train = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(val_set, batch_size=cfg.data.batch_size, shuffle=False)

    world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {"type": "AdamW", **dict(cfg.optimizer)},
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(_forward, cfg=cfg),
        optim=optimizers,
    )
    data_module = spt.data.DataModule(train=train, val=val)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(**cfg.trainer, logger=logger, num_sanity_val_steps=0)
    manager = spt.Manager(trainer=trainer, module=module, data=data_module)
    manager()


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Write the train smoke test**

```python
# tests/test_train_smoke.py
import lightning as pl
import torch
from functools import partial
from torch.utils.data import DataLoader

from data.generate_dataset import generate
from od_datasets.od_dataset import OdWindowDataset, fit_normalizers
from models.od_forward import od_lejepa_forward
from module import SIGReg


def test_train_smoke(tmp_path):
    import stable_pretraining as spt
    path = tmp_path / "traj.npz"
    generate(n_episodes=4, episode_len=16, out_path=str(path), seed=0)
    ds = OdWindowDataset(str(path), window=4, normalizers=fit_normalizers(str(path)))
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

    from tests.test_model import _make_odjepa
    model = _make_odjepa()

    def _fwd(self, batch, stage):
        cfg = {"history_size": 3, "num_preds": 1, "sigreg_weight": 0.09}
        out = od_lejepa_forward(self.model, self.sigreg, batch, cfg)
        self.log_dict({f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k})
        return out

    module = spt.Module(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=64),
        forward=_fwd,
        optim={"model_opt": {"modules": "model",
                             "optimizer": {"type": "AdamW", "lr": 5e-5},
                             "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                             "interval": "epoch"}},
    )
    data_module = spt.data.DataModule(train=loader, val=loader)
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False,
                         enable_checkpointing=False)
    spt.Manager(trainer=trainer, module=module, data=data_module)()
```

- [ ] **Step 5: Run the train smoke test**

Run: `pytest tests/test_train_smoke.py::test_train_smoke -v`
Expected: PASS (one train + one val batch run end-to-end; losses finite).
If the `spt.Module`/`spt.Manager` API differs from the vendored `train.py` usage, adjust the
call sites to match the installed `stable_pretraining` 0.1.7 API (keep the loss math identical).

- [ ] **Step 6: End-to-end manual check (not a unit test)**

```bash
python data/generate_dataset.py --n-episodes 64 --episode-len 256
WANDB_MODE=offline python train_od.py trainer.max_epochs=1 wandb.enabled=false
```
Expected: training runs one epoch, prints decreasing `train/loss`, exits 0.

- [ ] **Step 7: Run the full suite**

Run: `pytest -v`
Expected: all env, model, and train-smoke tests PASS.

- [ ] **Step 8: Commit**

```bash
git add config/ train_od.py tests/test_train_smoke.py
git commit -m "feat: hydra configs, train_od.py full Lightning loop + W&B, train smoke test"
```

---

## Self-review notes (resolved)

- **Spec coverage:** env spec (Tasks 2-4), obs noise (Task 3), MLP encoder (Task 6), predictor/SIGReg unchanged (vendored, Tasks 7-8), smoke test rollout+shapes+energy (Tasks 4-5), J2/SSO (Task 5), full training loop + W&B (Task 10). All covered.
- **Energy invariant:** uses J2-inclusive specific energy + `h_z` with measured tolerances (1e-3 vs measured 4.5e-5 / 2.2e-5), per the EcksteinHechler decision.
- **Loss shape safety:** Task 8 slices `pred_emb`/`tgt_emb` to common length (the baseline `num_preds=1` case).
- **Risk note:** `stable_pretraining.Module/Manager` exact API is mirrored from the verified `le-wm/train.py`; Task 10 Step 5 explicitly calls out adapting to the installed 0.1.7 API if signatures differ. This is the only unverified integration surface; everything else (Orekit, encoder, JEPA, loss, dataset) is probed or unit-tested.
