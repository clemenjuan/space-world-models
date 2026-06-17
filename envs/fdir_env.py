"""Gymnasium FDIR (fault detection) environment for 8-dim spacecraft telemetry.

Hidden state (8-dim, SI-ish units around a fixed operating point ``x*``):
``[solar_array_voltage, battery_soc, panel_temp, obc_temp,
   rw_speed_x, rw_speed_y, rw_speed_z, bus_current]``.

Nominal dynamics are pure NumPy linear mean-reversion on the *deviation*
``d = x - x*``:  ``d_{t+1} = A d_t + w``, ``w ~ N(0, diag(Q^2))``, ``x = x* + d``.
``A`` is near-diagonal (mean-reverting, spectral radius < 1) with deliberate
off-diagonal cross-couplings so a state-level fault propagates to correlated
channels. Faults are injected at the **state (dynamics) level**, not just on the
emitted observation, so coupled channels drift away from what a nominal-trained
model expects.

Observation: ``o_t = x_t + sensor_noise`` (np.float32). Action is a Discrete(4)
recovery command that does NOT affect dynamics in this step (plumbed, ignored).
No Orekit / Java / torch — pure NumPy.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Channel order / names for the 8-dim hidden state.
CHANNEL_NAMES = (
    "solar_array_voltage",
    "battery_soc",
    "panel_temp",
    "obc_temp",
    "rw_speed_x",
    "rw_speed_y",
    "rw_speed_z",
    "bus_current",
)
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# Fixed realistic operating point x* (one value per channel, same order).
#   solar_array_voltage [V], battery_soc [fraction], panel/obc temp [C],
#   rw_speed_{x,y,z} [rpm], bus_current [A].
DEFAULT_X_STAR = np.array(
    [28.0, 0.80, 40.0, 25.0, 1000.0, -500.0, 800.0, 5.0], dtype=np.float64
)

# Per-channel process-noise std (the std of w, units-appropriate). Small so the
# nominal deviation hovers tightly around x*.
DEFAULT_Q = np.array(
    [0.05, 0.002, 0.2, 0.2, 5.0, 5.0, 5.0, 0.05], dtype=np.float64
)

# Per-channel sensor-noise std (added to the true state to form the observation).
# Kept small relative to the nominal deviation std of each channel.
DEFAULT_SENSOR_SIGMA = np.array(
    [0.02, 0.001, 0.1, 0.1, 2.0, 2.0, 2.0, 0.02], dtype=np.float64
)


def build_A():
    """Build the 8x8 near-diagonal dynamics matrix ``A`` acting on the deviation.

    Diagonal entries lie in ~[0.80, 0.95] (mean-reverting toward x*). A few
    documented off-diagonal cross-couplings (~0.05-0.10) let a state-level fault
    on one channel propagate to physically correlated channels:

      * solar_array_voltage -> battery_soc   (array feeds the battery charge)
      * solar_array_voltage -> panel_temp    (illuminated array heats the panel)
      * rw_speed_{x,y,z}    -> bus_current   (reaction wheels draw bus current)

    ``A`` is kept diagonally dominant so its spectral radius stays < 1 (stable).
    Off-diagonals are small relative to the mean-reverting diagonal terms, which
    guarantees diagonal dominance (and hence eigenvalues inside the unit disk).
    """
    A = np.zeros((8, 8), dtype=np.float64)
    # Diagonal (mean reversion) — all within [0.80, 0.95].
    diag = np.array([0.90, 0.92, 0.85, 0.88, 0.95, 0.95, 0.95, 0.80])
    np.fill_diagonal(A, diag)
    i = CHANNEL_INDEX
    # Documented cross-couplings (effect_channel <- source_channel).
    A[i["battery_soc"], i["solar_array_voltage"]] = 0.08
    A[i["panel_temp"], i["solar_array_voltage"]] = 0.10
    A[i["bus_current"], i["rw_speed_x"]] = 0.05
    A[i["bus_current"], i["rw_speed_y"]] = 0.05
    A[i["bus_current"], i["rw_speed_z"]] = 0.05
    return A


# Module-level default A (the env copies this; tests can read FdirEnv.A / DEFAULT_A).
DEFAULT_A = build_A()


def _resolve_channel(channel):
    """Accept a channel name (str) or an integer index; return the int index."""
    if channel is None:
        return None
    if isinstance(channel, str):
        return CHANNEL_INDEX[channel]
    return int(channel)


class FdirEnv(gym.Env):
    """8-dim spacecraft telemetry env with linear dynamics + state-level faults.

    Faults are configured at construction and become active once
    ``self._step >= fault_step``. They modify the state evolution itself.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        x_star=DEFAULT_X_STAR,
        A=DEFAULT_A,
        Q=DEFAULT_Q,
        sensor_sigma=DEFAULT_SENSOR_SIGMA,
        init_dev_std=0.1,
        max_steps=1000,
        fault_mode=None,
        fault_channel=None,
        fault_step=100,
        drift_rate=0.5,
        spike_magnitude=5.0,
        spike_duration=1,
    ):
        super().__init__()
        self.x_star = np.asarray(x_star, dtype=np.float64).copy()
        # Copy A so the module-level default is never mutated; expose as self.A.
        self.A = np.asarray(A, dtype=np.float64).copy()
        self.Q = np.asarray(Q, dtype=np.float64).copy()
        self.sensor_sigma = np.asarray(sensor_sigma, dtype=np.float64).copy()
        # init_dev_std scales the per-channel process-noise std for the small
        # seeded deviation applied at reset (keeps the start near x*).
        self.init_dev_std = float(init_dev_std)
        self.max_steps = int(max_steps)

        # Fault configuration.
        if fault_mode not in (None, "stuck_at", "drift", "spike"):
            raise ValueError(f"unknown fault_mode: {fault_mode!r}")
        self.fault_mode = fault_mode
        self.fault_channel = _resolve_channel(fault_channel)
        self.fault_step = int(fault_step)
        self.drift_rate = float(drift_rate)
        self.spike_magnitude = float(spike_magnitude)
        self.spike_duration = int(spike_duration)
        if self.fault_mode is not None and self.fault_channel is None:
            raise ValueError("fault_channel is required when fault_mode is set")

        # Observation: raw telemetry units; bounds generous around x*.
        high = np.array(
            [60.0, 1.2, 150.0, 120.0, 6000.0, 6000.0, 6000.0, 50.0], dtype=np.float32
        )
        low = np.array(
            [-10.0, -0.2, -50.0, -50.0, -6000.0, -6000.0, -6000.0, -50.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Discrete recovery commands {0 nominal, 1 isolate_power, 2 safe_mode,
        # 3 reset_obc}. Action does NOT affect dynamics in this step.
        self.action_space = spaces.Discrete(4)

        self._d = None        # current deviation d = x - x*
        self._step = 0
        # Value the faulted channel's *state* is held at for stuck_at (set at onset).
        self._stuck_value = None

    def _fault_active(self):
        """Whether a fault is currently being injected at the present step."""
        return self.fault_mode is not None and self._step >= self.fault_step

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Start at x* plus a small seeded random deviation.
        self._d = self.np_random.normal(0.0, self.init_dev_std * self.Q)
        self._step = 0
        self._stuck_value = None
        x = self.x_star + self._d
        return self._measure(x), self._info(x)

    def step(self, action):
        self._step += 1
        # 1) Nominal linear evolution of the deviation: d_{t+1} = A d_t + w.
        w = self.np_random.normal(0.0, self.Q)
        self._d = self.A @ self._d + w
        x = self.x_star + self._d

        # 2) State-level fault injection (modifies x, then sync back into d so the
        #    fault persists/propagates through A on subsequent steps).
        if self._fault_active():
            c = self.fault_channel
            if self.fault_mode == "stuck_at":
                # On the first active step, latch the channel's current value; then
                # hold it. The rest of the state still evolves through A, so coupled
                # channels drift away from nominal.
                if self._stuck_value is None:
                    self._stuck_value = float(x[c])
                x[c] = self._stuck_value
            elif self.fault_mode == "drift":
                # Cumulative ramp: a constant increment added every active step.
                steps_active = self._step - self.fault_step + 1
                x[c] = x[c] + self.drift_rate * steps_active
            elif self.fault_mode == "spike":
                # Transient impulse for spike_duration steps starting at fault_step.
                if self._step < self.fault_step + self.spike_duration:
                    x[c] = x[c] + self.spike_magnitude
            # Re-sync deviation so the (possibly overridden) state feeds A next step.
            self._d = x - self.x_star

        terminated = False
        truncated = self._step >= self.max_steps
        return self._measure(x), 0.0, terminated, truncated, self._info(x)

    def _measure(self, x):
        """Emit a noisy observation o = x + sensor_noise (np.float32)."""
        noisy = x + self.np_random.normal(0.0, self.sensor_sigma)
        return noisy.astype(np.float32)

    def _info(self, x):
        """info dict carrying the true 8-dim state and the fault-active flag."""
        return {
            "state": x.astype(np.float32),
            "fault_active": bool(self._fault_active()),
        }
