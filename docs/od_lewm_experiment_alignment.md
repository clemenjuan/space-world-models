# OD LeWM Experiment Alignment

This note is a shared map of what the current orbit-determination LeWM experiment
is doing, what it is not doing yet, and what we should benchmark next.

## Problem

We simulate a satellite orbit and train a learned world model to predict the
future of the measurement stream. The current environment is orbit
determination-flavored, but the current model is not yet a full orbit
determination filter.

At each time step the simulator has a hidden truth state:

```text
ECI state = [x, y, z, vx, vy, vz]
```

That truth state is propagated by Orekit's Eckstein-Hechler analytical
propagator. The generated dataset saves this truth state so we can inspect it
and later use it for supervised probes or decoders.

The learner does not receive the full truth state. It receives a synthetic
ground-station measurement:

```text
observation = [range, azimuth, elevation, range_rate]
```

This is closer to a tracking sensor: a station sees line-of-sight geometry and
range/range-rate, not the satellite's full inertial state directly.

## Why Not Orbital Elements?

Orbital elements are a useful compact description of an orbit, but they are not
what a ground station measures directly. A station measures geometry between the
station and spacecraft. Orbital elements are inferred from multiple observations
over time.

Using `[range, azimuth, elevation, range_rate]` makes the problem more like real
orbit determination:

- the true 6D state exists in the simulator;
- the model sees partial, noisy measurements;
- state information must be inferred from temporal context.

We can still decode or report orbital elements later by converting a recovered
ECI state to elements.

## Why One Measurement Is Not Enough

A single measurement at one instant does not uniquely determine the full
6D ECI state. Range and angles constrain the line-of-sight position relative to
the station, and range-rate constrains only the radial component of relative
velocity. Tangential velocity and other geometry remain ambiguous.

That is why a better decoder should use history/context, not just one latent
token:

```text
observation history -> latent history/context -> estimated [x,y,z,vx,vy,vz]
```

The movement over several measurements contains the missing information.

## What The Current LeWM Learns

The current OD-JEPA path is:

```text
4D observation -> MLP encoder -> 192D latent
action -> action encoder -> 192D action latent
latent history + action latent -> autoregressive predictor -> next 192D latent
```

The loss compares predicted future latent vectors against target future latent
vectors produced by the encoder. SIGReg regularizes the latent distribution.

The trained OD-JEPA model itself has no built-in decoder. We now have a separate frozen supervised probe decoder for diagnostics. Without that probe, the board can show:

- true orbit propagation from the simulator;
- observations seen by the model;
- predicted vs target latent trajectories.

The probe can now show decoded ECI state error, but it is not yet accurate enough to trust as a learned predicted orbit. A usable learned ECI orbit requires a stronger decoder, explicit geometry/time context, or joint state supervision.

## Decode-Back Options

There are three increasingly useful decode targets:

1. Decode latent to observation:

```text
192D latent -> [range, azimuth, elevation, range_rate]
```

This checks whether the latent preserves the measurement stream.

2. Decode latent/history to ECI state:

```text
latent context -> [x, y, z, vx, vy, vz]
```

This checks whether the latent contains enough information for orbit
determination.

3. Decode latent/history to covariance-aware state estimate:

```text
latent context -> state mean + covariance
```

This would let the learned model report uncertainty like a classical estimator.

The easiest first experiment is a frozen probe decoder: train a small decoder on
top of a trained encoder without changing the world model. If that works, the
latent already contains orbit information. If not, train with an auxiliary state
loss.

## Algorithmic Baseline Comparison

For a fair comparison we need to define the classical baseline. Candidate
baselines:

- Orekit propagation only, starting from known truth state;
- least-squares orbit determination from range/angle/range-rate observations;
- Kalman or extended Kalman filtering with a dynamics model and measurement
  model.

The useful comparison table should include:

- wall-clock time per trajectory and per prediction step;
- CPU percent and peak RSS memory;
- state error: position RMSE, velocity RMSE;
- observation error: range/angle/range-rate RMSE;
- uncertainty quality: covariance trace/determinant, normalized innovation
  squared, and calibration coverage;
- robustness to measurement noise and gaps;
- rollout stability over horizon;
- model artifact size and load time.

The learned model can be faster at inference once trained, but it must earn that
against accuracy, uncertainty calibration, and generalization.

## Current Benchmark Artifact

The decoder probe is trained locally with:

```bash
uv run python scripts/train_od_latent_decoder.py
```

By default this probe does not log to WandB. It writes
`data/figures/od_latent_decoder.pt` and
`data/figures/od_latent_decoder_metrics.json` so the local board can consume it.

The comparison is generated with:

```bash
uv run python scripts/benchmark_od_methods.py
```

The board reads the resulting artifact at
`data/figures/od_method_benchmark.json` and renders latency, CPU, RSS, native
error traces, covariance trace, and a metrics table.

Current episode 0 snapshot:

- Orekit known-orbit propagation: 256 samples, 1.15 ms/sample, 31.4 MB RSS delta,
  383.3 MB process max RSS, 0.180 m position RMSE, 0.000186 m/s velocity RMSE,
  and residual covariance trace 0.0 in observation space.
- LeWM latent predictor: 253 windows, 1.58 ms/window batched, 17.7 ms/window in
  an online-ish eager loop, 22.2 MB RSS delta, 584.1 MB process max RSS,
  0.00323 mean latent MSE, and residual covariance trace 0.616 in 192D latent
  space.
- LeWM decoded ECI state: 253 windows, 0.400 ms/window for batched
  encode/predict/decode, 22.2 MB RSS delta, 584.4 MB process max RSS,
  905,906 m position RMSE, 982.6 m/s velocity RMSE, and residual covariance trace
  8.08e11 in ECI state residual space.

This is useful but not a final apples-to-apples OD contest. The Orekit row is a
known-orbit physics replay that starts from the same seed as the dataset. The
LeWM latent row is still latent prediction. The decoded-state row is a frozen
supervised probe, not a jointly trained OD estimator. Its large ECI error says
the current representation/decoder setup does not yet recover absolute inertial
state well from held-out episode measurements. A fair next baseline is batch
least-squares or EKF using the same observations, plus a decoder that explicitly
uses time/station geometry or is trained jointly with state supervision.

## Why The First Decode Failed

The current frozen decoder asks:

```text
LeWM latent sequence -> absolute ECI state [x, y, z, vx, vy, vz]
```

but its input is only the latent sequence derived from normalized
`[range, azimuth, elevation, range_rate]` windows. It does not receive the
timestamp, station ECI position/velocity, or the topocentric-to-ECI transform.
That missing geometry is not a small detail. Range and angles identify the
satellite position relative to the ground station at a specific time; converting
that relative line of sight into absolute ECI coordinates requires the station
frame at that time. Range-rate constrains radial relative velocity, while the
remaining velocity components have to be inferred from motion over a history
window.

So the current decoder is solving a harder and partly under-specified problem:
recover absolute inertial state from a learned measurement embedding without the
reference-frame information needed to express the answer. The result is a useful
negative diagnostic, not evidence that the latent dynamics are useless.

The next decode should make the frame conversion explicit:

```text
measurement/latent history + time + station pose -> topocentric relative state
topocentric relative state + station pose -> ECI state
```

or train a joint state head that receives those geometry features directly. The
first sanity check should be a non-LeWM measurement-geometry decoder:

```text
raw observation history + time + station pose -> ECI state
```

If that cannot reach reasonable held-out error, the learned latent decoder has no
fair target yet.

## Current Board Episode Convention

The board uses dataset episode 0 for:

- the observation plot;
- the LeWM latent prediction probe.

It also plots dataset episode 1 in the orbit view only as a comparison trajectory
from the generated dataset. Episode 1 is not used in the current LeWM prediction
plot.

## Next Experiment Draft

The next experiment is drafted in
`docs/od_constellation_batch_inference_experiment.md`. It separates two claims:

- representation claim: a geometry-aware LeWM decoder can recover state from the
  same measurements used by a classical estimator;
- throughput claim: once trained, batched learned inference scales better across
  many simultaneous objects than per-object analytical OD solves.
