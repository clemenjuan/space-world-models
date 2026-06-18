# OD Constellation Batch-Inference Experiment Draft

## Motivation

The current OD benchmark shows a surprising throughput signal: batched LeWM
inference is faster than the current analytical Orekit replay in the local board
snapshot. That result is promising, but the existing comparison is not yet a fair
orbit-determination experiment. Orekit is replaying a known seed/state, while the
LeWM rows are measuring latent prediction or a weak frozen decoder.

This experiment tests a sharper hypothesis:

```text
Given the same noisy tracking measurements, a geometry-aware learned OD decoder
can amortize inference across many satellites and deliver useful state estimates
faster than per-object analytical OD solves, while staying inside acceptable
accuracy and calibration bounds.
```

The important word is amortize. The learned model may not beat a single classical
solve on accuracy, but it could become valuable when a constellation generates
many independent tracking windows that can be processed as one batch.



## Repository Strategy

Keep this work in the current repository for now. The OD, FDIR, data generation,
checkpoint discovery, and local result boards still share enough infrastructure
that splitting repos would slow iteration. Use namespaced scripts, artifacts, and
docs inside this repo:

```text
data/cache/od_*.npz
data/figures/od_*.json|pt
scripts/train_od_*.py
scripts/benchmark_od_*.py
docs/od_*.md
```

Split a dedicated OD repository only after the experiment has a stable external
interface: reproducible datasets, a fixed benchmark payload schema, and at least
one reusable analytical baseline or decoder package. At that point, the split can
be a clean extraction rather than a premature fork.

## Claims To Separate

1. Representation claim: the LeWM latent history contains enough information to
   estimate orbit state when timestamp and station geometry are supplied.
2. Estimation claim: the decoded state is competitive with measurement-only
   analytical OD baselines on held-out orbits, noise levels, and tracking gaps.
3. Throughput claim: learned batch inference has better scaling with object count
   than running the analytical estimator independently for each object.
4. Hybrid claim: LeWM may be most useful as a fast initializer or residual
   corrector for a classical estimator, not as a full replacement.

The experiment should report these as separate rows rather than folding them into
one headline number.

## Why The Previous Decode Was Not Enough

The failed decoder trained:

```text
latent sequence -> absolute ECI state
```

without explicit timestamp or station-frame features. That is under-specified for
absolute ECI recovery. Range, azimuth, elevation, and range-rate are topocentric
measurements; to decode ECI state we need the station pose and velocity at the
target time. The next decoder should therefore receive either:

- raw observation history plus time and station pose;
- LeWM latent history plus time and station pose;
- both raw observations and LeWM latents, for an ablation.

The first milestone is not beating analytical OD. The first milestone is proving
that the geometry-aware decoder can beat the old no-geometry decoder by a large
margin on held-out episodes.

## Dataset Design

Start with the existing single-satellite OD generator, then generalize the saved
sample shape from:

```text
episode, time, feature
```

to:

```text
episode, object, time, feature
```

Recommended first grid:

- objects per episode: 1, 8, 32, 128;
- episode length: 256 steps at 30 s cadence;
- ground stations: keep the current Munich-like station for phase 1, then add 3
  and 8 station networks;
- measurements: range, azimuth, elevation, range-rate;
- noise levels: current noise, 3x noise, 10x noise;
- tracking gaps: none, random gaps, station visibility masks;
- held-out split: hold out entire orbital-element seeds, not just windows.

For each target time, save geometry features alongside observations:

```text
time_since_epoch_s
station_eci_position_m[3]
station_eci_velocity_m_s[3]
line_of_sight_unit_eci[3]
optional station_id embedding/index
```

This lets a decoder learn state estimation rather than silently learning frame
conversion.

## Methods

Use five method families:

1. Known-orbit propagation reference. This is the existing Orekit replay and
   remains an upper bound, not a measurement-only OD method.
2. Batch least-squares OD. Fit initial state from a measurement arc, then
   propagate to target times with the same dynamics model.
3. EKF or UKF OD. Sequential filtering with the same measurement model and
   process assumptions.
4. Geometry-aware LeWM decoder. Encode/predict measurement windows, concatenate
   geometry/time features, decode state mean and optionally covariance.
5. Hybrid LeWM + analytical OD. Use LeWM state as an initializer for least
   squares, or use LeWM residuals to warm-start/shorten analytical convergence.

The fair comparison rows are methods 2, 3, 4, and 5. Method 1 is a sanity ceiling.

## Decoder Variants

Run the decoder ladder in this order:

1. Raw geometry baseline:

```text
raw observation history + geometry -> ECI state
```

2. Frozen latent probe:

```text
LeWM predicted/target latent history + geometry -> ECI state
```

3. Latent plus raw observation ablation:

```text
latent history + raw observation history + geometry -> ECI state
```

4. Joint state-supervised LeWM:

```text
JEPA latent prediction loss + auxiliary state/covariance loss
```

The old no-geometry decoder should remain as a negative control.

## Metrics

Accuracy:

- position RMSE, median, P95, max in meters;
- velocity RMSE, median, P95, max in m/s;
- observation residual RMSE after projecting state back to measurements;
- along-track, cross-track, radial error decomposition.

Uncertainty and calibration:

- covariance trace and determinant;
- normalized innovation squared for filter rows;
- empirical coverage for 1-sigma, 2-sigma, and 3-sigma intervals;
- failure rate when covariance is overconfident.

Runtime and scaling:

- cold load time;
- steady-state wall time per object and per timestep;
- throughput objects/s for object counts 1, 8, 32, 128;
- CPU time, GPU time when applicable, peak RSS/VRAM;
- analytical iteration count for least-squares rows.

Robustness:

- degradation under noise/gaps;
- performance on held-out orbital regimes;
- stability over rollout horizon.

## Visualizations

The board should make the difference visible, not just tabular:

- 3D truth orbit vs decoded orbit vs analytical estimate for selected objects;
- position-error time series with log-scale y-axis;
- radial/along-track/cross-track error bands;
- observation residual plots for range, azimuth, elevation, and range-rate;
- throughput scaling chart: object count on x-axis, objects/s and ms/object on
  y-axes;
- accuracy/latency Pareto plot for all methods;
- calibration plot for predicted covariance coverage.

For constellation runs, add a small-multiple view that samples several objects
from the same batch so we can see whether speed comes with uneven failures.

## Acceptance Criteria

Phase 1 succeeds when the geometry-aware raw decoder beats the old no-geometry
latent decoder by at least one order of magnitude in position RMSE on the current
single-satellite dataset. This validates the target construction.

Phase 2 succeeds when the frozen LeWM geometry decoder reports state-space errors
on the same plots as batch least squares/EKF without changing the measurement
inputs. It does not need to win yet; it needs to be comparable enough to study.

Phase 3 succeeds when the constellation benchmark shows a clear throughput curve:
analytical OD scales roughly per object, while LeWM batch inference improves
objects/s as object count increases. Accuracy must remain within a pre-declared
mission tolerance band.

Phase 4 succeeds if the hybrid row reduces analytical solve time or iteration
count while preserving analytical accuracy and calibration.

## Implementation Plan

1. Extend OD generation to optionally save station/time geometry features.
2. Add a geometry-aware decoder script with raw-observation, latent-only, and
   latent-plus-raw modes.
3. Keep the old decoder artifact as a negative-control row on the board.
4. Add a measurement-only analytical OD baseline: batch least squares first, EKF
   second.
5. Generalize benchmark payloads from one episode/object to object batches.
6. Add board plots for state error, residuals, covariance calibration, and
   throughput scaling.
7. Run the scaling sweep on 1, 8, 32, and 128 objects with fixed random seeds.

## First Concrete Command Targets

The draft code path can look like this:

```bash
uv run python data/generate_dataset.py --n-episodes 64 --episode-len 256 --out data/cache/od_trajectories_geometry.npz --seed 0 --save-geometry
uv run python scripts/train_od_geometry_decoder.py --dataset data/cache/od_trajectories_geometry.npz --mode raw --wandb --wandb-name od-geometry-raw-w8-192ep
uv run python scripts/train_od_geometry_decoder.py --dataset data/cache/od_trajectories_geometry.npz --mode latent
uv run python scripts/benchmark_od_constellation.py --objects 1 8 32 128 --methods bls ekf lewm hybrid
uv run python scripts/build_results_board.py
```

Current implementation status:

- `data/generate_dataset.py --save-geometry` exists and writes `time_s`,
  `station_state_eci`, and `topocentric_basis_eci` alongside the existing `obs`,
  `action`, and `state` keys.
- `data/generate_dataset.py --source ... --out ...` exists for quickly
  augmenting an existing OD cache with station geometry.
- `scripts/train_od_geometry_decoder.py --mode raw` exists for the first
  raw-observation geometry sanity check.
- `scripts/train_od_geometry_decoder.py --mode raw_eci` exists for explicit
  line-of-sight ECI and measured-position ECI features.
- The OD results board now plots the tuned geometry-decoder orbit, sampled
  residual vectors, and empirical position/velocity uncertainty bands.
- `--mode latent` and `scripts/benchmark_od_constellation.py` are still intended
  interfaces for the next implementation slices.



## First Raw-Geometry Run

Completed on 2026-06-18 with WandB run
`od-geometry-raw-w8-192ep`:

```text
https://wandb.ai/sps-tum/space-world-models/runs/pu6sdr7j
```

Configuration:

- dataset: `data/cache/od_trajectories_geometry.npz` augmented from the existing
  192 episode OD cache;
- held-out eval episode: 0;
- window: 8 steps;
- decoder: raw observation geometry MLP, hidden dim 256, depth 2;
- epochs: 50;
- train samples: 47,559;
- eval samples: 249.

Results:

- best validation loss: 0.0010528;
- position RMSE: 87,759.9 m;
- position median: 76,109.7 m;
- position P95: 149,675.3 m;
- velocity RMSE: 196.24 m/s;
- prediction latency: 0.194 ms/sample.

This beats the old no-geometry latent decoder snapshot by roughly an order of
magnitude in position RMSE, but it is still far from a credible OD estimator. The
next useful slice is a stronger geometry-aware target: explicit line-of-sight ECI
features and/or a decoder that predicts topocentric relative state before the ECI
conversion.



## Tuned Raw-ECI Geometry Run

Completed on 2026-06-18 with WandB run
`od-geometry-raw-eci-w16-h512-d3-192ep`:

```text
https://wandb.ai/sps-tum/space-world-models/runs/b6a7wfee
```

Configuration:

- dataset: `data/cache/od_trajectories_geometry.npz` with exact
  `topocentric_basis_eci`;
- feature mode: `raw_eci`, adding line-of-sight ECI, measured-position ECI, and
  radial-velocity ECI hints;
- held-out eval episode: 0;
- window: 16 steps;
- decoder: hidden dim 512, depth 3;
- epochs: 80;
- train samples: 46,031;
- eval samples: 241.

Results:

- best validation loss: 0.0001103;
- position RMSE: 40,596.7 m;
- position median: 36,594.8 m;
- position P95: 64,312.8 m;
- velocity RMSE: 73.83 m/s;
- prediction latency: 0.012 ms/sample.

Relative to the first raw-geometry run, this reduced held-out position RMSE from
87.8 km to 40.6 km and velocity RMSE from 196.2 m/s to 73.8 m/s. The remaining
error is now plausibly dominated by single-station observability and the current
angle noise (`0.01 rad`) rather than just missing frame geometry.

## Main Risks

- Single-station arcs may be weakly observable for velocity over short windows.
  Mitigation: report observability failures, increase history length, and add
  multi-station variants.
- A learned decoder can look fast by ignoring hard cases. Mitigation: include
  held-out orbit families, gap/noise sweeps, and calibration checks.
- Batch inference latency can be dominated by model load or Python overhead.
  Mitigation: separate cold-start and warm steady-state timing.
- A pure learned estimate may be less defensible than a filter. Mitigation: keep
  the hybrid initializer row as a first-class candidate.

## Decision Gate

If the geometry-aware decoder still has kilometer-scale errors on held-out
single-satellite data, pause constellation scaling and fix the representation or
observability setup. If it reaches useful state accuracy, proceed to the
constellation throughput sweep where the batch-inference hypothesis can be tested
properly.
