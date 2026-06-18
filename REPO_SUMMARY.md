# Space World Models Repo Summary

This repo is a compact telemetry adaptation of LeWorldModel / LeJEPA. The core
idea is the same as the LeWM theory: learn a latent world model from offline
trajectories by encoding observations into embeddings, predicting future
embeddings from history and actions, and preventing representation collapse with
SIGReg.

The important adaptation is that this repo does not train from pixels. It uses
vector observations:

- OD: 4D ground-station tracking observations from an Orekit orbit simulator.
- FDIR: 8D spacecraft telemetry from a coupled linear fault-detection simulator.

Theory references:

- LeWorldModel site: <https://le-wm.github.io/>
- LeWorldModel paper: <https://arxiv.org/abs/2603.19312>
- LeJEPA / SIGReg paper: <https://arxiv.org/abs/2511.08544>

## One-Screen Architecture

The repo has two experiment tracks that share the same learned-world-model core:

```text
simulator -> generated .npz trajectories -> window dataset
          -> ODJEPA encoder/action encoder/predictor
          -> LeWM loss: prediction MSE + lambda * SIGReg
          -> diagnostics: latent prediction, OD decoder probe, FDIR surprise
```

The core tensors are:

```text
obs:    (episode, time, obs_dim)
action: (episode, time, action_dim)
state:  (episode, time, hidden_state_dim)  # saved for diagnostics, not training target
```

Training windows have length:

```text
window = history_size + num_preds
```

Current default:

```text
history_size = 3
num_preds = 1
embed_dim = 192
lambda_SIGReg = 0.09
SIGReg knots = 17
SIGReg random projections = 1024
```

## Mapping To LeWM Theory

The LeWM paper defines:

```text
z_t = encoder(o_t)
z_hat_{t+1} = predictor(z_t, a_t)
L_LeWM = L_pred + lambda * SIGReg(Z)
```

This repo implements the same theory with vector encoders:

```text
OD:
  o_t = [range, azimuth, elevation, range_rate]
  a_t = 3D LVLH acceleration placeholder, currently zero
  z_t in R^192

FDIR:
  o_t = 8D telemetry
  a_t = 4D one-hot recovery-command placeholder, currently nominal
  z_t in R^192
```

The main correspondence is:

| LeWM / LeJEPA concept | Repo implementation |
| --- | --- |
| Pixel encoder, usually ViT | `models/od_encoder.py`, an MLP for vector observations |
| Latent embedding `z_t` | `out["emb"]` in `models/od_jepa.py` |
| Action-conditioned predictor | `module.ARPredictor` with `ConditionalBlock` AdaLN conditioning |
| Prediction loss | `models/od_forward.py`: MSE between predicted and target embeddings |
| Anti-collapse regularizer | `module.SIGReg` |
| End-to-end joint optimization | `train_od.py` and `train_fdir.py` via `stable_pretraining.Module` |
| Latent planning with CEM/MPC | Generic hooks exist in `jepa.py`; this repo mostly uses diagnostics, not CEM control |
| Physical probing | `scripts/train_od_latent_decoder.py` decodes OD latent features to ECI state |
| Violation-of-expectation / surprise | `models/surprise.py` and FDIR board use latent prediction error |

Two important differences from the full LeWM setup:

1. The encoder is not a ViT over images. It is an MLP over low-dimensional
   telemetry vectors.
2. There is no current policy/planning loop for OD or FDIR recovery. The repo is
   focused on learning latent dynamics, probing physical information, and
   detecting faults by surprise.

## Component Interactions

### OD Pipeline

```text
envs/od_env.py
  Orekit orbit propagation
  hidden truth state: [x,y,z,vx,vy,vz]
  emitted observation: [range, az, el, range_rate]

data/generate_dataset.py
  rolls out OdEnv with zero action
  writes data/cache/od_trajectories.npz

od_datasets/od_dataset.py
  z-score normalizes obs/action
  slices fixed-length windows

train_od.py
  loads config/train/od.yaml
  builds ODJEPA from config/train/model/od.yaml
  trains with od_lejepa_forward()

scripts/build_results_board.py
  reads generated data, local stable-pretraining runs, optional checkpoint
  creates data/figures/results_board.html

scripts/train_od_latent_decoder.py
  freezes the latest OD LeWM checkpoint
  trains a supervised probe from latent features to ECI state

scripts/benchmark_od_methods.py
  compares Orekit known-orbit replay, LeWM latent prediction, and optional decoded state probe
```

### FDIR Pipeline

```text
envs/fdir_env.py
  coupled linear telemetry dynamics
  hidden/observed channels are both 8D, with sensor noise on observations
  optional state-level faults: stuck_at, drift, spike

data/generate_fdir.py
  rolls out only nominal episodes
  writes data/cache/fdir_trajectories.npz

train_fdir.py
  reuses the same ODJEPA class, dataset class, and LeWM loss
  swaps dimensions via config/train/model/fdir.yaml

models/surprise.py
  scores faulted trajectories by latent prediction error

scripts/build_fdir_results_board.py
  plots nominal/fault telemetry, latent surprise, z-score baseline, and training curves
```

## Main Math By File

### `envs/od_env.py`

Purpose: Gymnasium orbit-determination environment.

Hidden truth state:

```text
x_t = [r_x, r_y, r_z, v_x, v_y, v_z]
```

Dynamics are not hand-coded here. Orekit's `EcksteinHechlerPropagator` advances
the orbit under zonal gravity terms J2-J6:

```text
x_{t+1} = OrekitPropagate(x_t, dt)
```

The observation is a noisy topocentric ground-station measurement:

```text
o_t = [rho, az, el, rho_dot] + epsilon
epsilon ~ N(0, diag(noise_std^2))
```

where:

```text
rho     = ||r_sat - r_station||
az      = topocentric azimuth, wrapped to [-pi, pi]
el      = topocentric elevation
rho_dot = dot(r_rel, v_rel) / ||r_rel||
```

Actions are accepted as 3D LVLH acceleration placeholders but are currently
ignored by the simulator.

Checks to read:

- State shape and plausible orbital radius/speed: `tests/test_env.py`.
- Measurement bounds/noise: `tests/test_env.py`.
- Approximate energy and angular-momentum boundedness: `_energy_hz()` in
  `tests/test_env.py`.
- Sun-synchronous RAAN precession sanity check: `test_sso_raan_precession()`.

### `envs/orekit_setup.py`

Purpose: one-time JVM and Orekit data setup.

Main invariant:

```text
ensure_orekit() is idempotent within a process
```

It uses a process-global `_ready` flag and lock so repeated environment
construction does not restart the JVM.

### `data/generate_dataset.py`

Purpose: create OD training data.

For each episode:

```text
env = OdEnv(max_steps=episode_len)
obs_0, state_0 = env.reset(seed + episode)
a_t = [0, 0, 0]
roll out episode_len steps
save obs, action, state
```

The saved `state` is diagnostic truth. The LeWM loss trains on `obs` and
`action`, not directly on `state`.

### `envs/fdir_env.py`

Purpose: Gymnasium FDIR telemetry environment.

Hidden state:

```text
x_t = x_star + d_t
```

Nominal deviation dynamics:

```text
d_{t+1} = A d_t + w_t
w_t ~ N(0, diag(Q^2))
```

Observation:

```text
o_t = x_t + eta_t
eta_t ~ N(0, diag(sensor_sigma^2))
```

The dynamics matrix `A` is near diagonal with stable mean reversion:

```text
spectral_radius(A) < 1
```

Important cross-couplings:

```text
solar_array_voltage -> battery_soc
solar_array_voltage -> panel_temp
rw_speed_x/y/z      -> bus_current
```

Faults modify the state, then the deviation is synchronized:

```text
self._d = x - x_star
```

Fault modes:

```text
stuck_at: x[c] = latched onset value
drift:    x[c] = x[c] + drift_rate * active_steps
spike:    x[c] = x[c] + spike_magnitude for spike_duration steps
```

Actions are discrete recovery commands but are currently ignored by the
dynamics.

Checks to read:

- Shape/space contract: `tests/test_fdir.py`.
- `spectral_radius(A) < 1`: `tests/test_fdir.py`.
- Same-seed nominal/fault trajectories match before fault and diverge after:
  `tests/test_fdir.py`.

### `data/generate_fdir.py`

Purpose: create nominal-only FDIR training data.

The generated action is a constant nominal one-hot:

```text
a_t = [1, 0, 0, 0]
```

This keeps FDIR compatible with the same window dataset and action encoder used
by OD, while leaving room for later recovery-policy experiments.

### `od_datasets/od_dataset.py`

Purpose: fit normalizers and produce fixed windows.

Normalizer per key:

```text
mean = average over all episodes and timesteps
std  = std over all episodes and timesteps
std[std < 1e-8] = 1
x_norm = (x - mean) / std
```

Dataset index:

```text
index = [(episode, start) for every valid sliding window]
window data = trajectory[episode, start:start+window]
```

Despite the name, this file is dimension-agnostic and is reused by FDIR.

### `models/od_encoder.py`

Purpose: replace LeWM's ViT pixel encoder with a vector MLP.

Math:

```text
h_t = SiLU(LayerNorm(W_1 o_t + b_1))
z_raw_t = W_2 h_t + b_2
```

Shape:

```text
(B, T, obs_dim) -> (B, T, embed_dim)
```

For OD, `obs_dim = 4`. For FDIR, `obs_dim = 8`.

### `models/od_jepa.py`

Purpose: adapt the generic `JEPA` class from pixel input to vector observation
input.

Encode path:

```text
obs: (B, T, obs_dim)
flat obs: (B*T, obs_dim)
encoder(flat obs) -> z_raw: (B*T, D)
projector(z_raw) -> z: (B*T, D)
reshape -> emb: (B, T, D)
action_encoder(action) -> act_emb: (B, T, D)
```

It inherits `predict()`, rollout, and latent-cost methods from `jepa.py`.

### `jepa.py`

Purpose: generic JEPA wrapper.

Original pixel-oriented `encode()`:

```text
pixels -> encoder -> CLS token -> projector -> emb
action -> action_encoder -> act_emb
```

Prediction:

```text
pred = predictor(emb, act_emb)
pred = pred_proj(pred)
```

Rollout:

```text
given initial history and candidate action sequence:
  encode history
  repeatedly:
    z_hat_next = predict(last history_size embeddings, last actions)[:, -1]
    append z_hat_next
```

Latent planning cost:

```text
cost(candidate) = || z_hat_final - z_goal ||_2^2
```

This matches the LeWM planning theory, but this repo does not currently include
a complete CEM/MPC control script around it.

### `module.py`

Purpose: neural-network building blocks and SIGReg.

#### `SIGReg`

Input:

```text
proj = Z with shape (T, B, D)
```

Each call samples random unit directions:

```text
A in R^(D x num_proj)
||A[:, k]||_2 = 1
```

It projects embeddings to 1D sketches:

```text
y = Z A
```

For integration knots `t`, the empirical characteristic function along each
projection is:

```text
E[cos(t y)] + i E[sin(t y)]
```

The target standard-normal characteristic function is:

```text
phi(t) = exp(-t^2 / 2)
```

The implemented Epps-Pulley-style statistic is approximately:

```text
sum_t weights(t) * (
  (mean_B cos(t y) - phi(t))^2
  + (mean_B sin(t y))^2
)
```

then averaged across projections and time. This is the anti-collapse term from
LeJEPA / LeWM: it pushes the latent distribution toward an isotropic Gaussian
instead of letting all embeddings collapse to a constant.

#### `Attention`

Scaled dot-product causal attention:

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_head) + causal_mask) V
```

#### `ConditionalBlock`

Action-conditioned transformer block using AdaLN-zero:

```text
shift, scale, gate = MLP(action_condition)
x <- x + gate_attn * Attention(modulate(LN(x), shift_attn, scale_attn))
x <- x + gate_mlp  * MLP(modulate(LN(x), shift_mlp, scale_mlp))
```

`adaLN_modulation` is initialized to zero, so action conditioning begins
conservatively.

#### `ARPredictor`

Autoregressive predictor:

```text
x_tilde = z + positional_embedding
z_hat = Transformer(x_tilde, action_condition)
```

The transformer uses causal attention, so a token cannot see future tokens.

#### `Embedder`

Action encoder:

```text
action sequence -> Conv1d(kernel=1) -> MLP -> action embedding
```

#### `MLP`

Generic two-layer projection head used for both encoder output projection and
predictor output projection.

### `models/od_forward.py`

Purpose: the LeWM / LeJEPA training objective for OD and FDIR.

Given a batch window:

```text
emb = encoder/projector(obs)          # (B, T, D)
act_emb = action_encoder(action)      # (B, T, D)
ctx_emb = emb[:, :history_size]
ctx_act = act_emb[:, :history_size]
tgt_emb = emb[:, num_preds:]
pred_emb = predictor(ctx_emb, ctx_act)
```

Prediction loss:

```text
L_pred = mean((pred_emb[:, :m] - tgt_emb[:, :m])^2)
```

Regularization:

```text
L_sigreg = SIGReg(emb.transpose(0, 1))
```

Total:

```text
L = L_pred + lambda * L_sigreg
```

Current defaults make `T = 4`, `history_size = 3`, `num_preds = 1`; therefore
the target starts one step after the first input and the effective prediction is
the next latent embedding.

### `models/surprise.py`

Purpose: FDIR violation-of-expectation score.

For each absolute time `t >= history_size`:

```text
z_hat_t = predictor(z_{t-history_size:t}, a_{t-history_size:t})[:, -1]
surprise_t = || z_hat_t - z_t ||_2^2
```

This is not a supervised fault classifier. It is a latent dynamics residual:
faulted telemetry should become harder for a nominal-only model to predict.

### `train_od.py`

Purpose: full OD Lightning/stable-pretraining training loop.

Flow:

```text
load Hydra config -> fit normalizers -> build OdWindowDataset
-> train/val split -> instantiate model -> build SIGReg
-> stable_pretraining.Module(forward=od_lejepa_forward)
-> Trainer/Manager
```

The math is delegated to `models/od_forward.py`; this file wires the optimizer,
scheduler, logger, and data.

Optimizer:

```text
AdamW(lr=5e-5, weight_decay=1e-3)
LinearWarmupCosineAnnealingLR
gradient_clip_val=1.0
```

### `train_fdir.py`

Purpose: full FDIR training loop.

This is intentionally almost identical to `train_od.py`. The key changes come
from Hydra config:

```text
config/train/fdir.yaml
config/train/data/fdir.yaml
config/train/model/fdir.yaml
```

The FDIR model still uses:

```text
ODJEPA + od_lejepa_forward + SIGReg
```

Only dimensions and dataset path change:

```text
obs_dim: 8
action_dim: 4
dataset: data/cache/fdir_trajectories.npz
```

### `config/train/od.yaml`

Purpose: OD experiment hyperparameters.

Key values:

```text
seed = 3072
embed_dim = 192
history_size = 3
num_preds = 1
SIGReg weight = 0.09
max_epochs = 8
precision = 32
```

### `config/train/fdir.yaml`

Purpose: FDIR experiment hyperparameters.

Same theory values as OD, with:

```text
max_epochs = 16
```

### `config/train/model/od.yaml`

Purpose: OD model assembly.

Main dimensions:

```text
encoder input_dim = 4
action input_dim = 3
embed_dim = 192
predictor depth = 4
predictor heads = 8
predictor mlp_dim = 512
dim_head = 48
dropout = 0.1
```

The model target is `models.od_jepa.ODJEPA`.

### `config/train/model/fdir.yaml`

Purpose: FDIR model assembly.

Same architecture as OD, but:

```text
encoder input_dim = 8
action input_dim = 4
```

### `config/train/data/od.yaml`

Purpose: OD dataset config.

```text
path = data/cache/od_trajectories.npz
window = history_size + num_preds
batch_size = 64
train_split = 0.9
```

### `config/train/data/fdir.yaml`

Purpose: FDIR dataset config.

```text
path = data/cache/fdir_trajectories.npz
window = history_size + num_preds
batch_size = 64
train_split = 0.9
```

### `scripts/train_od_latent_decoder.py`

Purpose: supervised OD latent-state probe.

This is not part of LeWM training. It freezes the latest OD checkpoint and asks:

```text
Do learned latent features contain enough information to recover ECI state?
```

Feature construction:

```text
pred_features   = flatten(predicted latent sequence)
target_features = flatten(target encoder latent sequence)
state_y         = hidden ECI state at the final predicted offset
```

The decoder is an MLP:

```text
decoder(feature) -> [x, y, z, vx, vy, vz]
```

It standardizes both features and state before supervised MSE training.

Metrics:

```text
position_error = || predicted_position - true_position ||_2
velocity_error = || predicted_velocity - true_velocity ||_2
RMSE = sqrt(mean(error^2))
residual_covariance = cov(predicted_state - true_state)
```

Interpretation: this is a probe of latent physical content, similar in spirit to
LeWM's physical latent probing. It is not a production OD estimator.

### `scripts/benchmark_od_methods.py`

Purpose: compare current OD methods and produce JSON for the board.

Methods:

```text
orekit_known_orbit:
  same seeded Orekit propagation as dataset
  physics upper bound, not measurement-only OD

lewm_latent_predictor:
  predicts 192D latent embeddings
  reports latent MSE/cosine and latent residual covariance

lewm_decoded_state:
  optional frozen LeWM + supervised decoder probe
  reports ECI position/velocity errors
```

Important math:

```text
angle residuals are wrapped to [-pi, pi]
latent_mse = mean((z_hat - z_target)^2 over D)
cosine = dot(z_hat, z_target) / (||z_hat|| ||z_target||)
covariance = np.cov(residuals, rowvar=False)
```

The Orekit row is an upper bound because it starts from the same seed/truth
generation convention. A fair classical OD baseline would use only observations,
for example batch least squares or an EKF.

### `scripts/build_results_board.py`

Purpose: build `data/figures/results_board.html` for OD.

Main computations:

- Reads local stable-pretraining run metadata and metrics CSV.
- Loads OD dataset statistics and orbit traces.
- Optionally loads the latest OD checkpoint and computes latent prediction MSE,
  persistence baseline MSE, cosine similarity, and 3D PCA coordinates for
  predicted/target latents.
- Optionally embeds benchmark results from `od_method_benchmark.json`.

PCA math:

```text
center concatenated target/pred latent vectors
SVD(centered) -> principal axes
coords = centered @ top_3_axes
explained = singular_values[:3]^2 / sum(singular_values^2)
```

### `scripts/build_fdir_results_board.py`

Purpose: build `data/figures/fdir_results_board.html`.

Main computations:

- Reads FDIR stable-pretraining runs.
- Loads nominal FDIR dataset.
- Creates a deterministic faulted rollout for visualization.
- Computes raw observation z-score energy:

```text
z_energy_t = mean(((o_t - mean_nominal) / std_nominal)^2)
```

- Optionally loads latest FDIR checkpoint and computes nominal/fault latent
  surprise using `models/surprise.py`.
- Uses a simple threshold sanity check:

```text
passes = post_mean > pre_mean + 3 * pre_std
```

This threshold is a smoke-test diagnostic, not a calibrated detector.

### `spt_compat.py`

Purpose: compatibility shims for the local training environment.

No model math. It handles:

- UTF-8 console setup.
- `pyarrow` alias compatibility for older `datasets`.
- Stubbing `lance` on CPUs without AVX when only non-video vector datasets are
  needed.
- Guarding stable-pretraining signal logging on platforms without Unix signals.

### `tests/test_env.py`

Purpose: OD environment math checks.

Key checks:

- Orekit constants are available after idempotent bootstrap.
- Orbit state has plausible radius and speed.
- Measurements have plausible ranges.
- Noise changes observations.
- Energy and angular momentum stay bounded over a short rollout.
- RAAN precession is roughly sun-synchronous:

```text
0.95 < deg/day < 1.02
```

### `tests/test_model.py`

Purpose: OD model/loss shape checks.

Checks:

- Encoder maps `(B,T,4)` to `(B,T,192)`.
- ODJEPA encode/predict gives consistent latent/action shapes.
- `od_lejepa_forward()` returns finite differentiable losses.
- OD dataset generation and windowing produce expected shapes and normalized
  scales.

### `tests/test_fdir.py`

Purpose: FDIR environment, data, surprise, and mini-detection checks.

Checks:

- Environment emits finite 8D observations/states.
- Action space is `Discrete(4)`.
- `spectral_radius(A) < 1`.
- Same-seed nominal and faulted rollouts diverge only after fault onset.
- FDIR generated data has `(obs_dim=8, action_dim=4)` and one-hot nominal action.
- `surprise_score()` has expected shape.
- A tiny trained model detects a deterministic spike by:

```text
post_surprise_mean > pre_surprise_mean + 3 * pre_surprise_std
```

### `tests/test_train_smoke.py`

Purpose: one-epoch training integration smoke test.

It creates a tiny OD dataset, builds a small ODJEPA, runs the same
`od_lejepa_forward()` loss through `stable_pretraining`, and verifies the
training stack can execute.

## How To Read The Experiments

### OD

The current OD experiment is not yet a full orbit-determination filter. It is a
latent world model trained on partial noisy measurements. The hidden ECI state is
saved so you can inspect whether the latent contains recoverable orbit
information, but the core LeWM objective does not directly supervise ECI state.

What is valid to claim:

```text
The model learns a compact latent dynamics model of the OD measurement stream.
```

What needs a stronger baseline/decoder before claiming:

```text
The model is a production OD estimator.
The model beats classical OD from measurements.
```

Useful next mathematical checks:

- Compare against batch least-squares OD or EKF using only
  `[range, az, el, range_rate]`.
- Add uncertainty metrics: covariance calibration, NIS/NEES, coverage.
- Decode latent history plus explicit time/station geometry to ECI state.

### FDIR

The FDIR experiment is novelty detection under nominal-only training. It learns
nominal latent telemetry dynamics. A fault is detected when the real encoded
telemetry becomes hard to predict from nominal history.

What is valid to claim:

```text
Latent prediction error can act as a surprise signal for off-nominal telemetry.
```

What needs more work:

```text
calibrated anomaly probabilities
fault classification
recovery policy selection
precision/recall across many seeds/fault modes
```

## Common Commands

Generate OD data:

```bash
python data/generate_dataset.py --n-episodes 64 --episode-len 256
```

Train OD:

```bash
python train_od.py
```

Build OD board:

```bash
python scripts/build_results_board.py
```

Train OD latent decoder probe:

```bash
python scripts/train_od_latent_decoder.py
```

Benchmark OD methods:

```bash
python scripts/benchmark_od_methods.py
```

Generate FDIR nominal data:

```bash
python data/generate_fdir.py --n-episodes 64 --episode-len 256
```

Train FDIR:

```bash
python train_fdir.py
```

Build FDIR board:

```bash
python scripts/build_fdir_results_board.py
```

Run tests:

```bash
pytest
```

Depending on the local environment, the project may normally use `uv run` for
these commands.

## Mental Model For Checking The Math

Start at the simulator:

```text
OD:   Does Orekit truth and station measurement geometry make sense?
FDIR: Is A stable, and do faults modify state rather than only observation?
```

Then check the dataset:

```text
Are obs/action/state shapes correct?
Are normalizers fit over all episodes/timesteps?
Does window = history + prediction horizon?
```

Then check the model:

```text
Does encoder produce z_t in R^192?
Does action encoder produce conditioning with matching time dimension?
Does predictor only see history, not target future?
```

Then check the loss:

```text
L_pred = MSE(predicted future latent, encoded future latent)
L_sigreg = characteristic-function normality statistic over embeddings
L_total = L_pred + 0.09 * L_sigreg
```

Then check diagnostics:

```text
OD probe: can latent features decode ECI state?
FDIR surprise: does ||z_hat_t - z_t||^2 rise after faults?
Boards: are plots using the same normalization/window convention as training?
```

That chain is the repo's main interaction pattern.
