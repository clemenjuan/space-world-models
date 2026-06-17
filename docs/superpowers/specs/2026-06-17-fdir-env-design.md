# Step 2 — FDIR Environment (anomaly detection) wired to LeWM

**Date:** 2026-06-17
**Status:** Approved design (pending spec review)
**Scope:** One Gymnasium FDIR environment + a surprise-based anomaly metric reusing the Step 1
LeWM encoder/predictor **with no model-code changes**. Detection only — no recovery policy.

---

## 1. Goal

Add a fault-detection (FDIR) environment to the same repo. Telemetry evolves under a simple
linear nominal model; faults are injected at the **state (dynamics) level** so they propagate
through coupled channels. The LeWM world model (encoder + predictor) is trained
self-supervised on **nominal-only** telemetry, then anomalies are flagged by high prediction
error:

```
surprise_score_t = || z_hat_t - enc(o_t) ||^2
```

No labeled fault data, no extra loss terms, no recovery/action policy.

## 2. Reuse contract (no model changes)

The model code from Step 1 is reused verbatim — only instantiated with new dimensions:

- `models.od_encoder.OdEncoder(in_dim=8, hidden_dim=256, out_dim=192)` — MLP, parameterized
  by `in_dim`, so 8-dim telemetry needs no code change.
- `models.od_jepa.ODJEPA` — `encode()` already reads `info["obs"]` dim-agnostically.
- `module.ARPredictor`, `module.Embedder(input_dim=4)` (one-hot of 4 discrete actions),
  `module.SIGReg`, `module.MLP` — unchanged.
- `models.od_forward.od_lejepa_forward` — unchanged (`pred MSE + 0.09*SIGReg`).
- `embed_dim = 192`, `history_size = 3`, `num_preds = 1` — same as Step 1.
- Training harness (`spt`/Lightning/hydra), `od_datasets` windowing + z-score normalizer
  pattern, and `spt_compat` shims — reused.

**Verified:** `ODJEPA.encode` uses `info["obs"]` with no hardcoded dimension; `OdEncoder`
and `Embedder` take their input dim as a constructor argument. So the only model-related
change is instantiation arguments.

## 3. Repo layout (new files)

```
envs/fdir_env.py          # Gymnasium FDIR env: linear dynamics + state-level fault injection
models/surprise.py        # surprise_score(model, obs_seq, action_seq) -> per-step ||z_hat-enc(o)||^2
data/generate_fdir.py     # roll out N nominal episodes (no faults, action 0) -> .npz
train_fdir.py             # analogue of train_od.py (OdEncoder in_dim=8, Embedder input_dim=4)
config/train/fdir.yaml          # top-level fdir training config
config/train/model/fdir.yaml    # model config (instantiate ODJEPA with in_dim=8)
config/train/data/fdir.yaml     # dataset config (path/window/batch)
tests/test_fdir.py        # env contract + surprise-spike detection smoke test
```

No changes to: `module.py`, `jepa.py`, `models/od_encoder.py`, `models/od_jepa.py`,
`models/od_forward.py`, `od_datasets/od_dataset.py`, `envs/od_env.py`, `spt_compat.py`.

## 4. Environment — `envs/fdir_env.py`

A Gymnasium env mirroring `OdEnv`'s interface (`reset(seed=...)`, `step(action)`, dict
`info`).

### 4.1 Telemetry / hidden state (8-dim)
`[solar_array_voltage, battery_soc, panel_temp, obc_temp, rw_speed_x, rw_speed_y,
rw_speed_z, bus_current]`, all SI-ish floats around realistic setpoints `x*`.

### 4.2 Nominal dynamics
`x_{t+1} - x* = A (x_t - x*) + w`, with `w ~ N(0, Q)`.
- `A`: stable (spectral radius < 1), near-diagonal with deliberate cross-coupling so a
  state-level fault propagates to correlated channels. Documented couplings, e.g.
  solar_array_voltage -> panel_temp, rw_speeds -> bus_current, voltage -> battery_soc.
- `x*`: fixed realistic operating point. `A`, `Q`, `x*` are module constants (configurable).

### 4.3 Fault injection (state-level)
Configured by `fault_mode in {None, "stuck_at", "drift", "spike"}`, `fault_channel`,
`fault_step`, and mode parameters. Activated when `t >= fault_step`:
- `stuck_at`: override the channel's state update to hold its value at onset; coupling
  still feeds neighbours, so they drift away from what a nominal-trained model expects.
- `drift`: add a constant per-step increment to the channel (ramp).
- `spike`: add a transient impulse to the channel for `spike_duration` steps.
Faults modify the **state evolution** (not just the emitted observation).

### 4.4 Observation & action
- `o_t = x_t + sensor_noise`, `sensor_noise ~ N(0, sigma^2)` per channel (configurable;
  small relative to nominal variation).
- `observation_space = Box(shape=(8,))` in raw units.
- `action_space = gym.spaces.Discrete(4)` {0=nominal, 1=isolate_power, 2=safe_mode,
  3=reset_obc}. One-hot encoding to a 4-vector happens at model-feed time (dataset / rollout
  / surprise helper), not inside the env. Action does **not** affect dynamics in Step 2.

### 4.5 `info`
`reset()`/`step()` return `info` with the true 8-dim `state` and a `fault_active` bool.

## 5. Surprise metric — `models/surprise.py`

`surprise_score(model, obs_seq, action_seq) -> Tensor[T - history_size]`:
- `obs_seq`: `(1, T, 8)`; `action_seq`: `(1, T, 4)` one-hot.
- `emb = model.encode({"obs": obs_seq, "action": action_seq})["emb"]`  (B,T,192).
- For each `t` in `[history_size, T)`: predict `z_hat_t` from `emb[:, t-history_size:t]`
  (and matching action window) via `model.predict`, take the last predicted step, and
  compute `|| z_hat_t - emb[:, t] ||^2`.
- Returns per-step scores aligned to absolute timesteps `history_size .. T-1`.
- Pure inference; no gradients, no new loss terms, no change to training.

## 6. Training (reuse harness)

- `data/generate_fdir.py`: roll out N nominal episodes (fault_mode=None, action 0), save
  `obs (N,L,8)`, `action (N,L,)` (or one-hot), `state` to `data/cache/fdir_trajectories.npz`.
- `train_fdir.py` + `config/train/fdir*.yaml` mirror `train_od.py`/`od.yaml` exactly, changing
  only: encoder `in_dim=8`, `action_encoder.input_dim=4`, dataset path, and one-hot action
  handling in the dataset. Self-supervised LeWM loss on nominal telemetry.
- Action column is constant (all nominal) -> its normalizer std is 0 -> guarded to 1 (same
  pattern as Step 1 coasting action). One-hot action handling: store discrete action and
  one-hot it in the dataset `__getitem__`, or store the one-hot directly; either is fine as
  long as the model receives a `(T,4)` float action.

## 7. Smoke test — `tests/test_fdir.py`

1. **Env contract:** rollout >= 150 steps; assert obs `(T,8)` finite, `action_space` is
   `Discrete(4)`; a `stuck_at` fault produces a state trajectory that diverges from the
   nominal (no-fault, same-seed) trajectory after `fault_step`.
2. **Detection:** generate a small nominal dataset -> train briefly (a few epochs, tiny model,
   CPU, seconds) -> run one >=150-step rollout with `stuck_at` fault at step 100 ->
   compute `surprise_score` -> assert the post-fault window mean significantly exceeds the
   pre-fault window mean: `mean(post) > mean(pre) + k * std(pre)` (k configurable, e.g. 3),
   deterministic via fixed seeds. Detection threshold is **relative** (post vs pre), not an
   absolute constant.

## 8. Out of scope

- Recovery policy / action selection (faults are detected, not acted upon).
- Labeled-fault / supervised training.
- Any modification to the world-model code.
- Multi-fault or simultaneous-channel scenarios beyond the three single-channel modes.
- Coupling the discrete action to dynamics (deferred to a future recovery step).
